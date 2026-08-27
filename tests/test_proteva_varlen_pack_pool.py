"""CPU tests for the proteva fa2-varlen pack -> segment-mean-pool helpers.

These exercise the deterministic tensor reshaping that the bench's unpadded
varlen embedding path relies on:

  * ``_pack_token_id_rows``: list of per-sequence token-id arrays ->
    ``(1, total_tokens)`` packed ``input_ids`` + ``cu_seqlens`` + per-segment
    ``position_ids`` + ``max_seqlen`` (the fa2-varlen layout the model was
    trained with, matching ``plm.hf.collator.ProteinPackedCollator``).
  * ``_segment_mean_pool``: ``(1, total_tokens, H)`` hidden states +
    ``cu_seqlens`` -> ``(num_segments, H)`` per-sequence mean-pooled embeddings.

The flash-attn forward between these two steps is GPU-validated separately;
here we only pin the padding-free reshape math.
"""

import types

import numpy as np
import torch
from transformers import BatchEncoding

from protein_benchmark_suite import (
    _pack_token_id_rows,
    _segment_mean_pool,
    embed_sequences,
)


def test_pack_token_id_rows_layout():
    rows = [
        np.array([5, 6, 7], dtype=np.int64),
        np.array([8, 9], dtype=np.int64),
        np.array([10, 11, 12, 13], dtype=np.int64),
    ]

    packed = _pack_token_id_rows(rows)

    # input_ids: (1, total_tokens), concatenated in row order, no padding.
    assert packed["input_ids"].shape == (1, 9)
    assert packed["input_ids"].dtype == torch.long
    assert packed["input_ids"][0].tolist() == [5, 6, 7, 8, 9, 10, 11, 12, 13]

    # cu_seqlens: int32 prefix-sum [0, 3, 5, 9], shape (num_segments + 1,).
    cu = packed["cu_seqlens_q"]
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 3, 5, 9]
    # k mirrors q for self-attention varlen.
    assert torch.equal(packed["cu_seqlens_k"], cu)

    # position_ids: per-segment arange, concatenated -> (1, total_tokens).
    assert packed["position_ids"].shape == (1, 9)
    assert packed["position_ids"][0].tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]

    # max_seqlen: longest segment length (the 4-token row).
    assert packed["max_seqlen_q"] == 4
    assert packed["max_seqlen_k"] == 4

    # attention_mask: all-ones (1, total_tokens) — packed varlen has no pads.
    assert packed["attention_mask"].shape == (1, 9)
    assert packed["attention_mask"][0].tolist() == [1] * 9


def test_pack_token_id_rows_single_row():
    rows = [np.array([1, 2, 3, 4], dtype=np.int64)]
    packed = _pack_token_id_rows(rows)
    assert packed["input_ids"][0].tolist() == [1, 2, 3, 4]
    assert packed["cu_seqlens_q"].tolist() == [0, 4]
    assert packed["position_ids"][0].tolist() == [0, 1, 2, 3]
    assert packed["max_seqlen_q"] == 4


def test_segment_mean_pool_back_to_per_sequence():
    # total_tokens = 9, H = 2; three segments of length 3, 2, 4.
    cu = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    hidden = torch.tensor(
        [
            # segment 0 (len 3) -> mean [2, 20]
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            # segment 1 (len 2) -> mean [5.5, 55]
            [5.0, 50.0],
            [6.0, 60.0],
            # segment 2 (len 4) -> mean [10.5, 105]
            [9.0, 90.0],
            [10.0, 100.0],
            [11.0, 110.0],
            [12.0, 120.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)  # (1, 9, 2)

    pooled = _segment_mean_pool(hidden, cu)

    assert pooled.shape == (3, 2)
    expected = torch.tensor(
        [[2.0, 20.0], [5.5, 55.0], [10.5, 105.0]], dtype=torch.float32
    )
    assert torch.allclose(pooled, expected), pooled


def test_segment_mean_pool_casts_bf16_to_float():
    # Model forwards in bf16; pooled output must come back as float for the probe.
    cu = torch.tensor([0, 2], dtype=torch.int32)
    hidden = torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=torch.bfloat16).unsqueeze(0)
    pooled = _segment_mean_pool(hidden, cu)
    assert pooled.dtype == torch.float32
    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled, torch.tensor([[2.0, 4.0]]))


def test_pack_then_pool_roundtrip_preserves_order_and_count():
    # End-to-end reshape contract: B rows in -> pack -> identity "forward" ->
    # pool -> B rows out, in the same order, with per-row token means.
    rows = [
        np.array([2, 4], dtype=np.int64),       # mean over a fake H below
        np.array([6, 8, 10], dtype=np.int64),
        np.array([12], dtype=np.int64),
    ]
    packed = _pack_token_id_rows(rows)

    # Fake encoder: embed each token id as [id, id*2] so we can predict means.
    ids = packed["input_ids"][0]
    hidden = torch.stack([ids.float(), ids.float() * 2.0], dim=-1).unsqueeze(0)

    pooled = _segment_mean_pool(hidden, packed["cu_seqlens_q"])

    assert pooled.shape == (3, 2)
    # Row means of the token ids: 3, 8, 12.
    assert torch.allclose(pooled[:, 0], torch.tensor([3.0, 8.0, 12.0]))
    assert torch.allclose(pooled[:, 1], torch.tensor([6.0, 16.0, 24.0]))


class _ToyTokenizer:
    """Minimal padded tokenizer for the Proteva dense-path regression test."""

    def __call__(self, sequences, **kwargs):
        del kwargs
        rows = [
            [1, *(3 + ord(char) % 20 for char in sequence), 2] for sequence in sequences
        ]
        width = max(map(len, rows))
        input_ids = [row + [0] * (width - len(row)) for row in rows]
        attention_mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return BatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )


class _ToyDenseProteva(torch.nn.Module):
    """Proteva-shaped model whose hidden states are deterministic token features."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = types.SimpleNamespace(model_type="proteva")
        self.encoder = types.SimpleNamespace(
            config=types.SimpleNamespace(flash_attn_mode="off")
        )
        self.batch_sizes = []

    def forward(self, input_ids, attention_mask, return_dict=True, **kwargs):
        del attention_mask, kwargs, return_dict
        self.batch_sizes.append(len(input_ids))
        values = input_ids.float()
        hidden = torch.stack([values, values * 2], dim=-1)
        return types.SimpleNamespace(last_hidden_state=hidden)


def test_dense_fp32_proteva_batches_and_matches_single_sequence_forwards():
    """Flash-off Proteva inference uses padded batches without changing embeddings."""

    sequences = ["ACDEFG", "H", "KLM", "NPQRST", "VW"]
    tokenizer = _ToyTokenizer()
    batched_model = _ToyDenseProteva()
    batched = embed_sequences(
        (tokenizer, batched_model),
        False,
        sequences,
        device="cpu",
        batch_size=2,
    )
    single_model = _ToyDenseProteva()
    single = embed_sequences(
        (tokenizer, single_model),
        False,
        sequences,
        device="cpu",
        batch_size=1,
    )

    assert batched_model.batch_sizes == [2, 2, 1]
    assert single_model.batch_sizes == [1] * len(sequences)
    assert np.array_equal(batched, single)


def test_legacy_flash_off_mode_reproduces_singleton_packed_execution():
    """Historical SCOPe rows can explicitly select their original protocol."""

    sequences = ["ACDEFG", "H", "KLM"]
    model = _ToyDenseProteva()
    embeddings = embed_sequences(
        (_ToyTokenizer(), model),
        False,
        sequences,
        device="cpu",
        batch_size=3,
        proteva_flash_off_mode="legacy_single_packed",
    )

    assert embeddings.shape == (3, 2)
    assert model.batch_sizes == [1, 1, 1]
