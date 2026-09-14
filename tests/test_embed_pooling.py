"""Pooling choices in ``embed_sequences`` for causal models (``--pooling``).

A causal model's position t has only seen tokens <= t, so its last real token is
the only state that has read the whole sequence; mean pooling mixes states built
from very different amounts of context (Wang et al. 2026; Skean et al. 2025).

The stand-in model's hidden state at position t is the number t, so each pooling
has a hand-computable answer on a right-padded batch.
"""

import torch
from transformers import BatchEncoding

from protein_benchmark_suite import embed_sequences


class PositionTokenizer:
    """One token per residue, right-padded with id 0."""

    def __call__(self, seqs, return_tensors="pt", padding=True, truncation=True, max_length=None):
        width = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), width, dtype=torch.long)
        mask = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = 1
            mask[i, : len(s)] = 1
        return BatchEncoding({"input_ids": ids, "attention_mask": mask})


class PositionModel(torch.nn.Module):
    """hidden_states[l][b, t] == t + 100 * l; last_hidden_state is the final layer."""

    config = type("Cfg", (), {"model_type": "stand_in"})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True, **kw):
        b, t = input_ids.shape
        pos = torch.arange(t, dtype=torch.float32).view(1, t, 1).expand(b, t, 1)
        states = tuple(pos + 100.0 * layer for layer in range(3))
        return type("Out", (), {"last_hidden_state": states[-1], "hidden_states": states})()


def embed(seqs, **kw):
    return embed_sequences((PositionTokenizer(), PositionModel()), False, seqs, "cpu", **kw)


def test_last_pooling_takes_each_sequences_final_real_token_not_the_padded_end():
    out = embed(["MKV", "M"], pooling="last")
    # final layer = 200 + position; "MKV" ends at position 2, "M" at position 0
    assert out[:, 0].tolist() == [202.0, 200.0]


class LastTokenSentenceTransformer:
    """What sentence-transformers 5 builds for a causal model with no modules.json:
    a Transformer module plus a Pooling module in ``lasttoken`` mode."""

    def __init__(self):
        transformer = type("T", (), {"tokenizer": PositionTokenizer(), "auto_model": PositionModel()})()
        pooling = type("P", (), {"pooling_mode": "lasttoken"})()
        self._modules = {"0": transformer, "1": pooling}

    def encode(self, seqs, **kw):
        import numpy as np
        return np.array([[200.0 + len(s) - 1] for s in seqs])  # final layer at the last token


def test_mean_pooling_is_a_mean_even_when_the_sentence_transformer_pools_last_token():
    st = LastTokenSentenceTransformer()
    out = embed_sequences(st, True, ["MKV", "M"], "cpu", pooling="mean")
    # final layer = 200 + position; "MKV" positions 0,1,2 -> mean 201; "M" -> 200.
    assert out[:, 0].tolist() == [201.0, 200.0]


def test_layer_selects_an_intermediate_hidden_state_under_either_pooling():
    # layer 1 = 100 + position. "MKV" positions 0,1,2 -> mean 101, last 102; "M" -> 100.
    assert embed(["MKV", "M"], pooling="last", layer=1)[:, 0].tolist() == [102.0, 100.0]
    assert embed(["MKV", "M"], pooling="mean", layer=1)[:, 0].tolist() == [101.0, 100.0]
