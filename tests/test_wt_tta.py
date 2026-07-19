"""Unit tests for WT test-time training (TTT) — CPU only, no network.

These exercise the architecture-agnostic TTT engine plus the AMPLIFY adapter via
a tiny AMPLIFY-shaped stub model (config.model_type == "AMPLIFY"), so the real
``resolve_mlm_head`` AMPLIFY branch is tested without downloading AMPLIFY_120M.
"""

import copy
import math

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput


# --------------------------------------------------------------------------- #
# Tiny AMPLIFY-shaped stub: mirrors the attribute names the AMPLIFY adapter
# relies on — ``transformer_encoder`` (ModuleList of blocks), ``layer_norm_2``
# (final norm), ``decoder`` (MLM head), ``embed`` token embedding — and a
# forward returning ``MaskedLMOutput(logits=..., hidden_states=...)``.
# --------------------------------------------------------------------------- #
class _StubConfig:
    model_type = "AMPLIFY"

    def __init__(self, vocab_size, hidden_size, num_attention_heads, max_length):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_length = max_length


class _StubBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.ffn = nn.Linear(hidden, hidden)

    def forward(self, x, attention_mask=None, *args, **kwargs):
        # AMPLIFY blocks return (hidden, attention) — mirror that contract.
        return x + torch.tanh(self.ffn(x)), None


class StubAMPLIFY(nn.Module):
    """Minimal AMPLIFY look-alike for TTT unit tests."""

    def __init__(self, vocab_size=12, hidden_size=16, n_layers=3, num_heads=2,
                 max_length=64):
        super().__init__()
        self.config = _StubConfig(vocab_size, hidden_size, num_heads, max_length)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.transformer_encoder = nn.ModuleList(
            [_StubBlock(hidden_size) for _ in range(n_layers)]
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                return_dict=True, **kwargs):
        x = self.embed(input_ids)
        hidden_states = []
        for blk in self.transformer_encoder:
            x, _ = blk(x, attention_mask)
            if output_hidden_states:
                hidden_states.append(x)
        logits = self.decoder(self.layer_norm_2(x))
        return MaskedLMOutput(logits=logits, hidden_states=hidden_states or None)


class _StubTokenizer:
    """Whitespace-free char tokenizer good enough for TTT unit tests."""

    def __init__(self):
        # 0=pad 1=cls 2=eos 3=mask, then a few "amino acids"
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3
        self._vocab = {c: i + 4 for i, c in enumerate("ACDEFGH")}
        self.all_special_ids = [0, 1, 2, 3]

    def __call__(self, seqs, return_tensors=None, padding=True, truncation=True,
                 max_length=64, add_special_tokens=True):
        if isinstance(seqs, str):
            seqs = [seqs]
        rows = []
        for s in seqs:
            ids = [self.cls_token_id] + [self._vocab[c] for c in s] + [self.eos_token_id]
            rows.append(ids[:max_length])
        maxlen = max(len(r) for r in rows)
        ids = [r + [self.pad_token_id] * (maxlen - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (maxlen - len(r)) for r in rows]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(mask),
            }
        return {"input_ids": ids, "attention_mask": mask}


def _build_stub():
    torch.manual_seed(0)
    return StubAMPLIFY(), _StubTokenizer()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_resolve_mlm_head_finds_amplify_blocks_and_head():
    from wt_test_time_training import resolve_mlm_head

    model, tok = _build_stub()
    refs = resolve_mlm_head(model, tok)

    assert refs.encoder_blocks is model.transformer_encoder
    assert refs.head_module is model.decoder
    assert refs.final_norm is model.layer_norm_2
    assert refs.mask_token_id == tok.mask_token_id
    assert callable(refs.forward_logits)


def test_resolve_raises_clearly_on_headless_model():
    from wt_test_time_training import resolve_mlm_head

    model, tok = _build_stub()
    del model.decoder  # simulate an encoder-only export with no MLM head
    with pytest.raises(ValueError, match="decoder"):
        resolve_mlm_head(model, tok)


# --------------------------------------------------------------------------- #
# Vanilla ESM-C (Synthyra ESMplusplusForMaskedLM, config.model_type ==
# "ESMplusplus"): a standard HF MaskedLM whose forward returns .logits. The
# resolver must find its MLM head so ProteinGym MLM zero-shot can score it —
# without a branch it raised ValueError and every vanilla zero-shot shard died.
# --------------------------------------------------------------------------- #
class _StubESMConfig:
    model_type = "ESMplusplus"

    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class StubESMplusplus(nn.Module):
    """Minimal Synthyra-ESM++ look-alike: forward returns MaskedLMOutput.logits."""

    def __init__(self, vocab_size=12, hidden_size=16, n_layers=2):
        super().__init__()
        self.config = _StubESMConfig(vocab_size, hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        )
        self.sequence_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = x + torch.tanh(blk(x))
        return MaskedLMOutput(logits=self.sequence_head(x))


def test_resolve_mlm_head_supports_esmplusplus():
    from wt_test_time_training import resolve_mlm_head

    model, tok = StubESMplusplus(), _StubTokenizer()
    refs = resolve_mlm_head(model, tok)  # must NOT raise

    assert refs.mask_token_id == tok.mask_token_id
    assert callable(refs.forward_logits)
    ids = torch.tensor([[1, 4, 5, 6, 2]])
    mask = torch.ones_like(ids)
    lg = refs.forward_logits(ids, mask)
    assert lg.shape == (1, ids.shape[1], model.config.vocab_size)


def _pooled_embedding(model, tok, seq):
    """Mean-pooled last-hidden-state for the stub (mirrors the readout signal)."""
    enc = tok(seq, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    output_hidden_states=True, return_dict=True)
        h = model.layer_norm_2(out.hidden_states[-1])
        m = enc["attention_mask"].unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


WT = "ACDEFGHACDEFGH"


def test_adapt_changes_top_layers_and_freezes_body():
    from wt_test_time_training import (
        resolve_mlm_head, snapshot_trainable, adapt_to_wt, TTAConfig,
    )

    model, tok = _build_stub()
    refs = resolve_mlm_head(model, tok)
    cfg = TTAConfig(iters=8, n_layers=1, lr=0.5, seed=0, train_head=True)

    frozen_before = model.transformer_encoder[0].ffn.weight.detach().clone()
    top_before = model.transformer_encoder[-1].ffn.weight.detach().clone()

    snap = snapshot_trainable(refs, cfg)
    adapt_to_wt(model, refs, WT, tok, cfg, device="cpu", pristine_snapshot=snap)

    # Frozen body (block 0, with n_layers=1 only block -1 trains) is bit-identical.
    assert torch.equal(model.transformer_encoder[0].ffn.weight, frozen_before)
    # Top block actually moved.
    assert not torch.equal(model.transformer_encoder[-1].ffn.weight, top_before)


def test_restore_is_bit_identical():
    from wt_test_time_training import (
        resolve_mlm_head, snapshot_trainable, adapt_to_wt, TTAConfig,
    )

    model, tok = _build_stub()
    refs = resolve_mlm_head(model, tok)
    cfg = TTAConfig(iters=8, n_layers=2, lr=0.5, seed=1)

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    snap = snapshot_trainable(refs, cfg)
    handle = adapt_to_wt(model, refs, WT, tok, cfg, device="cpu", pristine_snapshot=snap)
    handle.restore()

    after = model.state_dict()
    for k, v in before.items():
        assert torch.equal(after[k], v), f"param {k} not restored bit-identically"


def test_adapt_moves_the_embedding_readout():
    from wt_test_time_training import (
        resolve_mlm_head, snapshot_trainable, adapt_to_wt, TTAConfig,
    )

    model, tok = _build_stub()
    refs = resolve_mlm_head(model, tok)
    cfg = TTAConfig(iters=10, n_layers=1, lr=0.5, seed=0)

    emb_before = _pooled_embedding(model, tok, WT)
    snap = snapshot_trainable(refs, cfg)
    adapt_to_wt(model, refs, WT, tok, cfg, device="cpu", pristine_snapshot=snap)
    emb_after = _pooled_embedding(model, tok, WT)

    assert not torch.allclose(emb_before, emb_after, atol=1e-5)


def test_adapt_is_deterministic_with_seed():
    from wt_test_time_training import (
        resolve_mlm_head, snapshot_trainable, adapt_to_wt, TTAConfig,
    )

    model, tok = _build_stub()
    refs = resolve_mlm_head(model, tok)
    cfg = TTAConfig(iters=6, n_layers=1, lr=0.5, seed=42)
    snap = snapshot_trainable(refs, cfg)

    adapt_to_wt(model, refs, WT, tok, cfg, device="cpu", pristine_snapshot=snap)
    run1 = model.transformer_encoder[-1].ffn.weight.detach().clone()

    # restore to pristine, run again with the same seed
    for p, saved in snap:
        p.data.copy_(saved)
    adapt_to_wt(model, refs, WT, tok, cfg, device="cpu", pristine_snapshot=snap)
    run2 = model.transformer_encoder[-1].ffn.weight.detach().clone()

    assert torch.equal(run1, run2)


def test_losses_are_finite_and_one_per_step():
    from wt_test_time_training import (
        resolve_mlm_head, snapshot_trainable, adapt_to_wt, TTAConfig,
    )

    model, tok = _build_stub()
    refs = resolve_mlm_head(model, tok)
    cfg = TTAConfig(iters=7, n_layers=2, lr=0.3, seed=0)
    snap = snapshot_trainable(refs, cfg)
    handle = adapt_to_wt(model, refs, WT, tok, cfg, device="cpu", pristine_snapshot=snap)

    assert len(handle.losses) == cfg.iters
    assert all(math.isfinite(x) for x in handle.losses)


def test_run_tta_zeroshot_scores_groups_and_restores_model():
    import numpy as np
    from wt_test_time_training import resolve_mlm_head, run_tta_zeroshot, TTAConfig

    model, tok = _build_stub()
    refs = resolve_mlm_head(model, tok)
    cfg = TTAConfig(iters=4, n_layers=1, lr=0.3, seed=0)

    wtA, wtB = "ACDEFGHACDEFGH", "GHACDEFGHACDEF"
    mutants = ["ACDEFGHACDEFGG", "ACDEFGHACDEFGA", "ACDEFGHACDEFGC",
               "GHACDEFGHACDEA", "GHACDEFGHACDEC", "GHACDEFGHACDEG",
               "ACDEFGHACDEFGD"]
    wts =     [wtA, wtA, wtA, wtB, wtB, wtB, "ACDEFGHACDEFGD"]
    groups = np.array(["A", "A", "A", "B", "B", "B", "C"])
    labels = np.array([0.1, 0.5, 0.9, 0.2, 0.4, 0.8, 0.3])

    # embed_fn closes over the (adapted) model — mirrors the suite's embed_sequences.
    def embed_fn(seqs):
        return np.stack([_pooled_embedding(model, tok, s).squeeze(0).numpy()
                         for s in seqs])

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scores = run_tta_zeroshot(
        model, refs, tok, mutants, wts, groups, labels,
        problem_type="regression", embed_fn=embed_fn, tta_cfg=cfg, device="cpu",
    )

    # Groups A and B have >=2 members; C (1 member) is skipped.
    assert len(scores) == 2
    assert all(math.isfinite(s) for s in scores)
    # Model is restored bit-identically after the whole sweep.
    after = model.state_dict()
    for k, v in before.items():
        assert torch.equal(after[k], v), f"{k} not restored after run_tta_zeroshot"
