"""Regression tests for the 2026-06-16 benchmark-harness fixes.

Each test pins one bug that produced silently-wrong or missing benchmark
numbers across the AMPLIFY / step0 / epoch1 comparison:

* ``score_substitution`` indexed past a truncated log-prob table -> IndexError
  crashed the whole ProteinGym MLM run (now: skip the over-long variant).
* ``AMPLIFY_LORA_TARGETS`` named ``Wqkv``/``fc1``/``fc2`` which this AMPLIFY rev
  does not expose -> LoRA adapted 0 attention/FFN Linears (silent no-op).
* ``_dedup_key`` keyed on ``date`` -> a re-run on the same day with corrected
  settings could not overwrite the stale row; different models/notes also
  collided across days. Now keyed on (model, notes, task, probe, split, metric).
* ``ss3`` main_metric was ``Accuracy`` while the linear probe reports
  ``F1_Macro`` -> LoRA-vs-probe comparison mixed metrics.

Pure-function / config tests — no GPU, no model download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))


# ---------------------------------------------------------------------------
# proteingym_mlm_zeroshot.score_substitution — truncation guard
# ---------------------------------------------------------------------------


def _logP(rows: int, vocab: int = 25):
    import torch

    # deterministic, finite table so a real score is non-zero
    return torch.arange(rows * vocab, dtype=torch.float32).reshape(rows, vocab)


def test_score_substitution_skips_variant_longer_than_table():
    """WT longer than the (truncated) logP table must return None, not crash."""
    from proteingym_mlm_zeroshot import score_substitution

    aa2id = {"A": 0, "K": 1, "M": 2}
    wt = "AKM" * 600  # 1800 residues
    mut = list(wt)
    mut[0] = "K"
    mut = "".join(mut)
    logP = _logP(rows=1022)  # tokenizer truncated to 1022 < 1800

    assert score_substitution(wt, mut, logP, aa2id) is None


def test_score_substitution_scores_within_bounds():
    """A variant that fits in the table returns a finite float (not None)."""
    from proteingym_mlm_zeroshot import score_substitution

    aa2id = {"A": 0, "K": 1, "M": 2}
    wt = "AKM"
    mut = "KKM"  # single substitution at position 0
    logP = _logP(rows=3)

    s = score_substitution(wt, mut, logP, aa2id)
    assert s is not None
    assert isinstance(s, float)


def test_score_substitution_indel_returns_none():
    """Length mismatch (indel) is still skipped (pre-existing behavior intact)."""
    from proteingym_mlm_zeroshot import score_substitution

    assert score_substitution("AKM", "AK", _logP(3), {"A": 0, "K": 1, "M": 2}) is None


# ---------------------------------------------------------------------------
# _hf_finetune_common — AMPLIFY LoRA target modules
# ---------------------------------------------------------------------------


def test_amplify_lora_targets_are_separate_projections():
    """This AMPLIFY rev exposes separate q/k/v/wo + SwiGLU w12/w3; the fused
    ``Wqkv`` and the older ``fc1``/``fc2`` names must NOT appear (they match
    nothing -> LoRA silently adapts no attention/FFN Linears)."""
    from _hf_finetune_common import lora_target_modules

    targets = lora_target_modules("amplify")
    assert targets == ["q", "k", "v", "wo", "w12", "w3"]
    for stale in ("Wqkv", "fc1", "fc2"):
        assert stale not in targets


def test_lora_target_modules_dispatch_by_family():
    """Each known family resolves to its own list; unknown -> PEFT all-linear."""
    from _hf_finetune_common import lora_target_modules

    # Proteva: ALL body linears (QLoRA best practice — not just attention).
    # attn_gate is the --head-gate Linear; omitting it left a body Linear
    # un-adapted. decoder/*_head (MLM + aux heads) stay excluded.
    assert lora_target_modules("proteva") == ["wq", "wk", "wv", "wo", "attn_gate", "w12", "w3"]
    assert lora_target_modules("esm") == ["query", "key", "value", "dense"]
    assert lora_target_modules("AMPLIFY") == ["q", "k", "v", "wo", "w12", "w3"]  # case-insensitive
    assert lora_target_modules("something_else") == "all-linear"


def test_proteva_lora_targets_cover_all_body_linears():
    """Regression: the Proteva target list must include the attention gate.

    Enumerating the encoder block (plm/model.py EncoderBlock) the body Linears
    are wq/wk/wv/wo + attn_gate (attention) and w12/w3 (SwiGLU FFN). LoRA should
    adapt every one (modern PEFT best practice), excluding only the MLM decoder
    + pretraining aux heads (those are not in the list)."""
    from _hf_finetune_common import PROTEVA_LORA_TARGETS

    for body_linear in ("wq", "wk", "wv", "wo", "attn_gate", "w12", "w3"):
        assert body_linear in PROTEVA_LORA_TARGETS, f"{body_linear} missing"
    for excluded in ("decoder", "di3_head", "cons_head", "plddt_head"):
        assert excluded not in PROTEVA_LORA_TARGETS


# ---------------------------------------------------------------------------
# collect_bench_results._dedup_key — notes-aware, date-independent
# ---------------------------------------------------------------------------


def test_dedup_key_same_notes_overwrites_across_dates():
    """A re-run (different date) with the SAME model+notes+task+metric collapses
    to one key, so the newer row overwrites the stale one."""
    from collect_bench_results import _dedup_key

    older = {"model": "m", "notes": "epoch1", "task": "ss3", "probe_type": "lora",
             "split": "test", "metric_name": "F1_Macro", "date": "2026-06-15"}
    newer = {**older, "date": "2026-06-16"}
    assert _dedup_key(older) == _dedup_key(newer)


def test_dedup_key_distinguishes_notes():
    """Different notes (e.g. real run vs a verify smoke) stay separate rows."""
    from collect_bench_results import _dedup_key

    real = {"model": "m", "notes": "step0", "task": "solubility", "probe_type": "lora",
            "split": "test", "metric_name": "AUC"}
    smoke = {**real, "notes": "verify-step0-lora"}
    assert _dedup_key(real) != _dedup_key(smoke)


# ---------------------------------------------------------------------------
# benchmark_tasks — ss3 metric matches the linear-probe metric
# ---------------------------------------------------------------------------


def test_ss3_main_metric_is_f1_macro():
    """ss3 must report F1_Macro so LoRA and linear-probe rows are comparable."""
    from benchmark_tasks import TASKS

    assert TASKS["ss3"].main_metric == "F1_Macro"


def test_early_stop_metric_direction():
    """Early-stopping best-model direction must follow the metric, not be hardcoded.

    Regression task `meltome` uses MSE (lower-is-better) — greater_is_better must
    be False for it, else early stopping keeps the WORST epoch. Higher-is-better
    metrics (Spearman/F1_Macro/AUC/Accuracy/Recall@10/MCC) must be True."""
    from _hf_finetune_common import metric_greater_is_better

    for lower in ("MSE", "mae", "RMSE", "loss"):
        assert metric_greater_is_better(lower) is False, lower
    for higher in ("Spearman", "F1_Macro", "AUC", "Accuracy", "Recall@10", "MCC"):
        assert metric_greater_is_better(higher) is True, higher
    # None / unknown -> treat as higher-is-better (safe default for our metrics)
    assert metric_greater_is_better(None) is True
