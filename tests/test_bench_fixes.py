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
    assert lora_target_modules("proteva") == [
        "wq", "wk", "wv", "wo", "attn_gate", "w12", "w3", "ve_first", "ve_last",
    ]  # + ve_first/ve_last value-embedding nn.Embeddings (PEFT-wrappable)
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


def test_fp32_flag_disables_bf16():
    """--fp32 must turn OFF bf16 (regression forwards need fp32 precision — a bf16
    forward is too coarse for the regression head). Without it, keep bf16 if HW
    supports it."""
    import torch
    from argparse import Namespace
    from pathlib import Path
    from _hf_finetune_common import build_training_args

    base = dict(num_train_epochs=1, per_device_train_batch_size=8,
                per_device_eval_batch_size=16, learning_rate=1e-4, weight_decay=0.0,
                logging_steps=50, seed=42, dataloader_num_workers=0, early_stop=False)
    ta_fp32 = build_training_args(Namespace(**base, fp32=True), Path("/tmp/x"))
    assert ta_fp32.bf16 is False
    # default (fp32 unset): bf16 follows hardware support
    ta_def = build_training_args(Namespace(**base, fp32=False), Path("/tmp/x"))
    assert ta_def.bf16 == torch.cuda.is_bf16_supported()


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


# ---------------------------------------------------------------------------
# Bug #1 — no validation split must NOT early-stop / model-select on the test
# set (leak). Only the validation split may drive best-model selection; without
# one, warn + train to the max-epoch cap with load_best_model_at_end OFF.
# ---------------------------------------------------------------------------


def _es_args(**over):
    """A Namespace of the FT common args with --early_stop ON."""
    from argparse import Namespace

    base = dict(num_train_epochs=1, per_device_train_batch_size=8,
                per_device_eval_batch_size=16, learning_rate=1e-4,
                weight_decay=0.0, logging_steps=50, seed=42,
                dataloader_num_workers=0, early_stop=True,
                early_stop_patience=1, fp32=False)
    base.update(over)
    return Namespace(**base)


def test_build_training_args_no_val_disables_load_best_model():
    """early_stop requested but NO validation split -> load_best_model_at_end
    must be OFF. Otherwise the best checkpoint is chosen to MAXIMIZE the metric
    on the only in-loop eval set (test), then that same test metric is reported:
    a selection leak."""
    from pathlib import Path

    from _hf_finetune_common import build_training_args

    ta = build_training_args(_es_args(), Path("/tmp/x"),
                             main_metric="Spearman", eval_available=False)
    assert bool(ta.load_best_model_at_end) is False


def test_build_training_args_val_present_keeps_load_best_model():
    """A val-present task (remote_homology / beta_lactamase_peer / ss3 all
    resolve a validation split at runtime) keeps early-stopping best-model
    selection ON — behavior UNCHANGED."""
    from pathlib import Path

    from _hf_finetune_common import build_training_args

    ta = build_training_args(_es_args(), Path("/tmp/x"),
                             main_metric="Spearman", eval_available=True)
    assert ta.load_best_model_at_end is True


def test_resolve_early_stopping_selects_validation_only():
    """Validation present -> in-loop eval is the VALIDATION split and exactly one
    EarlyStoppingCallback is installed (default-task path, must stay unchanged)."""
    from transformers import EarlyStoppingCallback

    from _hf_finetune_common import resolve_early_stopping

    tokenized = {"train": "TR", "validation": "VAL", "test": "TE"}
    eval_ds, callbacks = resolve_early_stopping(
        tokenized, early_stop=True, task="ss3", patience=1)
    assert eval_ds == "VAL"
    assert len(callbacks) == 1 and isinstance(callbacks[0], EarlyStoppingCallback)


def test_resolve_early_stopping_no_val_warns_skips_and_never_uses_test(caplog):
    """No validation split + --early_stop: must NOT fall back to test (leak) —
    warn, install no EarlyStoppingCallback, and leave the in-loop eval set empty
    so training runs to the max-epoch cap."""
    import logging

    from _hf_finetune_common import resolve_early_stopping

    tokenized = {"train": "TR", "test": "TE"}  # no 'validation' key
    with caplog.at_level(logging.WARNING):
        eval_ds, callbacks = resolve_early_stopping(
            tokenized, early_stop=True, task="remote_homology", patience=1)
    assert eval_ds is None      # NOT the test split
    assert callbacks == []      # no EarlyStoppingCallback
    assert "no validation split for remote_homology" in caplog.text


def test_resolve_early_stopping_off_is_noop():
    """Without --early_stop there is no in-loop eval and no callbacks (old
    fixed-epoch behavior), regardless of which splits exist."""
    from _hf_finetune_common import resolve_early_stopping

    tokenized = {"train": "TR", "validation": "VAL", "test": "TE"}
    eval_ds, callbacks = resolve_early_stopping(
        tokenized, early_stop=False, task="ss3", patience=1)
    assert eval_ds is None and callbacks == []


# ---------------------------------------------------------------------------
# Bug #2 — disprot residue FT must emit its main metric (MCC). Only the
# `disorder` branch computed MCC; the GENERIC branch (used by disprot,
# main_metric=MCC) returned Accuracy/F1_* only -> KeyError on eval_MCC.
# ---------------------------------------------------------------------------


def test_residue_generic_compute_metrics_emits_mcc():
    import numpy as np

    from finetune_residue import _build_compute_metrics

    cm = _build_compute_metrics("disprot", ["0", "1"])  # generic branch
    # 1 example, 4 residues, 2 classes; argmax(preds) == labels -> MCC == 1.0
    predictions = np.array([[[3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [-3.0, 3.0]]])
    labels = np.array([[0, 1, 0, 1]])
    out = cm((predictions, labels))
    assert "MCC" in out
    assert isinstance(out["MCC"], float) and -1.0 <= out["MCC"] <= 1.0


# ---------------------------------------------------------------------------
# Bug #4 — regression Spearman NaN (constant / NaN predictions = a collapsed
# run) must WARN, not silently return 0.0.
# ---------------------------------------------------------------------------


def test_regression_spearman_nan_warns_and_returns_zero(caplog):
    import logging
    import types

    import numpy as np

    from finetune_sequence import _build_compute_metrics

    cfg = types.SimpleNamespace(problem_type="regression", name="dummy_reg")
    cm = _build_compute_metrics(cfg)
    preds = np.full(10, 0.5)              # constant -> Spearman undefined (NaN)
    labels = np.arange(10, dtype=float)
    with caplog.at_level(logging.WARNING):
        out = cm((preds, labels))
    assert out["Spearman"] == 0.0        # aggregation-compatible sentinel kept
    assert "spearman" in caplog.text.lower()  # ...but the collapse is surfaced


def test_timed_fit_is_one_helper_used_by_every_probe_path():
    """Four copy-pasted perf_counter blocks meant four places to keep in step."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from protein_benchmark_suite import timed_fit

    X = np.random.RandomState(0).randn(50, 4)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression()
    seconds = timed_fit(model, X, y)
    assert seconds > 0
    assert hasattr(model, "coef_"), "timed_fit must actually fit the model"


def test_probe_fit_seconds_is_a_total_not_a_per_fold_mean():
    """CV rows route through _aggregate_cv_metrics, which means over folds. One
    column must not carry two units."""
    from protein_benchmark_suite import _aggregate_cv_metrics

    folds = [
        {"Accuracy": 0.8, "ProbeFitSec": 2.0},
        {"Accuracy": 0.6, "ProbeFitSec": 4.0},
    ]
    out = _aggregate_cv_metrics(folds)
    assert out["Accuracy"] == 0.7  # metrics still average
    assert out["ProbeFitSec"] == 6.0  # time sums


def test_classification_metrics_bootstrap_is_capped_on_huge_arrays():
    """Residue tasks resample 150k-600k rows; an uncapped bootstrap costs minutes
    per task against a ~60s probe fit."""
    import numpy as np

    from protein_benchmark_suite import bootstrap_draws_for

    assert bootstrap_draws_for(1000, n_rows=500) == 1000
    assert bootstrap_draws_for(1000, n_rows=500_000) < 1000
    assert bootstrap_draws_for(0, n_rows=500_000) == 0
