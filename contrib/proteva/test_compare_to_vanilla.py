"""Tests for the delta-vs-vanilla benchmark comparison view.

Pivots the long unified CSV (bench_results_all.csv) into one row per
(probe_type, task, metric) carrying the baseline (vanilla AMPLIFY) value,
each other model's value, and the delta vs the baseline — so linear-probe
and LoRA results can be read against the untrained base model at a glance.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from compare_to_vanilla import build_comparison  # noqa: E402


def _row(model, task, probe, split, metric, value):
    return {"model": model, "task": task, "probe_type": probe, "split": split,
            "metric_name": metric, "metric_value": value}


def test_delta_vs_vanilla_basic():
    rows = [
        _row("chandar-lab/AMPLIFY_120M", "Solubility (DeepSol)", "linear", "test", "AUC", "0.685"),
        _row("/ckpts/arch_warminit_step0", "Solubility (DeepSol)", "linear", "test", "AUC", "0.685"),
        _row("/ckpts/stage2_epoch1", "Solubility (DeepSol)", "linear", "test", "AUC", "0.746"),
    ]
    out = build_comparison(rows, baseline_substr="AMPLIFY", split="test")
    assert len(out) == 1
    rec = out[0]
    assert rec["task"] == "Solubility (DeepSol)"
    assert rec["probe_type"] == "linear"
    assert rec["metric"] == "AUC"
    assert rec["baseline"] == 0.685
    # epoch1 value + delta vs vanilla
    assert rec["models"]["/ckpts/stage2_epoch1"] == 0.746
    assert abs(rec["deltas"]["/ckpts/stage2_epoch1"] - 0.061) < 1e-9
    assert abs(rec["deltas"]["/ckpts/arch_warminit_step0"] - 0.0) < 1e-9


def test_split_filter_and_empty_values_skipped():
    rows = [
        _row("AMPLIFY_120M", "ss3", "linear", "test", "F1_Macro", "0.77"),
        _row("epoch1", "ss3", "linear", "test", "F1_Macro", "0.80"),
        _row("epoch1", "ss3", "linear", "validation", "F1_Macro", "0.79"),  # other split
        _row("epoch1", "ss3", "linear", "test", "F1_Macro", ""),            # empty -> skip
    ]
    out = build_comparison(rows, baseline_substr="AMPLIFY", split="test")
    assert len(out) == 1
    assert out[0]["models"]["epoch1"] == 0.80


def test_lora_rows_have_no_baseline_when_amplify_absent():
    """AMPLIFY can't LoRA-FT, so LoRA rows have no vanilla baseline -> delta None,
    but the row is still emitted so the FT models are visible."""
    rows = [
        _row("step0", "remote_homology", "lora", "test", "F1_Macro", "0.16"),
        _row("epoch1", "remote_homology", "lora", "test", "F1_Macro", "0.20"),
    ]
    out = build_comparison(rows, baseline_substr="AMPLIFY", split="test")
    assert len(out) == 1
    rec = out[0]
    assert rec["baseline"] is None
    assert rec["deltas"]["epoch1"] is None
    assert rec["models"]["epoch1"] == 0.20


def test_sorted_by_probe_then_task():
    rows = [
        _row("AMPLIFY_120M", "Zeta", "linear", "test", "AUC", "0.5"),
        _row("AMPLIFY_120M", "Alpha", "linear", "test", "AUC", "0.5"),
        _row("step0", "Beta", "lora", "test", "AUC", "0.5"),
    ]
    out = build_comparison(rows, baseline_substr="AMPLIFY", split="test")
    keys = [(r["probe_type"], r["task"]) for r in out]
    assert keys == sorted(keys)
