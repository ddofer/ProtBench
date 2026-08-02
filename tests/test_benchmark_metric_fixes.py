"""Metric-correctness fixes for downstream benchmark tasks (audit 2026-07-19).

- EC is genuinely MULTILABEL (a protein carries several comma-separated EC
  numbers). It was declared multiclass, so the label parser kept each comma
  string ('130,270') as its OWN class -> hundreds of singleton "powerset"
  classes -> macro-F1 structurally deflated + jittery. Fix: multilabel +
  F1_Micro headline (macro over singleton combos is meaningless).
- Conservation grades 1-9 are ORDINAL; nominal macro-F1 scores off-by-one the
  same as off-by-eight. FLIP reports Spearman. The residue probe must emit
  Spearman for ordinal tasks.
- Remote Homology has 1195 imbalanced folds; macro-F1 is dominated by 0-F1
  singletons (high variance). Top-1 Accuracy is the stable headline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from benchmark_tasks import TASKS  # noqa: E402
from token_classification_probe import _compute_metrics  # noqa: E402


def test_ec_is_multilabel_with_micro_f1_headline():
    cfg = TASKS["ec_classification"]
    assert cfg.problem_type == "multilabel"
    assert cfg.main_metric == "F1_Micro"


def test_remote_homology_headline_is_top1_accuracy():
    assert TASKS["remote_homology"].main_metric == "Accuracy"


def test_conservation_headline_is_spearman():
    assert TASKS["conservation_flip"].main_metric == "Spearman"


def test_token_probe_emits_real_spearman_for_ordinal_task():
    """main_metric=='Spearman' -> a rank correlation is computed, not the 0.0
    placeholder the setdefault used to leave."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    m = _compute_metrics(y_true, y_pred, main_metric="Spearman")
    assert "Spearman" in m
    assert m["Spearman"] > 0.99  # perfectly rank-correlated


def test_token_probe_no_spearman_for_nominal_task():
    """Nominal tasks (SS3, disorder) must NOT get a meaningless Spearman column."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    m = _compute_metrics(y_true, y_pred, main_metric="F1_Macro")
    assert "Spearman" not in m
