"""MCC and balanced accuracy belong on every classification result.

Accuracy and F1 both flatter a majority-class predictor on the imbalanced
tasks in this suite. MCC does not -- it goes to 0 for a constant prediction
regardless of the class ratio -- and balanced accuracy reports the mean
per-class recall, which is what "did it learn the rare class" actually asks.
Both are label-only, so unlike AUC/AP they are always computable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protein_benchmark_suite as pbs  # noqa: E402


def _separable(n=60, dim=4, classes=2, seed=0):
    """Trivially separable data, so the probe is not what is under test."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % classes
    X = np.eye(classes)[y] * 5.0 + rng.normal(0, 0.01, size=(n, classes))
    X = np.hstack([X, rng.normal(0, 0.01, size=(n, dim))])
    return X, y


@pytest.mark.parametrize("problem_type,classes", [("binary", 2), ("multiclass", 3)])
def test_probe_reports_mcc_and_balanced_accuracy(problem_type, classes):
    X, y = _separable(classes=classes)
    metrics = pbs.evaluate_classification_probe("linear", problem_type, X, y, X, y)
    assert "MCC" in metrics
    assert "BalancedAccuracy" in metrics
    assert metrics["MCC"] == pytest.approx(1.0)
    assert metrics["BalancedAccuracy"] == pytest.approx(1.0)




def test_metric_priority_is_untouched():
    """Adding metrics must not silently re-rank historical comparisons."""
    from benchmark_utils import METRIC_PRIORITY

    assert "MCC" not in METRIC_PRIORITY
    assert "BalancedAccuracy" not in METRIC_PRIORITY
