"""Optional percentile-bootstrap CIs on probe metrics.

A single benchmark number carries no sense of how much of the gap between two
models is sampling noise. Resampling the test predictions and recomputing each
metric gives that, without refitting the probe, so it costs a few seconds.

Off by default: it adds columns, and every existing consumer of these CSVs
expects the old schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protein_benchmark_suite as pbs  # noqa: E402


@pytest.fixture
def bootstrap_on(monkeypatch):
    monkeypatch.setattr(pbs, "BOOTSTRAP_N", 200)


def _noisy(n=200, seed=0):
    """Learnable but not perfect, so the CI has width to report."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = y.reshape(-1, 1) + rng.normal(0, 1.0, size=(n, 1))
    return X, y


def _regression(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.5, size=n)
    return X, y


def test_no_ci_columns_when_disabled():
    X, y = _noisy()
    metrics = pbs.evaluate_classification_probe("linear", "binary", X, y, X, y)
    assert not [k for k in metrics if "_CI_" in k]


def test_classification_ci_brackets_the_point_estimate(bootstrap_on):
    X, y = _noisy()
    metrics = pbs.evaluate_classification_probe("linear", "binary", X, y, X, y)
    assert metrics["Accuracy_CI_low"] <= metrics["Accuracy"]
    assert metrics["Accuracy"] <= metrics["Accuracy_CI_high"]
    assert metrics["Accuracy_CI_low"] < metrics["Accuracy_CI_high"]


def test_regression_ci_brackets_the_point_estimate(bootstrap_on):
    X, y = _regression()
    metrics = pbs.evaluate_regression_probe("linear", X, y, X, y)
    assert metrics["Spearman_CI_low"] <= metrics["Spearman"]
    assert metrics["Spearman"] <= metrics["Spearman_CI_high"]


def test_ci_is_reproducible_across_calls(bootstrap_on):
    """Bootstrap draws must be seeded, or the interval moves every run."""
    X, y = _noisy()
    a = pbs.evaluate_classification_probe("linear", "binary", X, y, X, y)
    b = pbs.evaluate_classification_probe("linear", "binary", X, y, X, y)
    assert a["Accuracy_CI_low"] == b["Accuracy_CI_low"]
    assert a["Accuracy_CI_high"] == b["Accuracy_CI_high"]


def test_ci_covers_only_the_label_only_metrics(bootstrap_on):
    """AUC/AP are computed from predict_proba after the fact; documented gap."""
    X, y = _noisy()
    metrics = pbs.evaluate_classification_probe("linear", "binary", X, y, X, y)
    assert "MCC_CI_low" in metrics
    assert "AUC" in metrics
    assert "AUC_CI_low" not in metrics


def test_bootstrap_flag_defaults_to_zero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["protein_benchmark_suite.py", "-m", "dummy"])
    assert pbs.parse_args().bootstrap == 0
