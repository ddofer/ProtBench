"""Regression tasks persist per-example test predictions, like classification.

A paired bootstrap of a regression delta (e.g. Stability Spearman between two
checkpoints) resamples test proteins, so it needs y_true/y_pred per example
under an id that is stable across models, not only the aggregate metric.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

import protein_benchmark_suite as pbs
from benchmark_tasks import TASKS
from prediction_artifacts import _sequence_sha256, read_prediction_rows


def _regression_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    return X, X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(0, 0.1, size=n)


def test_regression_probe_exposes_its_test_predictions():
    X_train, y_train = _regression_data(80, seed=0)
    X_test, y_test = _regression_data(20, seed=1)
    captured = []
    metrics = pbs.evaluate_regression_probe(
        "linear",
        X_train,
        y_train,
        X_test,
        y_test,
        prediction_callback=lambda *args: captured.append(args),
    )
    labels, predictions, scores = captured[0]
    np.testing.assert_array_equal(labels, y_test)
    assert scores is None
    assert spearmanr(labels, predictions)[0] == pytest.approx(metrics["Spearman"])


def test_regression_task_writes_a_prediction_artifact(tmp_path, monkeypatch):
    X_train, y_train = _regression_data(80, seed=0)
    X_test, y_test = _regression_data(20, seed=1)
    train_seqs = [f"M{'A' * i}K" for i in range(80)]
    test_seqs = [f"M{'C' * i}K" for i in range(20)]
    embeddings = dict(zip(train_seqs + test_seqs, [*X_train, *X_test]))
    monkeypatch.setattr(
        pbs,
        "prepare_data",
        lambda *args, **kwargs: (
            train_seqs,
            list(y_train),
            test_seqs,
            list(y_test),
            None,
            {"resolved_eval_split": "test", "eval_strategy": "test_split"},
        ),
    )
    monkeypatch.setattr(
        pbs,
        "embed_sequences",
        lambda model_obj, is_sbert, seqs, *args, **kwargs: np.stack(
            [embeddings[s] for s in seqs]
        ),
    )
    cfg = TASKS["stability"]

    metrics, split, _ = pbs.evaluate_task(
        cfg,
        model_obj=None,
        is_sbert=False,
        device="cpu",
        probe_type="linear",
        prediction_dir=str(tmp_path),
    )

    metadata, rows = read_prediction_rows(tmp_path / "Stability_Biomap_.jsonl.gz")
    assert metadata["problem_type"] == "regression"
    assert metadata["split"] == split == "test"
    assert [row["example_id"] for row in rows] == list(range(len(test_seqs)))
    assert [row["sequence_sha256"] for row in rows] == [
        _sequence_sha256(s) for s in test_seqs
    ]
    np.testing.assert_allclose([row["label"] for row in rows], y_test)
    assert all(row["score"] is None for row in rows)
    predictions = [row["prediction"] for row in rows]
    assert spearmanr(y_test, predictions)[0] == pytest.approx(metrics["Spearman"])
