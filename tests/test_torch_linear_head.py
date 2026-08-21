"""TorchLinearHead: sklearn-API torch linear head (frozen features -> nn.Linear)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from torch_linear_head import TorchLinearHead  # noqa: E402


def _blobs(n=600, d=16, k=3, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, d) * 4
    y = rng.randint(0, k, size=n)
    X = centers[y] + rng.randn(n, d)
    return X.astype(np.float32), y


def test_classification_fit_predict_proba():
    X, y = _blobs()
    head = TorchLinearHead(task="classification", seed=0, device="cpu").fit(X[:500], y[:500])
    preds = head.predict(X[500:])
    assert preds.shape == (100,)
    assert np.mean(preds == y[500:]) >= 0.95
    proba = head.predict_proba(X[500:])
    assert proba.shape == (100, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert list(head.classes_) == [0, 1, 2]


def test_regression_on_large_scale_targets():
    rng = np.random.RandomState(1)
    X = rng.randn(800, 10).astype(np.float32)
    w = rng.randn(10)
    y = 60.0 + 15.0 * (X @ w) + rng.randn(800) * 0.5  # Topt-like scale
    from sklearn.metrics import r2_score

    from protein_benchmark_suite import make_probe_model

    model = make_probe_model("torch_linear", "regression")  # scaler on X, y standardised
    model.fit(X[:600], y[:600])
    pred = model.predict(X[600:])
    assert pred.shape == (200,)
    assert r2_score(y[600:], pred) > 0.9


def test_multilabel_recovers_planted_labels():
    rng = np.random.RandomState(2)
    X = rng.randn(800, 12).astype(np.float32)
    W = rng.randn(12, 4)
    Y = (X @ W > 0).astype(np.int64)
    head = TorchLinearHead(task="multilabel", seed=0, device="cpu").fit(X[:600], Y[:600])
    pred = head.predict(X[600:])
    assert pred.shape == (200, 4)
    assert set(np.unique(pred)) <= {0, 1}
    assert (pred == Y[600:]).mean() > 0.9
    proba = head.predict_proba(X[600:])
    assert proba.shape == (200, 4) and (proba >= 0).all() and (proba <= 1).all()


def test_early_stopping_fires_and_tiny_sets_train():
    X, y = _blobs(n=2000, seed=3)
    y = np.where(np.random.RandomState(4).rand(2000) < 0.4, np.random.RandomState(5).randint(0, 3, 2000), y)
    head = TorchLinearHead(seed=0, device="cpu", max_epochs=50, patience=1).fit(X, y)
    assert head.n_epochs_ < 50
    tiny = TorchLinearHead(seed=0, device="cpu").fit(X[:20], y[:20])  # no room for a val split
    assert tiny.predict(X[:5]).shape == (5,)


def test_deterministic_given_seed():
    X, y = _blobs(seed=6)
    a = TorchLinearHead(seed=11, device="cpu").fit(X, y).predict_proba(X[:50])
    b = TorchLinearHead(seed=11, device="cpu").fit(X, y).predict_proba(X[:50])
    np.testing.assert_array_equal(a, b)


def test_registered_as_probe_type():
    from protein_benchmark_suite import PROBE_LABELS, make_probe_model

    assert "torch_linear" in PROBE_LABELS
    for problem in ("binary", "multiclass", "regression"):
        model = make_probe_model("torch_linear", problem)
        assert hasattr(model, "fit") and hasattr(model, "predict")
