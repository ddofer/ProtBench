"""Exact sklearn parity for bootstrap average precision with cached score order."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from ptm_compare import _prepare_average_precision


@pytest.mark.parametrize("ties", [False, True])
def test_cached_ap_matches_sklearn_for_cluster_weights(ties: bool) -> None:
    """Check tied/continuous scores with zero weights and resampled protein counts.

    Args:
        ties: Whether to quantize the fixed predictions to introduce score ties.
    """
    rng = np.random.default_rng(19)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    if ties:
        scores = np.round(scores, 1)
    prepared = _prepare_average_precision(labels, scores)
    for _ in range(30):
        weights = np.repeat(rng.multinomial(20, np.full(20, 1 / 20)), 10)
        assert prepared(weights) == pytest.approx(
            average_precision_score(labels, scores, sample_weight=weights), abs=1e-14
        )


def test_cached_ap_handles_constant_scores_and_empty_positive_weight() -> None:
    """Constant scores equal weighted prevalence; missing positives produce NaN."""
    labels = np.array([1, 0, 1, 0])
    prepared = _prepare_average_precision(labels, np.ones(4))
    assert prepared(np.array([2, 1, 0, 3])) == pytest.approx(1 / 3)
    assert np.isnan(prepared(np.array([0, 1, 0, 3])))
    assert np.isnan(prepared(np.zeros(4)))


def test_cached_ap_ignores_zero_weight_leading_score_ties() -> None:
    """Absent high-scoring groups must not introduce divisions by zero."""
    labels = np.array([1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.9, 0.6, 0.4, 0.1])
    weights = np.array([0, 0, 2, 1, 1])
    assert _prepare_average_precision(labels, scores)(weights) == pytest.approx(
        average_precision_score(labels, scores, sample_weight=weights), abs=1e-14
    )
