from __future__ import annotations

import numpy as np
import pytest

from ptm_ppi import binder_aware_features, paper_reproduction_features


def test_paper_reproduction_cancels_the_binder_exactly() -> None:
    wt = np.asarray([[1.0, 2.0]])
    ptm = np.asarray([[2.0, 4.0]])
    left = paper_reproduction_features(np.asarray([[3.0, 5.0]]), wt, ptm)
    right = paper_reproduction_features(np.asarray([[30.0, 50.0]]), wt, ptm)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(left[:, :2], 0.0)


def test_primary_features_change_when_the_binder_changes() -> None:
    wt = np.asarray([[1.0, 2.0]])
    ptm = np.asarray([[2.0, 4.0]])
    left = binder_aware_features(np.asarray([[3.0, 5.0]]), wt, ptm)
    right = binder_aware_features(np.asarray([[30.0, 50.0]]), wt, ptm)
    assert left.shape == (1, 14)
    assert not np.array_equal(left, right)


def test_ppi_features_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        binder_aware_features(
            np.zeros((2, 3)), np.zeros((2, 4)), np.zeros((2, 3))
        )
