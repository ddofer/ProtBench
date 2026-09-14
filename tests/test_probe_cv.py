"""Behaviour of the validation-selected linear probe (``-p linear_cv``).

Protocol (Radford et al. 2021, CLIP supp. A): L2 strength chosen on a validation
split from a log grid, fit to convergence, then refit on train and scored on test.
"""

import numpy as np

from probe_cv import fit_selected


def test_when_validation_is_the_training_set_selection_picks_the_weakest_regularisation():
    # Scoring on the data it was fit to, the least-regularised fit can never lose,
    # so the selected strength must be the grid's weakest end (largest C).
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 20))
    y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    grid = [1e-4, 1e-2, 1.0, 1e2, 1e4]
    _, chosen = fit_selected("binary", X, y, X, y, grid=grid)
    assert chosen == 1e4


def test_validation_rows_with_a_class_absent_from_training_are_ignored_for_selection():
    # Fold-level splits (remote homology) put classes in validation that train never
    # saw; no probe can score them, so they must not crash or steer the selection.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 10))
    y = np.digitize(X[:, 0], [-0.5, 0.5])          # classes 0, 1, 2
    X_val = np.vstack([X, rng.normal(size=(5, 10))])
    y_val = np.concatenate([y, np.full(5, 7)])    # class 7 never in train
    _, chosen_with_unseen = fit_selected("multiclass", X, y, X_val, y_val, grid=[1e-2, 1.0, 1e2])
    _, chosen_without = fit_selected("multiclass", X, y, X, y, grid=[1e-2, 1.0, 1e2])
    assert chosen_with_unseen == chosen_without


def test_multilabel_selection_on_the_training_set_picks_the_weakest_regularisation():
    # GO/EC: one head per label; same in-sample argument as the binary case.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 12))
    Y = np.stack([(X[:, 0] + 0.5 * rng.normal(size=200) > 0),
                  (X[:, 1] + 0.5 * rng.normal(size=200) > 0)], axis=1).astype(int)
    _, chosen = fit_selected("multilabel", X, Y, X, Y, grid=[1e-3, 1.0, 1e3])
    assert chosen == 1e3
