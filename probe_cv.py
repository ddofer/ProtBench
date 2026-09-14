"""Linear probe with the L2 strength selected on a validation split (``-p linear_cv``).

Follows the CLIP linear-probe protocol (Radford et al. 2021, supp. A): sweep the
inverse regularisation strength C on held-out validation data, then refit the
selected probe on train. Selection uses a proper, continuous validation loss
(log-loss / MSE) rather than accuracy, so near-ties do not decide the pick.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_GRID = tuple(10.0**e for e in range(-6, 7, 2))
MAX_ITER = 1000
SEED = 42


def make_model(problem_type: str, C: float):
    """StandardScaler + L2 linear head; C is the inverse strength for every problem type."""
    if problem_type == "regression":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0 / C))
    head = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER, random_state=SEED)
    if problem_type == "multilabel":
        return make_pipeline(StandardScaler(), OneVsRestClassifier(head, n_jobs=-1))
    return make_pipeline(StandardScaler(), head)


def validation_loss(problem_type: str, model, X_val, y_val) -> float:
    if problem_type == "regression":
        return float(np.mean((np.asarray(y_val, dtype=float) - model.predict(X_val)) ** 2))
    if problem_type == "multilabel":
        # Mean per-label binary log-loss over the indicator matrix.
        p = np.clip(model.predict_proba(X_val), 1e-15, 1 - 1e-15)
        y = np.asarray(y_val, dtype=float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    # Rows whose class never occurs in train cannot be scored by any probe; drop them.
    known = np.isin(y_val, model.classes_)
    return float(
        log_loss(np.asarray(y_val)[known], model.predict_proba(np.asarray(X_val)[known]), labels=model.classes_)
    )


def fit_selected(problem_type: str, X_train, y_train, X_val, y_val, grid=DEFAULT_GRID):
    """Return (probe fit on train at the best C, best C); ties keep the first C."""
    best_loss, best = float("inf"), None
    for C in grid:
        model = make_model(problem_type, C).fit(X_train, y_train)
        loss = validation_loss(problem_type, model, X_val, y_val)
        if loss < best_loss:
            best_loss, best = loss, (model, C)
    return best
