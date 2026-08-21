"""Frozen-backbone linear head trained with torch (via skorch), sklearn API.

Same embed-once pipeline as the ``linear`` probe, but the head is an
``nn.Linear`` fit by AdamW with early stopping on a small inner val split
(patience 3, LAST weights kept -- frozen features, no augmentation, so the best
epoch is within a step or two of the last). ``max_epochs`` is a cap that
early stopping normally makes irrelevant (typical stop: 5-15 epochs); it binds
only when the val loss is still falling, e.g. sparse multilabel (EC: 572 labels,
0.3 % positives) where 20 epochs cost 0.2 F1. Measured on ESM2-8M (see README):
patience 1 stopped regression heads 2-4 epochs in and lost ~0.09 Spearman vs
patience 3. ``make_probe_model`` wraps it in the same
``StandardScaler`` pipeline as LogisticRegression/Ridge, so every downstream
evaluator, CV path and metric block works unchanged.

Why a wrapper around skorch instead of skorch directly: ``make_probe_model``
never sees ``y``, so ``out_features`` / loss / input dim are only known at
``fit``. The wrapper builds the ``NeuralNet*`` then.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator
from skorch import NeuralNet, NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit

TASKS = ("classification", "regression", "multilabel")


class TorchLinearHead(BaseEstimator):
    """sklearn-API torch linear head. ``task``: classification | regression | multilabel."""

    def __init__(
        self,
        task: str = "classification",
        max_epochs: int = 100,
        min_steps: int = 300,
        patience: int = 3,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        val_frac: float = 0.05,
        min_val: int = 32,
        seed: int = 42,
        device: str | None = None,
    ):
        self.task = task
        self.max_epochs = max_epochs
        self.min_steps = min_steps
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.min_val = min_val
        self.seed = seed
        self.device = device

    def fit(self, X, y):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        if self.task == "classification":
            self.classes_, y = np.unique(y, return_inverse=True)
            y = y.astype(np.int64)
            net_cls, k, extra = NeuralNetClassifier, len(self.classes_), {"criterion": torch.nn.CrossEntropyLoss}
        elif self.task == "multilabel":
            y = np.asarray(y, dtype=np.float32)
            net_cls, k, extra = NeuralNet, y.shape[1], {"criterion": torch.nn.BCEWithLogitsLoss, "predict_nonlinearity": torch.sigmoid}
        else:
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            net_cls, k, extra = NeuralNetRegressor, 1, {}

        # Small inner val split for early stopping; skipped when it would eat
        # more than half the data (then just run max_epochs).
        n_val = max(int(self.val_frac * n), self.min_val)
        callbacks, split = [], None
        if 2 * n_val <= n:
            # Not stratified: 1000-class tasks (remote_homology) have singleton
            # classes that StratifiedShuffleSplit refuses; a random split is fine
            # for a loss-based stopping signal.
            split = ValidSplit(n_val / n, stratified=False, random_state=self.seed)
            callbacks = [EarlyStopping(patience=self.patience, load_best=False)]

        # Epoch cap is really a step budget: on a few-hundred-sample task one
        # epoch is 1-2 AdamW steps, so ``max_epochs`` alone under-trains vs
        # lbfgs. Guarantee ``min_steps`` optimizer steps; early stopping still
        # ends training as soon as the val loss stops improving.
        steps_per_epoch = max(1, (n - n_val if split else n) // self.batch_size)
        max_epochs = max(self.max_epochs, -(-self.min_steps // steps_per_epoch))

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Hand skorch device tensors, not numpy: its DataLoader indexes rows one
        # at a time, and per-row numpy->tensor->device copies roughly double the
        # step time (measured 5.8 vs 2.5 ms/step at H=320, batch 256).
        X, y = torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)

        torch.manual_seed(self.seed)
        self.net_ = net_cls(
            torch.nn.Linear,
            module__in_features=d,
            module__out_features=k,
            optimizer=torch.optim.AdamW,
            lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
            max_epochs=max_epochs,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            train_split=split,
            callbacks=callbacks,
            device=device,
            verbose=0,
            **extra,
        )
        self.net_.fit(X, y)
        self.n_epochs_ = len(self.net_.history)
        return self

    def predict_proba(self, X):
        # skorch applies softmax (CE) / sigmoid (BCEWithLogits) automatically.
        return self.net_.predict_proba(np.ascontiguousarray(X, dtype=np.float32))

    def predict(self, X):
        out = self.predict_proba(X)
        if self.task == "classification":
            return self.classes_[out.argmax(axis=1)]
        if self.task == "multilabel":
            return (out > 0.5).astype(np.int64)
        return out[:, 0]
