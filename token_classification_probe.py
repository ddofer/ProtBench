"""Residue-level (token-classification) linear-probe + embedding cache.

Wires SS3 / Disorder benchmarks into the existing linear-probe pipeline:

1. ``extract_residue_embeddings`` runs the encoder once per protein,
   collects per-residue hidden states from ``output.last_hidden_state``
   (or ``hidden_states[-1]`` as fallback), and stacks them into
   ``(N_residues, hidden)`` plus aligned ``(N_residues,)`` labels —
   padding and special-token positions are excluded via the tokenizer's
   ``special_tokens_mask`` AND ``attention_mask``.
2. ``fit_residue_linear_probe`` fits an
   ``sklearn.linear_model.LogisticRegression`` (multiclass / binary) —
   linear probe per spec, no MLP.
3. ``EmbeddingCache`` persists per-residue embeddings keyed by
   ``(model_hash, task, split)`` to ``.npz`` so repeated bench runs
   skip the forward pass.
4. ``evaluate_token_classification`` is the public dispatch that
   ``protein_benchmark_suite.evaluate_task`` calls in place of the
   historic ``task_exception='token_classification'`` error path.

The cache key includes the model checkpoint hash so different models
never share cached residue embeddings.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def cache_key(model_hash: str, task: str, split: str) -> str:
    """Stable cache key keyed by (model, task, split). Different model
    hashes always produce different keys so different encoders never
    collide on cache files."""
    safe_model = hashlib.sha1(model_hash.encode("utf-8")).hexdigest()[:16]
    safe_task = "".join(c if c.isalnum() or c in "_-" else "_" for c in task)
    safe_split = "".join(c if c.isalnum() or c in "_-" else "_" for c in split)
    return f"{safe_model}__{safe_task}__{safe_split}"


@dataclass
class EmbeddingCache:
    """Filesystem-backed cache of per-residue embeddings.

    Layout::

        <root>/<key>.npz   (arrays: ``X`` float32, ``y`` int64)
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        path = self._path(key)
        if not path.exists():
            raise KeyError(f"cache miss: {path}")
        data = np.load(path)
        return data["X"], data["y"]

    def put(self, key: str, X: np.ndarray, y: np.ndarray) -> None:
        path = self._path(key)
        # Atomic-ish: write to a sibling ``<key>.tmp.npz`` then rename so
        # concurrent readers never see a half-written file. ``np.savez``
        # auto-appends ``.npz`` only if missing, so passing a path that
        # already ends in ``.npz`` is a no-op for the extension.
        tmp = path.with_name(path.stem + ".tmp.npz")
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                X=X.astype("float32", copy=False),
                y=y.astype("int64", copy=False),
            )
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Residue embedding extraction
# ---------------------------------------------------------------------------


def _last_hidden_state(outputs) -> "Any":
    """Return per-token hidden states from a model output, regardless of
    the various HF / SBert shapes."""
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if hasattr(outputs, "__getitem__"):
        return outputs[0]
    raise RuntimeError(f"could not extract hidden states from {type(outputs)}")


def extract_residue_embeddings(
    *,
    encoder,
    tokenizer,
    sequences: Sequence[str],
    labels: Sequence[Sequence[int]],
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run ``encoder`` once per protein, return stacked per-residue
    embeddings + per-residue labels.

    Padding and special-token positions are excluded via the union of
    ``attention_mask`` (= 0 on PAD) AND ``special_tokens_mask`` (= 1
    on CLS / SEP / PAD). Only positions where the mask is 1 AND the
    special_tokens_mask is 0 are kept.

    Per-protein truncation matches the tokenizer's ``max_length``; the
    label list is sliced to the surviving non-special token count.
    """
    import torch

    if len(sequences) != len(labels):
        raise ValueError(
            f"sequences ({len(sequences)}) and labels ({len(labels)}) length mismatch"
        )
    if not sequences:
        # 0-dim embedding column count is unknown at this point.
        return np.zeros((0, 0), dtype="float32"), np.zeros((0,), dtype="int64")

    encoder.eval() if hasattr(encoder, "eval") else None

    # AMPLIFY needs an ADDITIVE attention mask (0.0 / -inf) + length padded to a
    # multiple of 8, exactly like the sequence-level embed path; a bool mask
    # raises "AMPLIFY expects an additive attention_mask". Detected by the same
    # canonical config.model_type marker embed_sequences uses. Loop-invariant —
    # resolved once here, not per batch.
    is_amplify = str(
        getattr(getattr(encoder, "config", None), "model_type", "")
    ).lower() == "amplify"
    if is_amplify:
        from model_utils import _prepare_amplify_inputs

        _amplify_param = next(encoder.parameters(), None)
        _amplify_dtype = _amplify_param.dtype if _amplify_param is not None else None

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for start in range(0, len(sequences), batch_size):
        chunk_seqs = list(sequences[start : start + batch_size])
        chunk_labels = list(labels[start : start + batch_size])
        toks = tokenizer(
            chunk_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        if hasattr(toks, "to"):
            toks = toks.to(device)
        else:
            for k, v in list(toks.items()):
                if hasattr(v, "to"):
                    toks[k] = v.to(device)

        with torch.inference_mode():
            if is_amplify:
                ids_p, add_mask, orig_len, _ = _prepare_amplify_inputs(
                    toks["input_ids"],
                    toks.get("attention_mask"),
                    device=device,
                    dtype=_amplify_dtype,
                )
                outputs = encoder(
                    input_ids=ids_p,
                    attention_mask=add_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = _last_hidden_state(outputs)[:, :orig_len, :]  # un-pad
                # AMPLIFY's final norm is NOT applied inside hidden_states.
                if hasattr(encoder, "layer_norm_2"):
                    hidden = encoder.layer_norm_2(hidden)
            else:
                outputs = encoder(
                    input_ids=toks["input_ids"],
                    attention_mask=toks.get("attention_mask"),
                )
                hidden = _last_hidden_state(outputs)  # (B, L, H)

        attn = toks.get("attention_mask")
        stm = toks["special_tokens_mask"]
        if attn is None:
            attn = torch.ones_like(stm)

        # keep positions where attention == 1 AND special_tokens_mask == 0
        keep = (attn.to(torch.bool)) & (~stm.to(torch.bool))

        hidden_np = hidden.detach().to("cpu").float().numpy()
        keep_np = keep.detach().to("cpu").numpy()

        for i, lab_list in enumerate(chunk_labels):
            n_keep = int(keep_np[i].sum())
            if n_keep == 0:
                continue
            row_hidden = hidden_np[i][keep_np[i]]  # (n_keep, H)
            # Truncate per-residue labels to match the surviving residue
            # count (matters when sequences exceeded ``max_length``).
            row_labels = np.asarray(lab_list[:n_keep], dtype="int64")
            n = min(row_hidden.shape[0], row_labels.shape[0])
            all_X.append(row_hidden[:n])
            all_y.append(row_labels[:n])

    if not all_X:
        return np.zeros((0, 0), dtype="float32"), np.zeros((0,), dtype="int64")
    X = np.concatenate(all_X, axis=0).astype("float32")
    y = np.concatenate(all_y, axis=0).astype("int64")
    return X, y


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------


def fit_residue_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    problem_type: str = "multiclass",
    max_iter: int = 1000,
    seed: int = 42,
):
    """Fit a per-residue LogisticRegression linear probe.

    Per spec: linear probe only, no MLP. ``n_jobs=-1`` lets sklearn
    parallelize the underlying solvers / OvR loops.
    """
    from sklearn.linear_model import LogisticRegression

    # n_jobs deprecated in sklearn >=1.8; LogisticRegression now uses internal
    # OpenMP/loky parallelism without the kwarg.
    # saga handles large n_samples (SS3: ~600k residues) where lbfgs stalls at 1000 iter
    probe = LogisticRegression(
        solver="saga",
        max_iter=max_iter,
        random_state=seed,
    )
    probe.fit(X, y)
    return probe


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    main_metric: str,
) -> Dict[str, float]:
    """Compute Accuracy + main_metric for the residue probe."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
    )

    if y_true.size == 0:
        return {"Accuracy": 0.0, "F1_Macro": 0.0, "MCC": 0.0}

    metrics: Dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_Macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        metrics["MCC"] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        metrics["MCC"] = 0.0

    # Ensure main_metric is always present (e.g. custom metric not in the
    # standard Accuracy/F1_Macro/MCC set computed above).
    metrics.setdefault(main_metric, 0.0)
    return metrics


def evaluate_token_classification(
    *,
    cfg,
    encoder,
    tokenizer,
    train_sequences: Sequence[str],
    train_labels: Sequence[Sequence[int]],
    test_sequences: Optional[Sequence[str]] = None,
    test_labels: Optional[Sequence[Sequence[int]]] = None,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 1024,
    cache: Optional[EmbeddingCache] = None,
    model_hash: str = "unknown-model",
    task_key: Optional[str] = None,
) -> Dict[str, float]:
    """Run the residue linear-probe pipeline for a token-classification task.

    If ``cache`` is provided, residue embeddings for each split are
    cached to disk under ``cache_key(model_hash, task, split)`` and
    reused on hit.

    If ``test_sequences`` is None, a 4-fold CV is run over the train
    residues. Default ``n_splits=4`` matches ``protein_benchmark_suite``.
    """
    from sklearn.model_selection import KFold

    task = task_key or getattr(cfg, "name", "task")

    def _extract_or_load(seqs, labs, split: str) -> Tuple[np.ndarray, np.ndarray]:
        if cache is not None:
            key = cache_key(model_hash, task, split)
            if cache.has(key):
                logger.info(
                    "residue cache HIT: model=%s task=%s split=%s",
                    model_hash,
                    task,
                    split,
                )
                return cache.get(key)
        X, y = extract_residue_embeddings(
            encoder=encoder,
            tokenizer=tokenizer,
            sequences=seqs,
            labels=labs,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        if cache is not None:
            cache.put(cache_key(model_hash, task, split), X, y)
        return X, y

    X_tr, y_tr = _extract_or_load(train_sequences, train_labels, "train")

    if test_sequences is not None and test_labels is not None:
        X_te, y_te = _extract_or_load(test_sequences, test_labels, "test")
        probe = fit_residue_linear_probe(
            X_tr, y_tr, problem_type=cfg.problem_type or "multiclass"
        )
        preds = probe.predict(X_te)
        return _compute_metrics(y_te, preds, main_metric=cfg.main_metric)

    # 4-fold CV fallback over train residues. Splitting at residue level
    # (rather than protein level) matches the existing CV fallback in
    # ``protein_benchmark_suite`` which also splits over flattened rows.
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    fold_metrics: List[Dict[str, float]] = []
    for tr_idx, te_idx in kf.split(X_tr):
        probe = fit_residue_linear_probe(
            X_tr[tr_idx], y_tr[tr_idx], problem_type=cfg.problem_type or "multiclass"
        )
        preds = probe.predict(X_tr[te_idx])
        fold_metrics.append(_compute_metrics(y_tr[te_idx], preds, main_metric=cfg.main_metric))
    keys = set().union(*(m.keys() for m in fold_metrics))
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
