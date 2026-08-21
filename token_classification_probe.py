"""Residue-level (token-classification) linear-probe + embedding cache.

Wires SS3 / Disorder benchmarks into the existing linear-probe pipeline:

1. ``extract_residue_embeddings`` runs the encoder once per protein,
   collects per-residue hidden states from ``output.last_hidden_state``
   (or ``hidden_states[-1]`` as fallback), and stacks them into
   ``(N_residues, hidden)`` plus aligned ``(N_residues,)`` labels —
   padding and special-token positions are excluded via the tokenizer's
   ``special_tokens_mask`` AND ``attention_mask``.
2. The probe is whatever ``-p`` selected, built by the same
   ``protein_benchmark_suite.make_probe_model`` registry as the sequence-level
   tasks (``linear`` = StandardScaler + LogisticRegression(lbfgs),
   ``torch_linear`` = StandardScaler + GPU nn.Linear head, ...).
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


CACHE_VERSION = "v2"  # v2 stores protein groups; older entries are plain misses


def cache_key(model_hash: str, task: str, split: str) -> str:
    """Stable cache key keyed by (model, task, split). Different model
    hashes always produce different keys so different encoders never
    collide on cache files. The version tag turns a format change into an
    ordinary miss -- the cache is a derived artifact, so re-extracting is
    cheaper than carrying a compatibility branch forever."""
    safe_model = hashlib.sha1(model_hash.encode("utf-8")).hexdigest()[:16]
    safe_task = "".join(c if c.isalnum() or c in "_-" else "_" for c in task)
    safe_split = "".join(c if c.isalnum() or c in "_-" else "_" for c in split)
    return f"{safe_model}__{safe_task}__{safe_split}__{CACHE_VERSION}"


@dataclass
class EmbeddingCache:
    """Filesystem-backed cache of per-residue embeddings.

    Layout::

        <root>/<key>.npz   (arrays: ``X`` float32, ``y`` int64,
                            ``g`` int64 = protein index per residue)
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Always ``(X, y, groups)``. ``seq_embed_cache`` reuses this class for
        whole-sequence embeddings and has neither labels nor proteins, so those
        come back as zeros rather than changing the arity per caller."""
        path = self._path(key)
        if not path.exists():
            raise KeyError(f"cache miss: {path}")
        data = np.load(path)
        g = data["g"] if "g" in data.files else np.zeros(len(data["y"]), dtype="int64")
        return data["X"], data["y"], g

    def put(
        self, key: str, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> None:
        path = self._path(key)
        # Atomic-ish: write to a sibling ``<key>.tmp.npz`` then rename so
        # concurrent readers never see a half-written file. ``np.savez``
        # auto-appends ``.npz`` only if missing, so passing a path that
        # already ends in ``.npz`` is a no-op for the extension.
        tmp = path.with_name(path.stem + ".tmp.npz")
        with open(tmp, "wb") as fh:
            arrays = {
                "X": X.astype("float32", copy=False),
                "y": y.astype("int64", copy=False),
            }
            if groups is not None:
                arrays["g"] = np.asarray(groups).astype("int64", copy=False)
            np.savez(fh, **arrays)
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


def iter_residue_embeddings(
    *,
    encoder,
    tokenizer,
    sequences: Sequence[str],
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 1024,
):
    """Yield one ``(L_i, H)`` per-residue embedding array per input sequence,
    in input order.

    Protein boundaries are preserved, which pairwise tasks (contact prediction)
    need and ``extract_residue_embeddings`` throws away when it concatenates.

    Padding and special-token positions are excluded via the union of
    ``attention_mask`` (= 0 on PAD) AND ``special_tokens_mask`` (= 1
    on CLS / SEP / PAD). Only positions where the mask is 1 AND the
    special_tokens_mask is 0 are kept. ``L_i`` is therefore the truncated
    length when a sequence exceeds ``max_length``, not the raw length.

    Batches are formed longest-first so padding waste is minimal and an OOM
    shows up on the first batch; kept positions are gathered on the device
    and only the ``(sum L_i, H)`` block crosses to host.
    """
    import torch

    if not sequences:
        return

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

    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]), reverse=True)
    out: List[Optional[np.ndarray]] = [None] * len(sequences)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        chunk_seqs = [sequences[i] for i in idx]
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
            keep = attn.to(torch.bool) & ~stm.to(torch.bool)
            lengths = keep.sum(dim=1).tolist()
            flat = hidden[keep].float().cpu().numpy()  # (sum L_i, H), masked on device

        offset = 0
        for i, n_keep in zip(idx, lengths):
            out[i] = flat[offset : offset + n_keep]
            offset += n_keep

    yield from out


def extract_residue_embeddings(
    *,
    encoder,
    tokenizer,
    sequences: Sequence[str],
    labels: Sequence[Sequence[int]],
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``encoder`` once per protein, return stacked per-residue embeddings,
    per-residue labels, and the protein index per residue (needed for
    protein-level CV, and 8 bytes/residue against a float32 (T, d) block).

    Per-protein truncation matches the tokenizer's ``max_length``; the
    label list is sliced to the surviving non-special token count. A
    residue/label count mismatch that truncation cannot explain means the
    tokenizer did not emit one token per residue (or labels are misaligned):
    it is logged, and raises if it affects more than half the proteins.
    """
    if len(sequences) != len(labels):
        raise ValueError(
            f"sequences ({len(sequences)}) and labels ({len(labels)}) length mismatch"
        )
    if not sequences:
        # 0-dim embedding column count is unknown at this point.
        return (
            np.zeros((0, 0), dtype="float32"),
            np.zeros((0,), dtype="int64"),
            np.zeros((0,), dtype="int64"),
        )

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_g: List[np.ndarray] = []
    per_protein = iter_residue_embeddings(
        encoder=encoder,
        tokenizer=tokenizer,
        sequences=sequences,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    n_misaligned = 0
    for prot_idx, (row_hidden, lab_list, seq) in enumerate(zip(per_protein, labels, sequences)):
        if row_hidden.shape[0] == 0:
            continue
        n_keep, n_lab = row_hidden.shape[0], len(lab_list)
        # Fewer residues than labels is expected only when the tokenizer
        # truncated the sequence (max_length includes ~2 special tokens).
        if n_keep != n_lab and not (n_keep < n_lab and len(seq) > max_length - 2):
            n_misaligned += 1
        n = min(n_keep, n_lab)
        all_X.append(row_hidden[:n])
        all_y.append(np.asarray(lab_list[:n], dtype="int64"))
        all_g.append(np.full(n, prot_idx, dtype="int64"))

    if n_misaligned:
        msg = (
            f"{n_misaligned}/{len(sequences)} proteins: residue count != label count "
            "and not explained by truncation (tokenizer not one-token-per-residue, "
            "or misaligned labels)"
        )
        if n_misaligned > len(sequences) / 2:
            raise ValueError(msg)
        logger.warning("  %s; extra positions dropped", msg)

    if not all_X:
        return (
            np.zeros((0, 0), dtype="float32"),
            np.zeros((0,), dtype="int64"),
            np.zeros((0,), dtype="int64"),
        )
    # copy=False: the concat result is already float32, and a third full-size
    # allocation is ~3 GB at 600k residues x 1280 dims.
    X = np.concatenate(all_X, axis=0).astype("float32", copy=False)
    y = np.concatenate(all_y, axis=0).astype("int64", copy=False)
    return X, y, np.concatenate(all_g, axis=0)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------


def drop_ignored_residues(
    X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None
):
    """Drop residues carrying ``IGNORE_LABEL`` from the embeddings AND labels.

    Some datasets mark residues with no ground truth (SS8's unassigned termini).
    They must be removed from both arrays together -- removing them earlier, at
    decode time, would shorten the label list and shift every later label
    against its residue embedding.
    """
    keep = y >= 0 if y.size else np.ones(0, dtype=bool)
    if keep.all():
        return (X, y) if groups is None else (X, y, groups)
    if groups is None:
        return X[keep], y[keep]
    return X[keep], y[keep], groups[keep]


def fit_residue_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    problem_type: str = "multiclass",
    probe_type: str = "linear",
):
    """Fit the selected probe on per-residue embeddings.

    Same registry as the sequence-level tasks (``make_probe_model``): ``linear``
    is StandardScaler + LogisticRegression(lbfgs, 100 iters) -- scaling lets
    lbfgs converge in ~100 iters even on SS3's ~600k residues (saga on raw
    features was ~900s/task vs ~60-120s); ``torch_linear`` is the GPU head.
    """
    from protein_benchmark_suite import _make_probe_model_for_training_size

    probe = _make_probe_model_for_training_size(probe_type, problem_type, len(X))
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
    problem_type: str = "multiclass",
) -> Dict[str, float]:
    """Same metric block (+ bootstrap CIs) as the sequence-level probes, plus
    Spearman for ordinal per-residue tasks (conservation grades 1-9), where
    nominal F1 gives an off-by-one prediction the same 0 credit as off-by-eight.
    """
    import functools

    import protein_benchmark_suite as pbs

    if y_true.size == 0:
        return {"Accuracy": 0.0, "F1_Macro": 0.0, "MCC": 0.0}

    metrics = pbs.classification_metrics(problem_type, y_true, y_pred)
    metrics.update(
        pbs._boot_ci(
            functools.partial(pbs.classification_metrics, problem_type),
            y_true,
            y_pred,
            pbs.BOOTSTRAP_N,
            pbs.BENCHMARK_SEED,
        )
    )
    if main_metric == "Spearman":
        from scipy.stats import spearmanr

        rho, _ = spearmanr(y_true, y_pred)
        metrics["Spearman"] = float(rho) if rho == rho else 0.0

    # Ensure main_metric is always present (e.g. custom metric not in the
    # standard set computed above).
    metrics.setdefault(main_metric, 0.0)
    return {k: float(v) for k, v in metrics.items()}


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
    probe_type: str = "linear",
) -> Dict[str, float]:
    """Run the residue probe pipeline for a token-classification task.

    If ``cache`` is provided, residue embeddings for each split are
    cached to disk under ``cache_key(model_hash, task, split)`` and
    reused on hit.

    If ``test_sequences`` is None, a 4-fold protein-level CV (GroupKFold) is
    run over the train residues -- residues of one protein never land on both
    sides of a fold. Default ``n_splits=4`` matches ``protein_benchmark_suite``.
    """
    import time

    from sklearn.model_selection import GroupKFold

    task = task_key or getattr(cfg, "name", "task")

    def _extract_or_load(seqs, labs, split: str):
        key = cache_key(model_hash, task, split) if cache is not None else None
        if key is not None and cache.has(key):
            logger.info("residue cache HIT: model=%s task=%s split=%s", model_hash, task, split)
            return cache.get(key)
        X, y, g = extract_residue_embeddings(
            encoder=encoder,
            tokenizer=tokenizer,
            sequences=seqs,
            labels=labs,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        if key is not None:
            cache.put(key, X, y, g)
        return X, y, g

    def _extract_scorable(seqs, labs, split: str):
        # Drop AFTER the cache write so the cache stays a faithful record of the
        # split and stays valid if the ignore rule ever changes.
        X, y, g = _extract_or_load(seqs, labs, split)
        X_kept, y_kept, g_kept = drop_ignored_residues(X, y, g)
        if y_kept.size != y.size:
            logger.info(
                "  %s/%s: %d of %d residues carry no ground truth; excluded",
                task,
                split,
                y.size - y_kept.size,
                y.size,
            )
        return X_kept, y_kept, g_kept

    X_tr, y_tr, g_tr = _extract_scorable(train_sequences, train_labels, "train")
    problem_type = "binary" if len(np.unique(y_tr)) == 2 else "multiclass"

    def _fit_score(X_a, y_a, X_b, y_b):
        from protein_benchmark_suite import _make_probe_model_for_training_size, timed_fit

        probe = _make_probe_model_for_training_size(probe_type, problem_type, len(X_a))
        fit_seconds = timed_fit(probe, X_a, y_a)
        m = _compute_metrics(
            y_b, probe.predict(X_b), main_metric=cfg.main_metric, problem_type=problem_type
        )
        m["ProbeFitSec"] = fit_seconds
        return m

    if test_sequences is not None and test_labels is not None:
        X_te, y_te, _ = _extract_scorable(test_sequences, test_labels, "test")
        return _fit_score(X_tr, y_tr, X_te, y_te)

    fold_metrics: List[Dict[str, float]] = []
    for tr_idx, te_idx in GroupKFold(n_splits=4).split(X_tr, y_tr, groups=g_tr):
        fold_metrics.append(_fit_score(X_tr[tr_idx], y_tr[tr_idx], X_tr[te_idx], y_tr[te_idx]))
    keys = set().union(*(m.keys() for m in fold_metrics))
    return {k: float(np.mean([m[k] for m in fold_metrics if k in m])) for k in keys}
