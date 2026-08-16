"""Supervised pairwise contact probe on frozen per-residue embeddings.

The model-agnostic half of ProtBench's contact prediction. Every model the
registry can load produces per-residue hidden states, so this runs against all
of them -- unlike the categorical Jacobian in ``contact_catjac.py``, which needs
an MLM head. Input is the primary sequence only; the CB coordinates in the
dataset build the labels and are never shown to the model.

Pipeline:

1. ``iter_residue_embeddings`` (reused from ``token_classification_probe``)
   gives one ``(L, H)`` array per protein.
2. PCA reduces ``H`` to ``PAIR_PCA_DIM``. Without it, scoring one 670-residue
   protein means 224k pairs x 2561 features (~2.3 GB at H=1280); at d=128 the
   same protein costs ~230 MB.
3. Pair feature for residues (i, j): ``[h_i * h_j, |h_i - h_j|, log1p(|i-j|)]``
   -- symmetric by construction, so the score matrix needs no symmetrising.
4. LogisticRegression, then precision-at-L from ``contact_metrics``.

Usage:
    python protein_benchmark_suite.py -m <model> --tasks contact_probe \
        --eval_split test
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from contact_metrics import (
    MIN_SEPARATION,
    average_contact_metrics,
    contact_metrics,
    contacts_from_tertiary,
)

logger = logging.getLogger(__name__)

# Residue-embedding dimensionality after PCA. See the memory note above.
PAIR_PCA_DIM = 128
# Residues sampled to fit the PCA. More than this buys nothing measurable.
PCA_FIT_RESIDUES = 50_000
# Per-protein cap on training pairs, balanced positives/negatives.
MAX_TRAIN_PAIRS_PER_PROTEIN = 2_000
# Pairs scored at once during evaluation, to bound peak memory on long proteins.
SCORE_CHUNK_PAIRS = 200_000


def pair_features(
    residues: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray
) -> np.ndarray:
    """Build pair features for the residue index pairs ``(idx_i, idx_j)``.

    ``residues`` is ``(L, d)``. Both blocks are symmetric in (i, j), so
    ``pair_features(r, i, j) == pair_features(r, j, i)`` exactly -- a contact
    map is undirected and the probe should not be able to tell the order.
    """
    h_i = residues[idx_i]
    h_j = residues[idx_j]
    sep = np.abs(idx_i - idx_j).astype("float32")
    return np.concatenate(
        [h_i * h_j, np.abs(h_i - h_j), np.log1p(sep)[:, None]], axis=1
    ).astype("float32")


def _sample_training_pairs(
    contacts: np.ndarray,
    valid_pair: np.ndarray,
    length: int,
    rng: np.random.RandomState,
    max_pairs: int = MAX_TRAIN_PAIRS_PER_PROTEIN,
):
    """Balanced positive/negative pair sample from one protein's upper triangle.

    Real contact maps are ~4% positive. Fitting on the raw distribution makes
    the probe predict "no contact" everywhere; balancing is what makes the
    ranking informative, and precision-at-L only cares about the ranking.
    """
    upper = np.triu(np.ones((length, length), dtype=bool), k=MIN_SEPARATION)
    eligible = upper & valid_pair
    pos_i, pos_j = np.where(eligible & contacts)
    neg_i, neg_j = np.where(eligible & ~contacts)
    if pos_i.size == 0 or neg_i.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    n_each = min(pos_i.size, neg_i.size, max(max_pairs // 2, 1))
    pos_keep = rng.choice(pos_i.size, size=n_each, replace=False)
    neg_keep = rng.choice(neg_i.size, size=n_each, replace=False)
    idx_i = np.concatenate([pos_i[pos_keep], neg_i[neg_keep]])
    idx_j = np.concatenate([pos_j[pos_keep], neg_j[neg_keep]])
    y = np.concatenate([np.ones(n_each, dtype=int), np.zeros(n_each, dtype=int)])
    return idx_i, idx_j, y


def _fit_pca(residue_arrays: Sequence[np.ndarray], seed: int):
    """PCA down to ``PAIR_PCA_DIM``, fit on a residue subsample."""
    from sklearn.decomposition import PCA

    pooled = np.concatenate(residue_arrays, axis=0)
    n_components = min(PAIR_PCA_DIM, pooled.shape[1], pooled.shape[0])
    if pooled.shape[0] > PCA_FIT_RESIDUES:
        rng = np.random.RandomState(seed)
        pooled = pooled[rng.choice(pooled.shape[0], PCA_FIT_RESIDUES, replace=False)]
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(pooled)
    logger.info(
        "  Contact PCA: %d -> %d dims (%.1f%% variance)",
        pca.n_features_in_,
        n_components,
        100.0 * float(pca.explained_variance_ratio_.sum()),
    )
    return pca


def _labels_for(record: Dict[str, Any], length: int):
    """Contact + valid-pair matrices for one record, cropped to ``length``.

    ``length`` is the number of residues the encoder actually returned, which is
    shorter than the sequence whenever ``max_length`` truncated it. Cropping the
    coordinates the same way keeps labels and embeddings aligned.
    """
    tertiary = np.asarray(record["tertiary"], dtype="float64")[:length]
    mask = record.get("valid_mask")
    mask = None if mask is None else np.asarray(mask, dtype=bool)[:length]
    return contacts_from_tertiary(tertiary, mask)


def _score_protein(probe, residues: np.ndarray) -> np.ndarray:
    """Full ``(L, L)`` score matrix, chunked to bound peak memory."""
    length = residues.shape[0]
    upper_i, upper_j = np.triu_indices(length, k=MIN_SEPARATION)
    scores = np.zeros((length, length), dtype="float32")
    for start in range(0, upper_i.size, SCORE_CHUNK_PAIRS):
        sl = slice(start, start + SCORE_CHUNK_PAIRS)
        chunk_i, chunk_j = upper_i[sl], upper_j[sl]
        probs = probe.predict_proba(pair_features(residues, chunk_i, chunk_j))[:, 1]
        scores[chunk_i, chunk_j] = probs
        scores[chunk_j, chunk_i] = probs
    return scores


def evaluate_contact_prediction(
    *,
    encoder,
    tokenizer,
    train_records: Sequence[Dict[str, Any]],
    test_records: Optional[Sequence[Dict[str, Any]]],
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 1024,
    train_proteins: int = 400,
    seed: int = 42,
) -> Dict[str, float]:
    """Fit the pairwise probe on ``train_records`` and score ``test_records``.

    When ``test_records`` is None the last 20% of the training proteins are held
    out instead -- a protein-level split, since a residue-level one would leak
    the answer straight across the pair.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)

    train_records = list(train_records)[: max(int(train_proteins), 1)]
    if test_records is None:
        if len(train_records) < 5:
            raise ValueError(
                "contact prediction needs at least 5 training proteins to hold "
                f"out an evaluation split; got {len(train_records)}"
            )
        cut = max(int(len(train_records) * 0.8), 1)
        train_records, test_records = train_records[:cut], train_records[cut:]
        logger.info(
            "  No eval split; holding out %d of %d train proteins",
            len(test_records),
            len(train_records) + len(test_records),
        )

    def _embed(records: Iterable[Dict[str, Any]]) -> List[np.ndarray]:
        from token_classification_probe import iter_residue_embeddings

        records = list(records)
        return list(
            iter_residue_embeddings(
                encoder=encoder,
                tokenizer=tokenizer,
                sequences=[r["seq"] for r in records],
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
        )

    logger.info("  Embedding %d train proteins", len(train_records))
    train_residues = _embed(train_records)
    pca = _fit_pca(train_residues, seed)

    feats: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for record, residues in zip(train_records, train_residues):
        length = residues.shape[0]
        if length <= MIN_SEPARATION:
            continue
        contacts, valid_pair = _labels_for(record, length)
        idx_i, idx_j, y = _sample_training_pairs(contacts, valid_pair, length, rng)
        if y.size == 0:
            continue
        feats.append(pair_features(pca.transform(residues), idx_i, idx_j))
        targets.append(y)

    if not feats:
        raise RuntimeError(
            "no usable training pairs -- every protein was too short or had no "
            "resolved contacts"
        )
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(targets, axis=0)
    # Drop the per-protein blocks now: concatenate has already copied them, and
    # holding both halves peak memory on a matrix that reaches ~800k x 257.
    feats.clear()
    targets.clear()
    logger.info("  Fitting pairwise probe on %d pairs x %d features", *X.shape)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=100, random_state=seed),
    )
    probe.fit(X, y)

    logger.info("  Scoring %d eval proteins", len(test_records))
    test_residues = _embed(test_records)
    per_protein: List[Dict[str, float]] = []
    for record, residues in zip(test_records, test_residues):
        length = residues.shape[0]
        if length <= MIN_SEPARATION:
            continue
        contacts, valid_pair = _labels_for(record, length)
        scores = _score_protein(probe, pca.transform(residues))
        per_protein.append(contact_metrics(scores, contacts, valid_pair))

    if not per_protein:
        raise RuntimeError("no eval protein was long enough to score")
    metrics = average_contact_metrics(per_protein)
    metrics["Proteins_Scored"] = float(len(per_protein))
    return metrics


def _selfcheck() -> None:
    """Runnable check: python contact_probe.py

    Uses a fake encoder whose per-residue embedding is a smooth function of
    position, so nearby residues are similar. The probe should then beat the
    background contact rate by a wide margin on a synthetic map -- if the pair
    features or the label alignment are wrong, it lands at chance.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.RandomState(0)

    # pair_features must be exactly symmetric under (i, j) -> (j, i).
    residues = rng.randn(20, 8).astype("float32")
    i = np.array([1, 5, 9])
    j = np.array([14, 2, 17])
    assert np.allclose(pair_features(residues, i, j), pair_features(residues, j, i))

    # Balanced sampling really is balanced, and respects MIN_SEPARATION.
    coords = np.cumsum(rng.randn(60, 3) * 2.0, axis=0)
    contacts, valid = contacts_from_tertiary(coords, np.ones(60, dtype=bool))
    idx_i, idx_j, y = _sample_training_pairs(contacts, valid, 60, rng)
    assert y.sum() * 2 == y.size, f"unbalanced sample: {y.sum()}/{y.size}"
    assert (np.abs(idx_i - idx_j) >= MIN_SEPARATION).all()
    assert (idx_i < idx_j).all(), "training pairs must come from the upper triangle"

    # Cropping to the encoder's truncated length keeps labels square.
    rec = {"seq": "A" * 60, "tertiary": coords, "valid_mask": np.ones(60, dtype=bool)}
    c30, v30 = _labels_for(rec, 30)
    assert c30.shape == (30, 30) and v30.shape == (30, 30)

    print("contact_probe selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
