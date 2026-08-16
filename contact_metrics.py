"""Residue-residue contact map labels and precision-at-L metrics.

Shared by both contact paths:

* ``contact_probe.py``   -- supervised pairwise probe inside the benchmark suite
* ``contact_catjac.py``  -- zero-shot categorical Jacobian, standalone script

Labels come from CB coordinates (``contacts_from_tertiary``); the model only
ever sees the primary sequence. Scoring follows the TAPE / CASP convention:
precision over the top ``L/k`` predicted pairs within a sequence-separation
band, where ``L`` is the full sequence length (not the number of candidate
pairs).

Usage:
    contacts, valid = contacts_from_tertiary(coords, mask)
    metrics = contact_metrics(scores, contacts, valid)   # P@L/5_long, ...
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Sequence-separation bands, (min_sep inclusive, max_sep exclusive).
SEPARATION_BANDS: Dict[str, Tuple[int, Optional[int]]] = {
    "short": (6, 12),
    "medium": (12, 24),
    "long": (24, None),
}

# Denominators for the top-L/k cut.
TOP_K_DIVISORS = (1, 2, 5)

# Pairs closer than this in sequence are trivially predictable and are excluded
# everywhere -- from training samples, from scoring, and from the diagonal band
# that the categorical Jacobian zeroes out.
MIN_SEPARATION = 6

CONTACT_THRESHOLD_ANGSTROM = 8.0


def separation_matrix(length: int) -> np.ndarray:
    """``|i - j|`` for every residue pair."""
    idx = np.arange(length)
    return np.abs(idx[:, None] - idx[None, :])


def contacts_from_tertiary(
    tertiary,
    valid_mask=None,
    threshold: float = CONTACT_THRESHOLD_ANGSTROM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn ``(L, 3)`` CB coordinates into a boolean contact map.

    Returns ``(contacts, valid_pair)``, both ``(L, L)`` bool. A pair is valid
    only when BOTH residues are marked resolved in ``valid_mask``; unresolved
    residues in ProteinNet carry ``(0, 0, 0)`` coordinates, so scoring them
    would invent contacts at the origin.
    """
    coords = np.asarray(tertiary, dtype="float64")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"tertiary must be (L, 3); got {coords.shape}")
    length = coords.shape[0]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    contacts = dist < float(threshold)

    if valid_mask is None:
        valid = np.ones(length, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape[0] != length:
            raise ValueError(
                f"valid_mask length {valid.shape[0]} != coordinate length {length}"
            )
    valid_pair = valid[:, None] & valid[None, :]
    np.fill_diagonal(valid_pair, False)
    return contacts, valid_pair


def apc(mat: np.ndarray) -> np.ndarray:
    """Average product correction (Dunn et al. 2008).

    Removes the per-position background that makes highly-coupled residues look
    coupled to everything. Standard post-processing for any coevolution-style
    score, including the categorical Jacobian.
    """
    mat = np.asarray(mat, dtype="float64")
    total = mat.sum()
    if total == 0:
        return mat.copy()
    row = mat.sum(axis=1, keepdims=True)
    col = mat.sum(axis=0, keepdims=True)
    return mat - (row @ col) / total


def precision_at_l(
    scores: np.ndarray,
    contacts: np.ndarray,
    valid_pair: np.ndarray,
    *,
    k: int,
    min_sep: int,
    max_sep: Optional[int] = None,
) -> float:
    """Precision over the top ``L/k`` scored pairs in a separation band.

    ``L`` is the sequence length, so the cut does not shrink when a band holds
    few candidate pairs -- that is the published convention. Returns NaN when
    the band has no valid pairs at all, so the caller can drop the protein from
    the average rather than scoring it 0.
    """
    length = scores.shape[0]
    sep = separation_matrix(length)
    band = sep >= int(min_sep)
    if max_sep is not None:
        band &= sep < int(max_sep)
    # Upper triangle only: the map is symmetric, and counting both (i,j) and
    # (j,i) would let one true contact fill two slots of the top-L/k budget.
    band &= np.triu(np.ones_like(band, dtype=bool), k=1)
    band &= np.asarray(valid_pair, dtype=bool)

    n_candidates = int(band.sum())
    if n_candidates == 0:
        return float("nan")

    top_n = min(max(length // int(k), 1), n_candidates)
    band_scores = np.asarray(scores, dtype="float64")[band]
    band_truth = np.asarray(contacts, dtype=bool)[band]
    # argpartition is O(n) vs a full sort; the tie order among equal scores is
    # arbitrary either way.
    top_idx = np.argpartition(-band_scores, top_n - 1)[:top_n]
    return float(band_truth[top_idx].mean())


def contact_metrics(
    scores: np.ndarray,
    contacts: np.ndarray,
    valid_pair: np.ndarray,
) -> Dict[str, float]:
    """Every (band, k) combination: ``P@L_short`` ... ``P@L/5_long``."""
    out: Dict[str, float] = {}
    for band, (min_sep, max_sep) in SEPARATION_BANDS.items():
        for k in TOP_K_DIVISORS:
            label = "P@L" if k == 1 else f"P@L/{k}"
            out[f"{label}_{band}"] = precision_at_l(
                scores, contacts, valid_pair, k=k, min_sep=min_sep, max_sep=max_sep
            )
    return out


def average_contact_metrics(per_protein: list) -> Dict[str, float]:
    """Mean over proteins, skipping NaN bands (protein too short for that band).

    Averaging per protein rather than pooling pairs is the convention: a single
    long protein would otherwise dominate the score.

    A band that was NaN for EVERY protein reports 0.0, which reads the same as a
    model that scored zero. That needs every protein to be too short for the
    band -- on the 40-protein ProteinNet test split (min length 75) it cannot
    happen, but check protein lengths before trusting a 0.0 on other data.
    """
    if not per_protein:
        return {}
    keys = sorted(set().union(*(m.keys() for m in per_protein)))
    out: Dict[str, float] = {}
    for key in keys:
        vals = np.array([m.get(key, np.nan) for m in per_protein], dtype="float64")
        vals = vals[np.isfinite(vals)]
        out[key] = float(vals.mean()) if vals.size else 0.0
    return out


def _selfcheck() -> None:
    """Runnable check: python contact_metrics.py"""
    # An oracle scorer must hit the ceiling of every band: top-L/k is a fixed
    # budget, so when a band holds fewer true contacts than slots the best
    # possible precision is n_true/top_n, not 1.0.
    rng = np.random.RandomState(0)
    length = 40
    coords = rng.randn(length, 3) * 10.0
    contacts, valid = contacts_from_tertiary(coords, np.ones(length, dtype=bool))
    oracle = contacts.astype("float64")
    upper = np.triu(np.ones((length, length), dtype=bool), k=1)
    sep = separation_matrix(length)
    for band, (min_sep, max_sep) in SEPARATION_BANDS.items():
        in_band = upper & valid & (sep >= min_sep)
        if max_sep is not None:
            in_band &= sep < max_sep
        n_candidates = int(in_band.sum())
        if n_candidates == 0:
            continue
        n_true = int((contacts & in_band).sum())
        for k in TOP_K_DIVISORS:
            top_n = min(max(length // k, 1), n_candidates)
            got = precision_at_l(
                oracle, contacts, valid, k=k, min_sep=min_sep, max_sep=max_sep
            )
            assert got == min(n_true, top_n) / top_n, f"{band} k={k}: {got}"
            # The inverted scorer picks non-contacts first, so it only scores
            # above zero once the budget outruns the available non-contacts.
            worst = precision_at_l(
                -oracle, contacts, valid, k=k, min_sep=min_sep, max_sep=max_sep
            )
            assert worst == max(0, top_n - (n_candidates - n_true)) / top_n

    # valid_mask must remove pairs touching an unresolved residue.
    mask = np.ones(length, dtype=bool)
    mask[:5] = False
    _, valid2 = contacts_from_tertiary(coords, mask)
    assert not valid2[0].any() and not valid2[:, 0].any()
    assert not np.diag(valid2).any()

    # APC is symmetric on symmetric input and leaves an all-zero matrix alone.
    sym = oracle + oracle.T
    corrected = apc(sym)
    assert np.allclose(corrected, corrected.T)
    assert np.allclose(apc(np.zeros((5, 5))), 0.0)

    # Separation bands are half-open and do not overlap.
    sep = separation_matrix(30)
    assert (sep[np.diag_indices(30)] == 0).all()
    assert SEPARATION_BANDS["short"][1] == SEPARATION_BANDS["medium"][0]

    print("contact_metrics selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
