"""Feature protocols for PTM effects on protein-protein interactions.

The PTM-Mamba supplementary head is retained only as a labeled reproduction:
subtracting ``concat(binder, wt) - concat(binder, ptm)`` cancels the binder
exactly.  The primary frozen-probe protocol below keeps explicit binder-target
interaction terms so predictions can depend on the binding partner.
"""

from __future__ import annotations

import numpy as np


def _validate_embeddings(
    binder: np.ndarray, wild_type: np.ndarray, modified: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    binder_array = np.asarray(binder, dtype=np.float32)
    wild_type_array = np.asarray(wild_type, dtype=np.float32)
    modified_array = np.asarray(modified, dtype=np.float32)
    arrays = (binder_array, wild_type_array, modified_array)
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("PPI embeddings must each have shape (N, H)")
    if arrays[0].shape != arrays[1].shape or arrays[0].shape != arrays[2].shape:
        raise ValueError(f"PPI embedding shapes differ: {[array.shape for array in arrays]}")
    return arrays


def paper_reproduction_features(
    binder: np.ndarray, wild_type: np.ndarray, modified: np.ndarray
) -> np.ndarray:
    """Reproduce Supplementary Code 4, including exact binder cancellation."""

    binder, wild_type, modified = _validate_embeddings(binder, wild_type, modified)
    binder_wt = np.concatenate([binder, wild_type], axis=-1)
    binder_ptm = np.concatenate([binder, modified], axis=-1)
    return binder_wt - binder_ptm


def binder_aware_features(
    binder: np.ndarray, wild_type: np.ndarray, modified: np.ndarray
) -> np.ndarray:
    """Construct frozen-probe features that retain partner-specific PTM effects.

    Alongside the three embeddings and the PTM delta, Hadamard interaction terms
    expose how the same wild-type/modified target relates to different binders.
    A linear classifier on these features is therefore binder-aware without
    updating any encoder weights.
    """

    binder, wild_type, modified = _validate_embeddings(binder, wild_type, modified)
    delta = modified - wild_type
    return np.concatenate(
        [
            binder,
            wild_type,
            modified,
            delta,
            binder * wild_type,
            binder * modified,
            binder * delta,
        ],
        axis=-1,
    )
