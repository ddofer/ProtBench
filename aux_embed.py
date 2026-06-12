"""Aux-head feature extraction for multi-head linear probe embeddings.

When a ProtevaOutput carries auxiliary head predictions, this module extracts
and concatenates them alongside the trunk hidden state into a richer probe
embedding vector:

  - Per-residue heads (dim=3 in packed varlen mode, shape ``(1, T, D)``):
    mean-pooled per segment using ``cu_seqlens`` boundaries.
  - Per-protein heads (dim=2, already segment-pooled in ``forward()``, ``(P, D)``):
    used directly.

All fields in ``ALL_AUX_FIELDS`` are optional — ``None`` values are silently
skipped. This keeps the feature set adaptive to whatever heads a checkpoint was
trained with (Stage-2 vs Stage-3, with/without RTD, etc.).

Usage::

    from aux_embed import extract_aux_features, build_probe_embedding

    aux = extract_aux_features(proteva_output, cu_seqlens_q, _log=True)
    probe_input = build_probe_embedding(trunk_np, aux)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# All ProtevaOutput aux fields in concatenation order.
# Fields absent from the output object (or set to None) are silently skipped.
# dim==3 → per-residue (1, T, D): mean-pooled per segment.
# dim==2 → per-protein (P, D): already segment-pooled in forward().
ALL_AUX_FIELDS: tuple[str, ...] = (
    # Per-residue
    "di3_logits",
    "am_logits",
    "cons_pred",
    "pfam_domain_pred",
    "sites_logits",
    "rtd_logits",
    # Per-protein
    "tax_domain_logits",
    "tax_phylum_logits",
    "tax_class_logits",
    "interpro_logits",
    "go_mf_logits",
    "go_bp_logits",
    "go_cc_logits",
    "plddt_pred",
)


def extract_aux_features(
    output,
    cu_seqlens: torch.Tensor,
    *,
    _log: bool = False,
) -> Optional[torch.Tensor]:
    """Extract and concatenate all non-None aux head tensors from a ProtevaOutput.

    Args:
        output: ProtevaOutput (or any object with optional aux-head attributes).
        cu_seqlens: Cumulative sequence lengths ``(P+1,)`` int32/int64, used to
            mean-pool per-residue tensors over the packed token axis per segment.
        _log: If True, emit an INFO log listing present fields and total aux dim,
            or a WARNING when no heads are found.

    Returns:
        ``(P, D_aux)`` float32 tensor (present heads concatenated), or ``None``
        when no non-None tensor head is found in ``output``.
    """
    seg_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).int()
    parts: list[torch.Tensor] = []
    present: list[str] = []

    for field in ALL_AUX_FIELDS:
        t = getattr(output, field, None)
        if not isinstance(t, torch.Tensor):
            continue
        present.append(field)
        if t.dim() == 3:
            # Per-residue: (1, T, D) → (P, D) via segment mean-pool.
            # seg_lens originates from cu_seqlens (CPU); move to the tensor's device
            # so segment_reduce doesn't raise a device-mismatch on CUDA.
            parts.append(torch.segment_reduce(t[0].float(), "mean", lengths=seg_lens.to(t[0].device)))
        else:
            # Per-protein: (P, D) already pooled.
            parts.append(t.float())

    if not parts:
        if _log:
            logger.warning("aux_embed: no non-None aux heads found; falling back to trunk")
        return None

    if _log:
        total_dim = sum(p.shape[-1] for p in parts)
        logger.info("aux_embed fields=%s total_dim=%d", present, total_dim)

    return torch.cat(parts, dim=-1)


def build_probe_embedding(
    trunk: np.ndarray,
    aux: Optional[torch.Tensor],
) -> np.ndarray:
    """Concatenate trunk and aux embeddings into a single probe input array.

    Args:
        trunk: ``(P, H)`` float32 numpy array (mean-pooled trunk hidden state).
        aux: ``(P, D_aux)`` torch tensor from ``extract_aux_features``, or ``None``.

    Returns:
        ``(P, H + D_aux)`` numpy array when ``aux`` is provided, otherwise
        ``trunk`` unchanged (same object).
    """
    if aux is None:
        return trunk
    return np.concatenate([trunk, aux.detach().cpu().numpy()], axis=-1)
