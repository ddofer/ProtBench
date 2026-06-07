"""TDD tests for aux-head embedding extraction for linear probes.

Covers ``aux_embed.py``:
* ``extract_aux_features`` — dim-based dispatch, per-segment pooling, graceful None handling.
* ``build_probe_embedding`` — trunk + aux concatenation.

All CPU, no model loading, synthetic tensors.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from aux_embed import ALL_AUX_FIELDS, build_probe_embedding, extract_aux_features  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cu(lengths: list[int]) -> torch.Tensor:
    """Build cu_seqlens from a list of segment lengths."""
    t = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    t[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    return t


def _ns(**kwargs) -> types.SimpleNamespace:
    """Build a SimpleNamespace mock ProtevaOutput with given fields; rest are None."""
    all_none = {f: None for f in ALL_AUX_FIELDS}
    all_none.update(kwargs)
    return types.SimpleNamespace(**all_none)


# ---------------------------------------------------------------------------
# extract_aux_features — unit tests
# ---------------------------------------------------------------------------


def test_extract_no_heads_returns_none():
    out = _ns()
    assert extract_aux_features(out, _cu([3, 5])) is None


def test_extract_per_residue_only():
    # di3_logits: (1, T=8, D=20) with 2 segments of len 3 and 5
    t = torch.zeros(1, 8, 20)
    out = _ns(di3_logits=t)
    result = extract_aux_features(out, _cu([3, 5]))
    assert result is not None
    assert result.shape == (2, 20)


def test_extract_per_protein_only():
    # plddt_pred: (P=2, D=1)
    t = torch.ones(2, 1)
    out = _ns(plddt_pred=t)
    result = extract_aux_features(out, _cu([3, 5]))
    assert result is not None
    assert result.shape == (2, 1)


def test_extract_mixed_heads():
    # di3 (3D, D=20) + plddt (2D, D=1) → (2, 21)
    out = _ns(
        di3_logits=torch.zeros(1, 8, 20),
        plddt_pred=torch.ones(2, 1),
    )
    result = extract_aux_features(out, _cu([3, 5]))
    assert result is not None
    assert result.shape == (2, 21)


def test_extract_partial_heads():
    # cons_pred (dim=3, D=1) + tax_domain_logits (dim=2, D=4) → (P, 5)
    out = _ns(
        cons_pred=torch.zeros(1, 6, 1),
        tax_domain_logits=torch.zeros(3, 4),
    )
    result = extract_aux_features(out, _cu([2, 2, 2]))
    assert result is not None
    assert result.shape == (3, 5)


def test_extract_residue_mean_values_correct():
    """Per-segment mean of cons_pred must equal the analytic mean."""
    # Segment 0: tokens [0,1,2] → mean = 1.0
    # Segment 1: tokens [3,4]   → mean = 3.5
    vals = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(1, 5, 1)
    out = _ns(cons_pred=vals)
    result = extract_aux_features(out, _cu([3, 2]))
    assert result is not None
    np.testing.assert_allclose(result[:, 0].numpy(), [1.0, 3.5], rtol=1e-5)


def test_mtp_logits_list_skipped():
    # mtp_logits is a list, not a Tensor — must not raise and must be ignored
    out = _ns(mtp_logits=[torch.zeros(1, 8, 64)])
    # With only the list field, no real tensor heads → None
    assert extract_aux_features(out, _cu([4, 4])) is None


def test_extract_output_is_float32():
    # All parts should be cast to float32
    out = _ns(cons_pred=torch.ones(1, 4, 1, dtype=torch.float16))
    result = extract_aux_features(out, _cu([2, 2]))
    assert result is not None
    assert result.dtype == torch.float32


def test_extract_concat_order():
    """Fields appear in ALL_AUX_FIELDS order: residue-then-protein, matching the tuple."""
    # Only two non-None: di3_logits (D=20) and plddt_pred (D=1)
    # di3 comes before plddt in ALL_AUX_FIELDS → left 20 cols = di3, last col = plddt
    di3_val = torch.full((1, 4, 20), 2.0)
    plddt_val = torch.full((2, 1), 7.0)
    out = _ns(di3_logits=di3_val, plddt_pred=plddt_val)
    result = extract_aux_features(out, _cu([2, 2]))
    assert result is not None
    assert result.shape == (2, 21)
    np.testing.assert_allclose(result[:, :20].numpy(), 2.0)
    np.testing.assert_allclose(result[:, 20:].numpy(), 7.0)


# ---------------------------------------------------------------------------
# build_probe_embedding
# ---------------------------------------------------------------------------


def test_build_no_aux():
    trunk = np.ones((5, 8), dtype=np.float32)
    result = build_probe_embedding(trunk, None)
    assert result is trunk  # same object — no copy


def test_build_with_aux_shape():
    trunk = np.zeros((3, 8), dtype=np.float32)
    aux = torch.ones(3, 4)
    result = build_probe_embedding(trunk, aux)
    assert result.shape == (3, 12)


def test_build_values_correct():
    trunk = np.full((2, 4), 1.0, dtype=np.float32)
    aux = torch.full((2, 2), 9.0)
    result = build_probe_embedding(trunk, aux)
    np.testing.assert_allclose(result[:, :4], 1.0)
    np.testing.assert_allclose(result[:, 4:], 9.0)


# ---------------------------------------------------------------------------
# Integration: mode dispatch logic (mirrors the embed_sequences Proteva path)
# ---------------------------------------------------------------------------


def _embed_via_mode(
    trunk_np: np.ndarray,
    output,
    cu_seqlens: torch.Tensor,
    probe_embed_mode: str,
) -> np.ndarray:
    """Replicate the Proteva embed_sequences dispatch for testing."""
    if probe_embed_mode == "trunk":
        return trunk_np
    aux = extract_aux_features(output, cu_seqlens, _log=False)
    if aux is None or probe_embed_mode == "trunk_and_aux":
        return build_probe_embedding(trunk_np, aux)
    # aux_only
    return aux.detach().cpu().numpy()


def test_trunk_mode_shape_unchanged():
    trunk = np.zeros((2, 8), dtype=np.float32)
    out = _ns(plddt_pred=torch.ones(2, 1))
    result = _embed_via_mode(trunk, out, _cu([3, 5]), "trunk")
    assert result.shape == (2, 8)


def test_trunk_and_aux_mode_wider():
    trunk = np.zeros((2, 8), dtype=np.float32)
    out = _ns(plddt_pred=torch.ones(2, 1))  # D_aux = 1
    result = _embed_via_mode(trunk, out, _cu([3, 5]), "trunk_and_aux")
    assert result.shape == (2, 9)


def test_aux_only_mode():
    trunk = np.zeros((2, 8), dtype=np.float32)
    out = _ns(plddt_pred=torch.ones(2, 3))  # D_aux = 3
    result = _embed_via_mode(trunk, out, _cu([3, 5]), "aux_only")
    assert result.shape == (2, 3)


def test_no_heads_fallback_to_trunk():
    # aux all None + mode=trunk_and_aux → returns trunk unchanged
    trunk = np.ones((2, 8), dtype=np.float32)
    out = _ns()  # all None
    result = _embed_via_mode(trunk, out, _cu([3, 5]), "trunk_and_aux")
    assert result.shape == (2, 8)
    np.testing.assert_array_equal(result, trunk)
