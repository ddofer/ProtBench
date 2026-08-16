"""Contact-map label construction and precision-at-L metrics.

Offline: every fixture is synthetic, no model and no dataset download.
"""

import numpy as np
import pytest

from contact_metrics import (
    MIN_SEPARATION,
    SEPARATION_BANDS,
    apc,
    average_contact_metrics,
    contact_metrics,
    contacts_from_tertiary,
    precision_at_l,
    separation_matrix,
)


def _line_coords(length, spacing=3.0):
    """Residues on a straight line: contacts are exactly the near-diagonal."""
    coords = np.zeros((length, 3))
    coords[:, 0] = np.arange(length) * spacing
    return coords


def test_contacts_from_tertiary_threshold():
    coords = _line_coords(10, spacing=3.0)
    contacts, _ = contacts_from_tertiary(coords, threshold=8.0)
    # |i-j| * 3.0 < 8.0  =>  |i-j| <= 2
    expected = separation_matrix(10) <= 2
    assert (contacts == expected).all()


def test_valid_mask_excludes_unresolved_residues():
    coords = _line_coords(12)
    mask = np.ones(12, dtype=bool)
    mask[3] = False
    _, valid = contacts_from_tertiary(coords, mask)
    assert not valid[3].any()
    assert not valid[:, 3].any()
    assert valid[1, 5]


def test_valid_pair_excludes_self_pairs():
    _, valid = contacts_from_tertiary(_line_coords(8))
    assert not np.diag(valid).any()


def test_valid_mask_length_mismatch_raises():
    with pytest.raises(ValueError, match="valid_mask length"):
        contacts_from_tertiary(_line_coords(8), np.ones(7, dtype=bool))


def test_non_3d_coordinates_raise():
    with pytest.raises(ValueError, match=r"must be \(L, 3\)"):
        contacts_from_tertiary(np.zeros((8, 2)))


def test_precision_at_l_known_answer():
    """Hand-built 20x20 map: 4 planted long-range contacts, scores rank 2 of
    them first. Top-L/5 = top-4, so precision must be exactly 2/4."""
    length = 20
    contacts = np.zeros((length, length), dtype=bool)
    valid = np.ones((length, length), dtype=bool)
    np.fill_diagonal(valid, False)
    planted = [(0, 19), (1, 18), (2, 17), (3, 16)]
    for i, j in planted:
        contacts[i, j] = contacts[j, i] = True

    scores = np.zeros((length, length))
    # Two true contacts scored highest, then two decoys, then everything else.
    ranked = [(0, 19), (1, 18), (0, 15), (1, 14)]
    for rank, (i, j) in enumerate(ranked):
        scores[i, j] = scores[j, i] = 10.0 - rank

    got = precision_at_l(scores, contacts, valid, k=5, min_sep=24)
    assert np.isnan(got), "no pair in a 20-residue protein has separation >= 24"

    got = precision_at_l(scores, contacts, valid, k=5, min_sep=12)
    assert got == pytest.approx(0.5), f"expected 2 of top-4 correct, got {got}"


def test_precision_counts_each_pair_once():
    """A symmetric score matrix must not let one contact fill two top-L slots."""
    length = 20
    contacts = np.zeros((length, length), dtype=bool)
    valid = np.ones((length, length), dtype=bool)
    np.fill_diagonal(valid, False)
    contacts[0, 19] = contacts[19, 0] = True
    scores = np.zeros((length, length))
    scores[0, 19] = scores[19, 0] = 1.0
    # Top-L/5 = 4 pairs, exactly one of which is the single true contact.
    assert precision_at_l(scores, contacts, valid, k=5, min_sep=12) == pytest.approx(
        0.25
    )


def test_empty_band_returns_nan_not_zero():
    """A protein too short for a band must be droppable, not scored zero."""
    contacts, valid = contacts_from_tertiary(_line_coords(10))
    scores = np.zeros((10, 10))
    assert np.isnan(precision_at_l(scores, contacts, valid, k=1, min_sep=24))


def test_contact_metrics_reports_every_band_and_k():
    contacts, valid = contacts_from_tertiary(_line_coords(60))
    metrics = contact_metrics(np.zeros((60, 60)), contacts, valid)
    for band in SEPARATION_BANDS:
        for label in ("P@L", "P@L/2", "P@L/5"):
            assert f"{label}_{band}" in metrics


def test_average_skips_nan_bands():
    per_protein = [
        {"P@L/5_long": 0.4, "P@L/5_short": float("nan")},
        {"P@L/5_long": 0.6, "P@L/5_short": 0.2},
    ]
    avg = average_contact_metrics(per_protein)
    assert avg["P@L/5_long"] == pytest.approx(0.5)
    assert avg["P@L/5_short"] == pytest.approx(0.2)


def test_average_of_all_nan_is_zero_not_nan():
    avg = average_contact_metrics([{"P@L_long": float("nan")}])
    assert avg["P@L_long"] == 0.0


def test_apc_preserves_symmetry_and_handles_zero():
    rng = np.random.RandomState(0)
    mat = rng.rand(12, 12)
    sym = mat + mat.T
    corrected = apc(sym)
    assert np.allclose(corrected, corrected.T)
    assert np.allclose(apc(np.zeros((5, 5))), 0.0)


def test_separation_bands_are_half_open_and_contiguous():
    assert SEPARATION_BANDS["short"][0] == MIN_SEPARATION
    assert SEPARATION_BANDS["short"][1] == SEPARATION_BANDS["medium"][0]
    assert SEPARATION_BANDS["medium"][1] == SEPARATION_BANDS["long"][0]
    assert SEPARATION_BANDS["long"][1] is None
