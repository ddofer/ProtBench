from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ptm_ppi_assets import REQUIRED_COLUMNS, audit_ptmint, load_ptmint


def _row(**updates: str) -> dict[str, str]:
    row = {
        "Uniprot": "U1",
        "PTM": "Phos",
        "Site": "2",
        "AA": "C",
        "Int_uniprot": "I1",
        "Effect": "Enhance",
        "Gene_sequence": "ACDE",
        "Int_gene_sequence": "LMNP",
        "cluster_ID": "c0",
        "split": "train",
    }
    row.update(updates)
    return row


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_ptmint_audit_finds_split_overlap_and_invalid_events(tmp_path: Path) -> None:
    path = tmp_path / "ptmint.csv"
    _write(
        path,
        [
            _row(),
            _row(split="test"),
            _row(
                Site="3",
                AA="D",
                Int_uniprot="I2",
                Int_gene_sequence="QRST",
                Effect="Inhibit",
                cluster_ID="c1",
                split="validation",
            ),
            _row(
                Uniprot="U2",
                Gene_sequence="AAA",
                Site="4",
                AA="A",
                Int_uniprot="I3",
                Int_gene_sequence="WXYZ",
                cluster_ID="c2",
                split="test",
            ),
            _row(
                Uniprot="U2",
                Gene_sequence="AAA",
                Site="2",
                AA="C",
                Int_uniprot="I4",
                Int_gene_sequence="PPPP",
                cluster_ID="c3",
            ),
        ],
    )

    report, candidates = audit_ptmint(path, enforce_expected_counts=False)
    assert report["rows"] == 5
    assert report["data_quality"] == {
        "invalid_coordinate_rows": [3],
        "residue_mismatch_rows": [4],
        "duplicate_event_rows": 1,
        "valid_event_deduplicated_rows": 2,
    }
    assert report["exact_values_present_multiple_splits"] == {
        "target_sequences": 2,
        "ordered_sequence_pairs": 1,
    }
    assert report["provided_split_overlap"]["train_test"] == {
        "exact_target_sequences": 2,
        "exact_ordered_sequence_pairs": 1,
    }
    assert report["clusters_present_in_multiple_splits"] == 1
    assert [row.row_index for row in candidates] == [0, 2]


def test_ptmint_loader_rejects_unknown_split(tmp_path: Path) -> None:
    path = tmp_path / "ptmint.csv"
    _write(path, [_row(split="future")])
    with pytest.raises(ValueError, match="invalid split"):
        load_ptmint(path, enforce_expected_counts=False)


def test_ptmint_loader_rejects_multi_residue_site_label(tmp_path: Path) -> None:
    path = tmp_path / "ptmint.csv"
    _write(path, [_row(AA="ST")])
    with pytest.raises(ValueError, match="AA must be one residue"):
        load_ptmint(path, enforce_expected_counts=False)
