from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from ptm_inputs import audit_ptm_labels, canonical_ptm_family, iter_ptm_sites


def _write_fixture(path: Path) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["", "AC_ID", "pos", "label", "ori_seq", "token"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "": 1,
                    "AC_ID": "P1",
                    "pos": 1,
                    "label": "Phosphoserine",
                    "ori_seq": "MSK",
                    "token": "<Phosphoserine>",
                },
                {
                    "": 2,
                    "AC_ID": "P2",
                    "pos": 0,
                    "label": "N6-succinyllysine",
                    "ori_seq": "KAA",
                    "token": "<N6-succinyllysine>",
                },
                {
                    "": 3,
                    "AC_ID": "P3",
                    "pos": 2,
                    "label": "Sulfotyrosine",
                    "ori_seq": "AAY",
                    "token": "<Sulfotyrosine>",
                },
            ]
        )


def test_single_gzip_layer_with_double_suffix_and_zero_based_positions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ptm_labels.csv.gz.gz"
    _write_fixture(path)
    rows = list(iter_ptm_sites(path))
    assert [row.residue for row in rows] == ["S", "K", "Y"]


def test_ptm_mapping_audit_separates_native_broad_and_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "ptm_labels.csv.gz.gz"
    _write_fixture(path)
    fine_vocab = {"Phosphoserine": ("[PTM_PHOSPHO_S]", "S")}
    report = audit_ptm_labels(path, fine_vocab=fine_vocab)
    assert report["rows"] == 3
    assert report["unique_accessions"] == 3
    assert report["native_supported_rows"] == 1
    assert report["broad_supported_rows"] == 2
    assert report["any_supported_rows"] == 2
    assert report["source_token_mismatches"] == 0
    assert "not a held-out benchmark" in str(report["source"])


@pytest.mark.parametrize(
    ("label", "residue", "expected"),
    [
        ("Phosphothreonine", "T", "phospho_ST"),
        ("Phosphotyrosine", "Y", "phospho_Y"),
        ("O-linked (GalNAc...) threonine", "T", "o_glyc_ST"),
        ("N-linked (GlcNAc...) asparagine", "N", "n_glyc_N"),
        ("N6-succinyllysine", "K", "rare_k_acyl"),
        ("Sulfotyrosine", "Y", "other"),
    ],
)
def test_canonical_ptm_family(label: str, residue: str, expected: str) -> None:
    assert canonical_ptm_family(label, residue) == expected


def test_out_of_range_position_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv.gz"
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["AC_ID", "pos", "label", "ori_seq", "token"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "AC_ID": "P1",
                "pos": 3,
                "label": "Phosphoserine",
                "ori_seq": "MSK",
                "token": "<Phosphoserine>",
            }
        )
    with pytest.raises(ValueError, match="zero-based position"):
        list(iter_ptm_sites(path))
