from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ptm_mamba_assets import (
    audit_ptm_mamba,
    load_ptm_mamba,
    protein_disjoint_split,
    score_residue_type_baseline,
)

FIELDS = ("Unnamed: 0", "AC_ID", "pos", "label", "ori_seq", "token")


def _write(path: Path, rows: list[tuple[str, int, str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, (accession, position, label, sequence) in enumerate(rows):
            writer.writerow(
                {
                    "Unnamed: 0": index,
                    "AC_ID": accession,
                    "pos": position,
                    "label": label,
                    "ori_seq": sequence,
                    "token": f"<{label}>",
                }
            )


def test_loader_validates_zero_based_site_positions(tmp_path: Path) -> None:
    _write(tmp_path / "split_data_train.csv", [("P1", 1, "Phosphoserine", "ASA")])
    rows = load_ptm_mamba(tmp_path)
    assert len(rows) == 1
    assert rows[0].position == 1
    assert rows[0].residue == "S"

    _write(tmp_path / "split_data_train.csv", [("P1", 3, "Phosphoserine", "ASA")])
    with pytest.raises(ValueError, match="position 3 outside sequence"):
        load_ptm_mamba(tmp_path)


def test_audit_exposes_author_split_leakage_and_site_disjointness(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "split_data_train.csv", [("P1", 1, "type-a", "ASA")])
    _write(
        tmp_path / "split_data_val.csv",
        [("P2", 1, "type-a", "ASA")],
    )
    _write(
        tmp_path / "split_data_test.csv",
        [("P1", 2, "type-b", "ASA"), ("P3", 0, "type-a", "KAA")],
    )

    report = audit_ptm_mamba(tmp_path)
    assert report["overlap"]["train_test"] == {
        "accessions": 1,
        "exact_sequences": 1,
        "accession_positions": 0,
    }


def test_new_split_keeps_accession_and_exact_sequence_components_together(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "split_data_train.csv",
        [("P1", 1, "type-a", "ASA"), ("P2", 0, "type-b", "KAA")],
    )
    _write(tmp_path / "split_data_val.csv", [("P3", 2, "type-a", "ASA")])
    _write(tmp_path / "split_data_test.csv", [("P2", 1, "type-b", "KTA")])
    rows = load_ptm_mamba(tmp_path)

    first = protein_disjoint_split(rows, seed=17)
    second = protein_disjoint_split(rows, seed=17)
    assert first == second
    by_accession = {}
    by_sequence = {}
    for row, split in zip(rows, first, strict=True):
        by_accession.setdefault(row.accession, set()).add(split)
        by_sequence.setdefault(row.sequence_sha256, set()).add(split)
    assert all(len(splits) == 1 for splits in by_accession.values())
    assert all(len(splits) == 1 for splits in by_sequence.values())


def test_residue_type_baseline_scores_known_site_type_without_embeddings(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "split_data_train.csv",
        [
            ("P1", 0, "type-a", "KAA"),
            ("P2", 0, "type-a", "KTA"),
            ("P3", 0, "type-b", "KSA"),
        ],
    )
    _write(tmp_path / "split_data_test.csv", [("P4", 0, "type-a", "KGA")])
    rows = load_ptm_mamba(tmp_path)
    splits = [row.author_split for row in rows]
    metrics = score_residue_type_baseline(rows, splits)
    assert metrics["n_total"] == 1
    assert metrics["top1"] == 1.0
