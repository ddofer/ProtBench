from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ptm_site_assets import (
    NHAC_WINDOWS,
    audit_nhac,
    audit_proteinbert_phosphosite,
    load_nhac,
    load_proteinbert_phosphosite,
)


def _write(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq", "label"])
        writer.writerows(rows)


def test_length_filter_counts_overlap_conflicts_and_clean_test(tmp_path: Path) -> None:
    shared = "A" * 4
    _write(tmp_path / "PhosphositePTM.train.csv", [(shared, "0100"), ("AA", "00")])
    _write(tmp_path / "PhosphositePTM.valid.csv", [("CCCC", "0010")])
    _write(
        tmp_path / "PhosphositePTM.test.csv",
        [(shared, "0001"), ("CCCC", "0010"), ("GGGG", "1000")],
    )
    report, clean = audit_proteinbert_phosphosite(tmp_path, min_length=4, max_length=4)
    assert report["splits"]["train"]["raw_rows"] == 2
    assert report["splits"]["train"]["filtered_rows"] == 1
    assert report["exact_overlap"]["train_test"] == {
        "exact_sequence_hashes": 1,
        "row_pairs": 1,
        "conflicting_label_row_pairs": 1,
    }
    assert report["deduplicated_test"]["rows"] == 1
    assert [row.sequence for row in clean] == ["GGGG"]


def test_misaligned_labels_fail_after_filter(tmp_path: Path) -> None:
    for split in ("train", "valid", "test"):
        rows = [("AAAA", "000")] if split == "train" else [("CCCC", "0000")]
        _write(tmp_path / f"PhosphositePTM.{split}.csv", rows)
    with pytest.raises(ValueError, match="sequence/label lengths"):
        load_proteinbert_phosphosite(tmp_path, min_length=4, max_length=4)


def test_nhac_audit_preserves_class_balance_and_finds_split_overlap(tmp_path: Path) -> None:
    path = tmp_path / "NHAC.csv"
    fieldnames = ["unique_id", *(f"seq_{size}" for size in NHAC_WINDOWS), "label", "set"]

    def row(unique_id: str, label: str, split: str) -> dict[str, str]:
        values = {"unique_id": unique_id, "label": label, "set": split}
        values.update(
            {
                f"seq_{size}": "A" * (size // 2) + "K" + "G" * (size // 2)
                for size in NHAC_WINDOWS
            }
        )
        return values

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row("P1", "1", "train"), row("P1", "1", "test")])
    report = audit_nhac(path)
    assert report["rows"] == 2
    assert report["positive"] == 2
    assert report["duplicate_unique_id_rows"] == 1
    assert report["exact_split_overlap"]["train_test"]["seq_61"] == 1
    assert report["paper_split_counts_match"] is False
    splits = load_nhac(path, window_size=11)
    assert splits["train"][0].sequence[5] == "K"
    assert splits["test"][0].label == 1
