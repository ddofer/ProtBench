from pathlib import Path

from ptm_dbptm_assets import (
    audit_dbptm_task,
    load_dbptm_windows,
    record_group_split,
)


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{header}\n{sequence}\n" for header, sequence in records))


def test_loader_preserves_labels_and_parses_header_position(tmp_path: Path):
    pos = tmp_path / "Demo_pos.fasta"
    neg = tmp_path / "Demo_neg.fasta"
    _write_fasta(pos, [("P12345_HUMAN_42", "AAAAAAAAAAKAAAAAAAAAA")])
    _write_fasta(neg, [("P67890_HUMAN_7", "AAAAAAAAAAKAAAAAAAAAA")])

    rows = load_dbptm_windows(
        task="demo",
        positive_path=pos,
        negative_path=neg,
        expected_center_residues=frozenset("K"),
    )

    assert [
        (row.label, row.record_id, row.site_position, row.center_residue)
        for row in rows
    ] == [
        (1, "P12345_HUMAN", 42, "K"),
        (0, "P67890_HUMAN", 7, "K"),
    ]


def test_audit_reports_exact_label_conflicts_and_off_target_centers(tmp_path: Path):
    pos = tmp_path / "Demo_pos.fasta"
    neg = tmp_path / "Demo_neg.fasta"
    _write_fasta(
        pos,
        [
            ("P12345_HUMAN_42", "AAAAAAAAAAKAAAAAAAAAA"),
            ("P12345_HUMAN_43", "AAAAAAAAAASAAAAAAAAAA"),
        ],
    )
    _write_fasta(neg, [("P67890_HUMAN_7", "AAAAAAAAAAKAAAAAAAAAA")])

    report = audit_dbptm_task(
        task="demo",
        positive_path=pos,
        negative_path=neg,
        expected_center_residues=frozenset("K"),
    )

    assert report["rows"] == 3
    assert report["positive"] == 2
    assert report["negative"] == 1
    assert report["exact_window_label_conflicts"] == 1
    assert report["off_target_center"] == {"negative": 0, "positive": 1}


def test_record_group_split_is_deterministic_and_keeps_records_together(tmp_path: Path):
    pos = tmp_path / "Demo_pos.fasta"
    neg = tmp_path / "Demo_neg.fasta"
    _write_fasta(
        pos,
        [
            ("P1_HUMAN_42", "AAAAAAAAAAKAAAAAAAAAA"),
            ("P2_HUMAN_43", "AAAAAAAAAAKAAAAAAAAAA"),
        ],
    )
    _write_fasta(
        neg,
        [
            ("P1_HUMAN_44", "AAAAAAAAAAKAAAAAAAAAA"),
            ("P3_HUMAN_45", "AAAAAAAAAAKAAAAAAAAAA"),
        ],
    )
    rows = load_dbptm_windows(
        task="demo",
        positive_path=pos,
        negative_path=neg,
        expected_center_residues=frozenset("K"),
    )

    first = record_group_split(rows, seed=17)
    assert first == record_group_split(rows, seed=17)
    by_record: dict[str, set[str]] = {}
    for row, split in zip(rows, first, strict=True):
        by_record.setdefault(row.record_id, set()).add(split)
    assert all(len(splits) == 1 for splits in by_record.values())
