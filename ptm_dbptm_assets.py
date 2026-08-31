"""Audit dbPTM's published 21-residue PTM-site benchmark windows.

These assets are centred peptide windows, not a protein-disjoint benchmark.
They are useful external site-context diagnostics once the encoder-specific
runner is pointed at them, but they must not be presented as corpus-clean or
as a full-protein generalisation result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ptm_inputs import file_sha256

WINDOW_LENGTH = 21


@dataclass(frozen=True)
class DbPTMTaskSpec:
    directory: str
    positive_filename: str
    negative_filename: str
    expected_center_residues: frozenset[str]


DBPTM_TASKS: dict[str, DbPTMTaskSpec] = {
    "acetylation": DbPTMTaskSpec(
        "Acetylation", "Acetylation_pos.fasta", "Acetylation_neg.fasta", frozenset("K")
    ),
    "methylation": DbPTMTaskSpec(
        "Methylation", "Methylation_pos.fasta", "Methylation_neg.fasta", frozenset("KR")
    ),
    "n_linked_glycosylation": DbPTMTaskSpec(
        "N-linkedGlycosylation",
        "N-linkedGlycosylation_pos.fasta",
        "N-linkedGlycosylation_neg.fasta",
        frozenset("N"),
    ),
    "o_linked_glycosylation": DbPTMTaskSpec(
        "O-linkedGlycosylation",
        "O-linkedGlycosylation_pos.fasta",
        "O-linkedGlycosylation_neg.fasta",
        frozenset("ST"),
    ),
    "phosphorylation_by_cdk": DbPTMTaskSpec(
        "PhosphorylationByCDK", "CDK_pos.fasta", "CDK_neg.fasta", frozenset("STY")
    ),
    "s_nitrosylation": DbPTMTaskSpec(
        "S-nitrosylation",
        "S-nitrosylation_pos.fasta",
        "S-nitrosylation_neg.fasta",
        frozenset("C"),
    ),
    "succinylation": DbPTMTaskSpec(
        "Succinylation",
        "Succinylation_pos.fasta",
        "Succinylation_neg.fasta",
        frozenset("K"),
    ),
    "ubiquitination": DbPTMTaskSpec(
        "Ubiquitination",
        "Ubiquitination_pos.fasta",
        "Ubiquitination_neg.fasta",
        frozenset("K"),
    ),
}


@dataclass(frozen=True)
class DbPTMWindow:
    task: str
    label: int
    row_index: int
    header: str
    record_id: str
    site_position: int
    sequence: str

    def __post_init__(self) -> None:
        if self.label not in {0, 1}:
            raise ValueError("dbPTM labels must be binary")
        if len(self.sequence) != WINDOW_LENGTH:
            raise ValueError(f"dbPTM windows must be {WINDOW_LENGTH} residues")

    @property
    def center_residue(self) -> str:
        return self.sequence[WINDOW_LENGTH // 2]

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode()).hexdigest()


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_parts).upper()))
                header = line[1:].strip()
                sequence_parts = []
            elif header is None:
                raise ValueError(f"{path}:{line_number}: sequence before FASTA header")
            else:
                sequence_parts.append("".join(line.split()))
    if header is not None:
        records.append((header, "".join(sequence_parts).upper()))
    if not records:
        raise ValueError(f"{path}: no FASTA records")
    return records


def _parse_header(header: str, *, path: Path, row_index: int) -> tuple[str, int]:
    try:
        record_id, position_text = header.rsplit("_", maxsplit=1)
        position = int(position_text)
    except ValueError as exc:
        raise ValueError(
            f"{path}: record {row_index + 1}: expected trailing _<site_position>"
        ) from exc
    if not record_id or position < 1:
        raise ValueError(f"{path}: record {row_index + 1}: invalid header {header!r}")
    return record_id, position


def load_dbptm_windows(
    *,
    task: str,
    positive_path: Path,
    negative_path: Path,
    expected_center_residues: frozenset[str],
) -> list[DbPTMWindow]:
    """Load labelled dbPTM windows without silently filtering annotation noise."""

    if not task:
        raise ValueError("task must be non-empty")
    if not expected_center_residues:
        raise ValueError("expected_center_residues must be non-empty")
    rows: list[DbPTMWindow] = []
    for label, path in ((1, positive_path), (0, negative_path)):
        for row_index, (header, sequence) in enumerate(_read_fasta(path)):
            if len(sequence) != WINDOW_LENGTH:
                raise ValueError(
                    f"{path}: record {row_index + 1}: expected {WINDOW_LENGTH} residues, "
                    f"got {len(sequence)}"
                )
            if not sequence.isalpha():
                raise ValueError(f"{path}: record {row_index + 1}: non-letter sequence")
            record_id, site_position = _parse_header(
                header, path=path, row_index=row_index
            )
            rows.append(
                DbPTMWindow(
                    task=task,
                    label=label,
                    row_index=row_index,
                    header=header,
                    record_id=record_id,
                    site_position=site_position,
                    sequence=sequence,
                )
            )
    return rows


def record_group_split(
    rows: Sequence[DbPTMWindow], *, seed: int = 1337
) -> list[str]:
    """Assign dbPTM entry-name groups to deterministic 70/15/15 splits.

    The source archive has no official train/test assignment. Grouping by the
    FASTA entry name prevents windows from one source protein record appearing
    on both sides of a diagnostic split. It does not establish remote-homology
    or pretraining-corpus separation.
    """

    assignments: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row.task, row.record_id)
        if key in assignments:
            continue
        bucket = int.from_bytes(
            hashlib.sha256(f"{seed}\0{row.task}\0{row.record_id}".encode()).digest()[:8],
            "big",
        ) % 100
        assignments[key] = (
            "train" if bucket < 70 else "validation" if bucket < 85 else "test"
        )
    return [assignments[(row.task, row.record_id)] for row in rows]


def audit_dbptm_task(
    *,
    task: str,
    positive_path: Path,
    negative_path: Path,
    expected_center_residues: frozenset[str],
) -> dict[str, object]:
    """Describe one dbPTM window task and its directly observable confounds."""

    rows = load_dbptm_windows(
        task=task,
        positive_path=positive_path,
        negative_path=negative_path,
        expected_center_residues=expected_center_residues,
    )
    positives = [row for row in rows if row.label]
    negatives = [row for row in rows if not row.label]
    positive_windows = {row.sequence_sha256 for row in positives}
    negative_windows = {row.sequence_sha256 for row in negatives}
    centers = {
        "positive": dict(
            sorted(Counter(row.center_residue for row in positives).items())
        ),
        "negative": dict(
            sorted(Counter(row.center_residue for row in negatives).items())
        ),
    }
    off_target = {
        "positive": sum(
            row.center_residue not in expected_center_residues for row in positives
        ),
        "negative": sum(
            row.center_residue not in expected_center_residues for row in negatives
        ),
    }
    assignments = record_group_split(rows)
    split_rows = {
        split: [
            row
            for row, assigned in zip(rows, assignments, strict=True)
            if assigned == split
        ]
        for split in ("train", "validation", "test")
    }
    return {
        "task": task,
        "window_length": WINDOW_LENGTH,
        "expected_center_residues": "".join(sorted(expected_center_residues)),
        "paths": {
            "positive": str(positive_path.resolve()),
            "negative": str(negative_path.resolve()),
        },
        "sha256": {
            "positive": file_sha256(positive_path),
            "negative": file_sha256(negative_path),
        },
        "rows": len(rows),
        "positive": len(positives),
        "negative": len(negatives),
        "unique_windows": len({row.sequence_sha256 for row in rows}),
        "duplicate_window_rows": len(rows) - len({row.sequence_sha256 for row in rows}),
        "exact_window_label_conflicts": len(positive_windows & negative_windows),
        "unique_record_ids": len({row.record_id for row in rows}),
        "center_residue_counts": centers,
        "off_target_center": off_target,
        "record_group_split": {
            "seed": 1337,
            "method": "FASTA record-id hash 70/15/15",
            "remote_homology_disjoint": False,
            "splits": {
                split: {
                    "rows": len(values),
                    "positive": sum(row.label for row in values),
                    "negative": len(values) - sum(row.label for row in values),
                    "record_ids": len({row.record_id for row in values}),
                }
                for split, values in split_rows.items()
            },
        },
        "evaluation_policy": (
            "These are 21-residue site windows, not protein-disjoint full sequences. "
            "Retain off-target centres in the source audit; a task-specific candidate "
            "residue score must state its filtering policy. Search source proteins against "
            "the pretraining corpus before calling the benchmark corpus-clean."
        ),
    }


def audit_dbptm_benchmark(root: Path) -> dict[str, object]:
    """Audit the downloaded dbPTM 2018 benchmark archive extraction."""

    tasks: dict[str, object] = {}
    for task, spec in DBPTM_TASKS.items():
        directory = root / spec.directory
        tasks[task] = audit_dbptm_task(
            task=task,
            positive_path=directory / spec.positive_filename,
            negative_path=directory / spec.negative_filename,
            expected_center_residues=spec.expected_center_residues,
        )
    return {
        "schema_version": 1,
        "asset": "dbPTM published benchmark windows",
        "source": "dbPTM download benchmark archive (2018 file timestamps)",
        "root": str(root.resolve()),
        "tasks": tasks,
        "evaluation_policy": (
            "Use as external 21-residue PTM-site context diagnostics only. Do not mix "
            "these windows with full-protein site metrics or claim protein-disjoint or "
            "pretraining-corpus-clean evaluation until separate sequence identity audits run."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_dbptm_benchmark(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
