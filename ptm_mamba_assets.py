"""Load and audit the public PTM-Mamba known-site type dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ptm_benchmark import PTMTypePrediction, score_ptm_types

AUTHOR_FILES = {
    "train": "split_data_train.csv",
    "validation": "split_data_val.csv",
    "test": "split_data_test.csv",
}
REQUIRED_COLUMNS = {"AC_ID", "pos", "label", "ori_seq"}


@dataclass(frozen=True)
class PTMMambaSite:
    author_split: str
    source_row: int
    accession: str
    position: int
    target_type: str
    sequence: str

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode()).hexdigest()

    @property
    def residue(self) -> str:
        return self.sequence[self.position]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ptm_mamba(root: Path) -> list[PTMMambaSite]:
    """Read available author CSVs and validate their documented zero-based sites."""
    rows: list[PTMMambaSite] = []
    found = 0
    for split, filename in AUTHOR_FILES.items():
        path = root / filename
        if not path.is_file():
            continue
        found += 1
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for source_row, raw in enumerate(reader):
                accession = str(raw["AC_ID"]).strip()
                target_type = str(raw["label"]).strip()
                sequence = "".join(str(raw["ori_seq"]).split()).upper()
                if not accession or not target_type or not sequence:
                    raise ValueError(f"{path}:{source_row + 2}: empty required value")
                try:
                    position = int(str(raw["pos"]).strip())
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{source_row + 2}: invalid position {raw['pos']!r}"
                    ) from error
                if position < 0 or position >= len(sequence):
                    raise ValueError(
                        f"{path}:{source_row + 2}: position {position} outside "
                        f"sequence of length {len(sequence)}"
                    )
                rows.append(
                    PTMMambaSite(
                        author_split=split,
                        source_row=source_row,
                        accession=accession,
                        position=position,
                        target_type=target_type,
                        sequence=sequence,
                    )
                )
    if not found:
        raise FileNotFoundError(f"no PTM-Mamba split_data_*.csv files under {root}")
    return rows


def protein_disjoint_split(
    rows: Sequence[PTMMambaSite], *, seed: int = 1337
) -> list[str]:
    """Assign accession/exact-sequence connected components to 70/15/15 splits."""
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_accession: dict[str, int] = {}
    by_sequence: dict[str, int] = {}
    for index, row in enumerate(rows):
        for key, mapping in (
            (row.accession, by_accession),
            (row.sequence_sha256, by_sequence),
        ):
            previous = mapping.setdefault(key, index)
            union(index, previous)

    members: dict[int, list[int]] = {}
    for index in range(len(rows)):
        members.setdefault(find(index), []).append(index)
    assignments: dict[int, str] = {}
    for root, indices in members.items():
        identity = min(
            f"{rows[index].accession}\0{rows[index].sequence_sha256}"
            for index in indices
        )
        bucket = (
            int.from_bytes(
                hashlib.sha256(f"{seed}\0{identity}".encode()).digest()[:8], "big"
            )
            % 100
        )
        assignments[root] = (
            "train" if bucket < 70 else "validation" if bucket < 85 else "test"
        )
    return [assignments[find(index)] for index in range(len(rows))]


def score_residue_type_baseline(
    rows: Sequence[PTMMambaSite], assignments: Sequence[str]
) -> dict[str, object]:
    """Score P(PTM type | unmodified residue) on known positive test sites."""
    if len(rows) != len(assignments):
        raise ValueError("rows and assignments must have equal length")
    train_counts: dict[str, Counter[str]] = {}
    global_counts: Counter[str] = Counter()
    for row, split in zip(rows, assignments, strict=True):
        if split == "train":
            train_counts.setdefault(row.residue, Counter())[row.target_type] += 1
            global_counts[row.target_type] += 1
    if not global_counts:
        raise ValueError("residue baseline requires training rows")
    all_types = set(global_counts)
    rankings = {
        residue: tuple(
            sorted(
                all_types,
                key=lambda target: (
                    -counts[target],
                    -global_counts[target],
                    target,
                ),
            )
        )
        for residue, counts in train_counts.items()
    }
    global_ranking = tuple(
        sorted(all_types, key=lambda target: (-global_counts[target], target))
    )
    predictions = [
        PTMTypePrediction(
            row_id=f"ptm-mamba:{row.author_split}:{row.source_row}:{row.accession}",
            position=row.position,
            target_type=row.target_type,
            ranked_types=rankings.get(row.residue, global_ranking),
        )
        for row, split in zip(rows, assignments, strict=True)
        if split == "test"
    ]
    return score_ptm_types(predictions)


def _overlap(
    left: Sequence[PTMMambaSite], right: Sequence[PTMMambaSite]
) -> dict[str, int]:
    return {
        "accessions": len(
            {row.accession for row in left} & {row.accession for row in right}
        ),
        "exact_sequences": len(
            {row.sequence_sha256 for row in left}
            & {row.sequence_sha256 for row in right}
        ),
        "accession_positions": len(
            {(row.accession, row.position) for row in left}
            & {(row.accession, row.position) for row in right}
        ),
    }


def audit_ptm_mamba(root: Path, *, seed: int = 1337) -> dict[str, object]:
    rows = load_ptm_mamba(root)
    author = {
        split: [row for row in rows if row.author_split == split]
        for split in AUTHOR_FILES
    }
    reassigned = protein_disjoint_split(rows, seed=seed)
    author_assignments = [row.author_split for row in rows]
    return {
        "schema_version": 1,
        "asset": "PTM-Mamba known-site PTM type dataset",
        "paths": {
            split: {
                "path": str((root / filename).resolve()),
                "sha256": _file_sha256(root / filename),
            }
            for split, filename in AUTHOR_FILES.items()
            if (root / filename).is_file()
        },
        "rows": len(rows),
        "accessions": len({row.accession for row in rows}),
        "exact_sequences": len({row.sequence_sha256 for row in rows}),
        "types": dict(sorted(Counter(row.target_type for row in rows).items())),
        "author_splits": {
            split: {
                "rows": len(values),
                "accessions": len({row.accession for row in values}),
                "exact_sequences": len({row.sequence_sha256 for row in values}),
            }
            for split, values in author.items()
        },
        "overlap": {
            f"{left}_{right}": _overlap(author[left], author[right])
            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        },
        "protein_disjoint_split": {
            "seed": seed,
            "method": "accession/exact-sequence connected components; hash 70/15/15",
            "remote_homology_disjoint": False,
            "rows": dict(sorted(Counter(reassigned).items())),
            "types_by_split": {
                split: dict(
                    sorted(
                        Counter(
                            row.target_type
                            for row, assigned in zip(rows, reassigned, strict=True)
                            if assigned == split
                        ).items()
                    )
                )
                for split in ("train", "validation", "test")
            },
        },
        "residue_identity_type_baseline": {
            "author_split": score_residue_type_baseline(rows, author_assignments),
            "protein_disjoint_split": score_residue_type_baseline(rows, reassigned),
        },
        "evaluation_policy": (
            "Use the author split only for paper comparability: it is site-disjoint "
            "but not protein-disjoint. Use the connected-component split as the "
            "primary exact-protein-disjoint known-site type task. Audit sequence "
            "similarity separately before calling it remote-homology generalization."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_ptm_mamba(args.root, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rows": report["rows"],
                "accessions": report["accessions"],
                "exact_sequences": report["exact_sequences"],
                "overlap": report["overlap"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
