"""Audit and normalize external PTM residue-site benchmark assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ptm_inputs import file_sha256

SPLITS = ("train", "valid", "test")
NHAC_SPLITS = ("train", "val", "test")
NHAC_WINDOWS = (11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61)
NHAC_PAPER_SPLIT_COUNTS = {
    "train": {"rows": 4024, "positive": 637, "negative": 3387},
    "val": {"rows": 477, "positive": 76, "negative": 401},
    "test": {"rows": 1092, "positive": 173, "negative": 919},
}


@dataclass(frozen=True)
class PTMSiteBenchmarkRow:
    split: str
    row_index: int
    sequence: str
    labels: str

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode()).hexdigest()


def load_proteinbert_phosphosite(
    root: Path,
    *,
    min_length: int = 256,
    max_length: int = 512,
) -> tuple[dict[str, list[PTMSiteBenchmarkRow]], dict[str, int]]:
    """Load the exact length-filtered ProteinBERT PhosphositePTM splits."""

    if min_length < 1 or max_length < min_length:
        raise ValueError("invalid length filter")
    splits: dict[str, list[PTMSiteBenchmarkRow]] = {}
    raw_counts: dict[str, int] = {}
    for split in SPLITS:
        path = root / f"PhosphositePTM.{split}.csv"
        rows: list[PTMSiteBenchmarkRow] = []
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"seq", "label"} <= set(reader.fieldnames or ()):
                raise ValueError(f"{path}: expected seq,label columns")
            raw_count = 0
            for row_index, raw in enumerate(reader):
                raw_count += 1
                sequence = "".join(str(raw["seq"]).split()).upper()
                labels = "".join(str(raw["label"]).split())
                if len(sequence) < min_length or len(sequence) > max_length:
                    continue
                if len(labels) != len(sequence):
                    raise ValueError(
                        f"{path}:{row_index + 2}: sequence/label lengths "
                        f"{len(sequence)}/{len(labels)} differ"
                    )
                if set(labels) - {"0", "1"}:
                    raise ValueError(f"{path}:{row_index + 2}: labels are not binary")
                rows.append(PTMSiteBenchmarkRow(split, row_index, sequence, labels))
        splits[split] = rows
        raw_counts[split] = raw_count
    return splits, raw_counts


def _label_counts(rows: list[PTMSiteBenchmarkRow]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row.sequence_sha256][row.labels] += 1
    return counts


def _pair_overlap(
    left: list[PTMSiteBenchmarkRow], right: list[PTMSiteBenchmarkRow]
) -> dict[str, int]:
    left_counts = _label_counts(left)
    right_counts = _label_counts(right)
    shared = set(left_counts) & set(right_counts)
    pairs = 0
    conflicting_pairs = 0
    for seq_hash in shared:
        for left_label, left_n in left_counts[seq_hash].items():
            for right_label, right_n in right_counts[seq_hash].items():
                n_pairs = left_n * right_n
                pairs += n_pairs
                conflicting_pairs += n_pairs * int(left_label != right_label)
    return {
        "exact_sequence_hashes": len(shared),
        "row_pairs": pairs,
        "conflicting_label_row_pairs": conflicting_pairs,
    }


def audit_proteinbert_phosphosite(
    root: Path,
    *,
    min_length: int = 256,
    max_length: int = 512,
) -> tuple[dict[str, object], list[PTMSiteBenchmarkRow]]:
    """Audit published splits and return the exact-deduplicated test rows."""

    splits, raw_counts = load_proteinbert_phosphosite(
        root, min_length=min_length, max_length=max_length
    )
    per_split: dict[str, object] = {}
    for split, rows in splits.items():
        path = root / f"PhosphositePTM.{split}.csv"
        unique_hashes = {row.sequence_sha256 for row in rows}
        per_split[split] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "raw_rows": raw_counts[split],
            "filtered_rows": len(rows),
            "unique_sequence_hashes": len(unique_hashes),
            "duplicate_sequence_rows": len(rows) - len(unique_hashes),
            "positive_sites": sum(row.labels.count("1") for row in rows),
        }

    train_hashes = {row.sequence_sha256 for row in splits["train"]}
    valid_hashes = {row.sequence_sha256 for row in splits["valid"]}
    prior_hashes = train_hashes | valid_hashes
    excluded_test = [row for row in splits["test"] if row.sequence_sha256 in prior_hashes]
    clean_test = [row for row in splits["test"] if row.sequence_sha256 not in prior_hashes]
    report: dict[str, object] = {
        "schema_version": 1,
        "asset": "ProteinBERT PhosphositePTM",
        "source_revision": "nadavbra/proteinbert_data_files@4636c3ddfe11e3e553bbc384753a269c84d331cf",
        "length_filter": {"min_inclusive": min_length, "max_inclusive": max_length},
        "splits": per_split,
        "exact_overlap": {
            "train_valid": _pair_overlap(splits["train"], splits["valid"]),
            "train_test": _pair_overlap(splits["train"], splits["test"]),
            "valid_test": _pair_overlap(splits["valid"], splits["test"]),
        },
        "deduplicated_test": {
            "rows": len(clean_test),
            "unique_sequence_hashes": len({row.sequence_sha256 for row in clean_test}),
            "excluded_rows": len(excluded_test),
            "excluded_sequence_hashes": len(
                {row.sequence_sha256 for row in excluded_test}
            ),
            "excluded_source_row_indices": [row.row_index for row in excluded_test],
        },
        "evaluation_policy": (
            "Retain the full author test split for paper comparability. Use the test "
            "view excluding every exact train/validation sequence as the primary "
            "diagnostic. Conflicting label tracks on exact sequences make the author "
            "overlap both leakage and annotation inconsistency, not harmless duplication."
        ),
    }
    return report, clean_test


def write_clean_test(rows: list[PTMSiteBenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_row_index", "seq", "label"))
        writer.writeheader()
        writer.writerows(
            {
                "source_row_index": row.row_index,
                "seq": row.sequence,
                "label": row.labels,
            }
            for row in rows
        )


def audit_nhac(path: Path, *, deduplicated_path: Path | None = None) -> dict[str, object]:
    """Audit the TransPTM non-histone acetylation collection and split leakage."""

    required = {"unique_id", "label", "set", *(f"seq_{size}" for size in NHAC_WINDOWS)}
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            row = {key: str(value).strip() for key, value in raw.items()}
            if row["set"] not in NHAC_SPLITS:
                raise ValueError(f"{path}:{line_number}: invalid split {row['set']!r}")
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"{path}:{line_number}: invalid label {row['label']!r}")
            for size in NHAC_WINDOWS:
                sequence = row[f"seq_{size}"].upper()
                if len(sequence) != size:
                    raise ValueError(
                        f"{path}:{line_number}: seq_{size} has length {len(sequence)}"
                    )
                if sequence[size // 2] != "K":
                    raise ValueError(
                        f"{path}:{line_number}: seq_{size} is not centered on lysine"
                    )
                row[f"seq_{size}"] = sequence
            rows.append(row)

    split_rows = {
        split: [row for row in rows if row["set"] == split] for split in NHAC_SPLITS
    }
    per_split = {
        split: {
            "rows": len(values),
            "positive": sum(row["label"] == "1" for row in values),
            "negative": sum(row["label"] == "0" for row in values),
            "unique_ids": len({row["unique_id"] for row in values}),
            "unique_seq_61": len({row["seq_61"] for row in values}),
        }
        for split, values in split_rows.items()
    }

    def overlap(field: str, left: str, right: str) -> int:
        return len(
            {row[field] for row in split_rows[left]}
            & {row[field] for row in split_rows[right]}
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "asset": "TransPTM NHAC",
        "source_revision": "TransPTM/TransPTM@edb80f19990b190e8792e8314a6cc695b078b928",
        "path": str(path),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "positive": sum(row["label"] == "1" for row in rows),
        "negative": sum(row["label"] == "0" for row in rows),
        "duplicate_unique_id_rows": len(rows) - len({row["unique_id"] for row in rows}),
        "splits": per_split,
        "paper_reported_splits": NHAC_PAPER_SPLIT_COUNTS,
        "paper_split_counts_match": all(
            per_split[split][metric] == expected
            for split, expected_counts in NHAC_PAPER_SPLIT_COUNTS.items()
            for metric, expected in expected_counts.items()
        ),
        "exact_split_overlap": {
            f"{left}_{right}": {
                "unique_id": overlap("unique_id", left, right),
                "seq_11": overlap("seq_11", left, right),
                "seq_61": overlap("seq_61", left, right),
            }
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        },
        "evaluation_policy": (
            "Retain the author split for PTM-Mamba comparability. For the primary "
            "diagnostic, group the source rows by exact longest window (and preferably "
            "by accession parsed from unique_id) before assigning train/validation/test. "
            "Do not silently substitute NHAC_deduplicated.csv: it removes positive "
            "records and changes the published class balance. The current repository "
            "split also does not reproduce the paper's per-split class counts, so exact "
            "PTM-Mamba comparability requires the authors' frozen benchmark files."
        ),
    }
    if deduplicated_path is not None:
        with deduplicated_path.open(newline="") as handle:
            dedup_rows = list(csv.DictReader(handle))
        report["repository_deduplicated_variant"] = {
            "path": str(deduplicated_path),
            "sha256": file_sha256(deduplicated_path),
            "rows": len(dedup_rows),
            "positive": sum(str(row["label"]) == "1" for row in dedup_rows),
            "negative": sum(str(row["label"]) == "0" for row in dedup_rows),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--proteinbert-phosphosite-dir", type=Path)
    source.add_argument("--nhac", type=Path)
    parser.add_argument("--nhac-deduplicated", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean-test-out", type=Path)
    parser.add_argument("--min-length", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args(argv)
    clean_test: list[PTMSiteBenchmarkRow] = []
    if args.proteinbert_phosphosite_dir:
        report, clean_test = audit_proteinbert_phosphosite(
            args.proteinbert_phosphosite_dir,
            min_length=args.min_length,
            max_length=args.max_length,
        )
    else:
        report = audit_nhac(args.nhac, deduplicated_path=args.nhac_deduplicated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.clean_test_out and args.proteinbert_phosphosite_dir:
        write_clean_test(clean_test, args.clean_test_out)
    elif args.clean_test_out:
        parser.error("--clean-test-out applies only to ProteinBERT PhosphositePTM")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
