"""Load and audit the PTMint PTM-effect PPI benchmark.

The Hugging Face split is useful for reproducing the PTM-Mamba PPI task, but it
is not protein-disjoint. This module preserves that split for comparability and
reports the exact target/pair overlap needed before defining a primary split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ptm_inputs import file_sha256

PTMINT_DATASET = "RosettaCommons/PTMint"
PTMINT_REVISION = "7be87d54e436852b6976fc0f3180687422447c55"
PTMINT_FILENAME = "ptmint_all_with_splits.csv"
PTMINT_SPLIT_COUNTS = {"train": 3980, "validation": 597, "test": 579}
PTMINT_EFFECT_COUNTS = {"Enhance": 3391, "Inhibit": 1765}
SPLITS = tuple(PTMINT_SPLIT_COUNTS)
EFFECTS = tuple(PTMINT_EFFECT_COUNTS)
REQUIRED_COLUMNS = {
    "Uniprot",
    "PTM",
    "Site",
    "AA",
    "Int_uniprot",
    "Effect",
    "Gene_sequence",
    "Int_gene_sequence",
    "cluster_ID",
    "split",
}


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


@dataclass(frozen=True)
class PTMintRow:
    """One PTM effect on a target-partner interaction."""

    split: str
    row_index: int
    target_accession: str
    partner_accession: str
    ptm_type: str
    site: int
    site_residue: str
    target_sequence: str
    partner_sequence: str
    effect: str
    cluster_id: str

    @property
    def target_sha256(self) -> str:
        return _sequence_hash(self.target_sequence)

    @property
    def partner_sha256(self) -> str:
        return _sequence_hash(self.partner_sequence)

    @property
    def pair_sha256(self) -> str:
        return _sequence_hash(f"{self.target_sequence}\0{self.partner_sequence}")

    @property
    def coordinate_valid(self) -> bool:
        return 1 <= self.site <= len(self.target_sequence)

    @property
    def residue_matches(self) -> bool:
        return (
            self.coordinate_valid
            and self.target_sequence[self.site - 1] == self.site_residue
        )

    @property
    def event_key(self) -> tuple[str, int, str, str, str]:
        return (
            self.target_accession,
            self.site,
            self.ptm_type,
            self.partner_accession,
            self.effect,
        )


def load_ptmint(
    path: Path,
    *,
    enforce_expected_counts: bool = True,
) -> list[PTMintRow]:
    """Load the pinned PTMint CSV while preserving source row indices."""

    rows: list[PTMintRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row_index, raw in enumerate(reader):
            split = str(raw["split"]).strip()
            effect = str(raw["Effect"]).strip()
            if split not in SPLITS:
                raise ValueError(f"{path}:{row_index + 2}: invalid split {split!r}")
            if effect not in EFFECTS:
                raise ValueError(f"{path}:{row_index + 2}: invalid effect {effect!r}")
            try:
                site = int(str(raw["Site"]).strip())
            except ValueError as exc:
                raise ValueError(f"{path}:{row_index + 2}: invalid site") from exc
            target_sequence = "".join(str(raw["Gene_sequence"]).split()).upper()
            partner_sequence = "".join(str(raw["Int_gene_sequence"]).split()).upper()
            required_values = {
                "Uniprot": str(raw["Uniprot"]).strip(),
                "PTM": str(raw["PTM"]).strip(),
                "AA": str(raw["AA"]).strip().upper(),
                "Int_uniprot": str(raw["Int_uniprot"]).strip(),
                "Gene_sequence": target_sequence,
                "Int_gene_sequence": partner_sequence,
            }
            empty = sorted(key for key, value in required_values.items() if not value)
            if empty:
                raise ValueError(f"{path}:{row_index + 2}: empty values for {empty}")
            if len(required_values["AA"]) != 1:
                raise ValueError(f"{path}:{row_index + 2}: AA must be one residue")
            rows.append(
                PTMintRow(
                    split=split,
                    row_index=row_index,
                    target_accession=required_values["Uniprot"],
                    partner_accession=required_values["Int_uniprot"],
                    ptm_type=required_values["PTM"],
                    site=site,
                    site_residue=required_values["AA"],
                    target_sequence=target_sequence,
                    partner_sequence=partner_sequence,
                    effect=effect,
                    cluster_id=str(raw["cluster_ID"]).strip(),
                )
            )

    if enforce_expected_counts:
        split_counts = Counter(row.split for row in rows)
        effect_counts = Counter(row.effect for row in rows)
        if dict(split_counts) != PTMINT_SPLIT_COUNTS:
            raise ValueError(
                f"PTMint split count drift: expected {PTMINT_SPLIT_COUNTS}, "
                f"observed {dict(split_counts)}"
            )
        if dict(effect_counts) != PTMINT_EFFECT_COUNTS:
            raise ValueError(
                f"PTMint effect count drift: expected {PTMINT_EFFECT_COUNTS}, "
                f"observed {dict(effect_counts)}"
            )
    return rows


def _split_overlap(
    rows: list[PTMintRow],
    field: str,
    left: str,
    right: str,
) -> int:
    left_values = {getattr(row, field) for row in rows if row.split == left}
    right_values = {getattr(row, field) for row in rows if row.split == right}
    return len(left_values & right_values)


def _cross_split_count(rows: list[PTMintRow], field: str) -> int:
    splits_by_value: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_value[str(getattr(row, field))].add(row.split)
    return sum(len(splits) > 1 for splits in splits_by_value.values())


def audit_ptmint(
    path: Path,
    *,
    enforce_expected_counts: bool = True,
) -> tuple[dict[str, object], list[PTMintRow]]:
    """Report split leakage and return valid, event-deduplicated rows."""

    rows = load_ptmint(path, enforce_expected_counts=enforce_expected_counts)
    invalid_coordinates = [row for row in rows if not row.coordinate_valid]
    residue_mismatches = [
        row for row in rows if row.coordinate_valid and not row.residue_matches
    ]

    event_counts = Counter(row.event_key for row in rows)
    seen_events: set[tuple[str, int, str, str, str]] = set()
    primary_candidates: list[PTMintRow] = []
    for row in rows:
        if not row.coordinate_valid or not row.residue_matches:
            continue
        if row.event_key in seen_events:
            continue
        seen_events.add(row.event_key)
        primary_candidates.append(row)

    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.cluster_id:
            cluster_splits[row.cluster_id].add(row.split)
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    per_split = {
        split: {
            "rows": sum(row.split == split for row in rows),
            "enhance": sum(
                row.split == split and row.effect == "Enhance" for row in rows
            ),
            "inhibit": sum(
                row.split == split and row.effect == "Inhibit" for row in rows
            ),
            "unique_targets": len(
                {row.target_sha256 for row in rows if row.split == split}
            ),
            "unique_pairs": len(
                {row.pair_sha256 for row in rows if row.split == split}
            ),
        }
        for split in SPLITS
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "asset": "PTMint PTM-effect PPI",
        "source": {
            "dataset": PTMINT_DATASET,
            "revision": PTMINT_REVISION,
            "filename": PTMINT_FILENAME,
            "path": str(path),
            "sha256": file_sha256(path),
        },
        "rows": len(rows),
        "effects": dict(sorted(Counter(row.effect for row in rows).items())),
        "splits": per_split,
        "data_quality": {
            "invalid_coordinate_rows": [row.row_index for row in invalid_coordinates],
            "residue_mismatch_rows": [row.row_index for row in residue_mismatches],
            "duplicate_event_rows": sum(count - 1 for count in event_counts.values()),
            "valid_event_deduplicated_rows": len(primary_candidates),
        },
        "exact_values_present_multiple_splits": {
            "target_sequences": _cross_split_count(rows, "target_sha256"),
            "ordered_sequence_pairs": _cross_split_count(rows, "pair_sha256"),
        },
        "provided_split_overlap": {
            f"{left}_{right}": {
                "exact_target_sequences": _split_overlap(
                    rows, "target_sha256", left, right
                ),
                "exact_ordered_sequence_pairs": _split_overlap(
                    rows, "pair_sha256", left, right
                ),
            }
            for left, right in split_pairs
        },
        "clusters_present_in_multiple_splits": sum(
            len(splits) > 1 for splits in cluster_splits.values()
        ),
        "evaluation_policy": (
            "Use the provided split only as a labeled PTM-Mamba/PTMint reproduction. "
            "It is cluster-separated but not exact-target- or exact-pair-disjoint. "
            "Before primary evaluation, remove invalid and duplicate events, then freeze "
            "a target-sequence-grouped split and report remaining partner overlap. "
            "This audit does not test overlap with the Proteva pretraining corpus."
        ),
    }
    return report, primary_candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report, _ = audit_ptmint(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
