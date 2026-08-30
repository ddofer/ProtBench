"""Structured PTM-site inputs and vocabulary audits.

The PTM-Mamba ``ptm_labels.csv`` is pretraining data, not a held-out benchmark.
This module therefore only validates and canonicalizes it. It never assigns a
train/test split and its report labels the asset as unsuitable for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import runpy
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

PTM_FAMILIES = (
    "phospho_ST",
    "phospho_Y",
    "acetyl_K",
    "methyl_KR",
    "ubiquitin_K",
    "sumo_K",
    "o_glyc_ST",
    "n_glyc_N",
    "s_nitrosyl_C",
    "palmitoyl_myristoyl",
    "rare_k_acyl",
    "other",
)

REQUIRED_COLUMNS = {"AC_ID", "pos", "label", "ori_seq", "token"}


@dataclass(frozen=True)
class PTMSite:
    accession: str
    position: int
    label: str
    sequence: str
    source_token: str

    @property
    def residue(self) -> str:
        return self.sequence[self.position]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text(path: Path):
    """Open by magic bytes, so a misleading ``.gz.gz`` suffix is harmless."""

    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("rt", encoding="utf-8-sig", newline="")


def iter_ptm_sites(path: Path) -> Iterator[PTMSite]:
    """Stream strict zero-based PTM-Mamba CSV records."""

    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            accession = str(row["AC_ID"]).strip()
            label = str(row["label"]).strip()
            sequence = re.sub(r"\s+", "", str(row["ori_seq"])).upper()
            token = str(row["token"]).strip()
            try:
                position = int(row["pos"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid position") from exc
            if not accession or not label or not sequence:
                raise ValueError(f"{path}:{line_number}: empty required value")
            if position < 0 or position >= len(sequence):
                raise ValueError(
                    f"{path}:{line_number}: zero-based position {position} outside "
                    f"sequence length {len(sequence)}"
                )
            yield PTMSite(accession, position, label, sequence, token)


def canonical_ptm_family(label: str, residue: str) -> str:
    """Map a source label to Proteva's broad 12-class PTM vocabulary."""

    norm = re.sub(r"[\s_\-]+", "", label.lower())
    residue = residue.upper()
    if "phospho" in norm:
        if residue in {"S", "T"}:
            return "phospho_ST"
        if residue == "Y":
            return "phospho_Y"
        return "other"
    if "acetyl" in norm:
        return "acetyl_K" if residue == "K" else "other"
    if "methyl" in norm:
        return "methyl_KR" if residue in {"K", "R"} else "other"
    if "ubiquitin" in norm:
        return "ubiquitin_K" if residue == "K" else "other"
    if "sumoyl" in norm or "sumo" in norm:
        return "sumo_K" if residue == "K" else "other"
    if any(term in norm for term in ("nglyc", "nlinked")):
        return "n_glyc_N" if residue == "N" else "other"
    if any(term in norm for term in ("oglyc", "olinked", "galnac", "glcnac")):
        return "o_glyc_ST" if residue in {"S", "T"} else "other"
    if "nitrosyl" in norm:
        return "s_nitrosyl_C" if residue == "C" else "other"
    if "palmitoyl" in norm or "myristoyl" in norm:
        return "palmitoyl_myristoyl"
    if any(term in norm for term in ("succinyl", "malonyl", "glutaryl", "crotonyl")):
        return "rare_k_acyl" if residue == "K" else "other"
    return "other"


def _family_from_native_token(token: str, residue: str) -> str:
    label = token.removeprefix("[PTM_").removesuffix("]").replace("_", " ")
    return canonical_ptm_family(label, residue)


def load_fine_vocab(path: Path) -> dict[str, tuple[str, str]]:
    """Load Proteva's data-only ``FINE_PTM`` mapping without importing the package."""

    namespace = runpy.run_path(str(path))
    raw = namespace.get("FINE_PTM")
    if not isinstance(raw, dict):
        raise TypeError(f"{path} does not define FINE_PTM")
    out = {}
    for label, mapping in raw.items():
        if (
            not isinstance(label, str)
            or not isinstance(mapping, tuple)
            or len(mapping) != 2
        ):
            raise ValueError(f"{path}: malformed FINE_PTM entry {label!r}: {mapping!r}")
        token, residue = mapping
        out[label] = (str(token), str(residue))
    return out


def audit_ptm_labels(
    path: Path,
    *,
    fine_vocab: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Audit source integrity and exact/broad Proteva mapping coverage."""

    fine_vocab = fine_vocab or {}
    labels: Counter[str] = Counter()
    label_residues: dict[str, Counter[str]] = {}
    label_sequences: dict[str, set[str]] = {}
    accessions: set[str] = set()
    sequence_hashes: set[str] = set()
    sequence_lengths: dict[str, int] = {}
    accession_sequences: dict[str, str] = {}
    duplicate_sites = 0
    seen_sites: set[tuple[str, int, str]] = set()
    token_mismatches = 0
    accession_sequence_conflicts = 0
    native_rows = 0
    native_residue_mismatches = 0
    broad_rows = 0
    any_supported_rows = 0

    for site in iter_ptm_sites(path):
        labels[site.label] += 1
        label_residues.setdefault(site.label, Counter())[site.residue] += 1
        accessions.add(site.accession)
        seq_hash = hashlib.sha256(site.sequence.encode()).hexdigest()
        sequence_hashes.add(seq_hash)
        sequence_lengths.setdefault(seq_hash, len(site.sequence))
        label_sequences.setdefault(site.label, set()).add(seq_hash)
        prior_hash = accession_sequences.setdefault(site.accession, seq_hash)
        accession_sequence_conflicts += int(prior_hash != seq_hash)
        key = (site.accession, site.position, site.label)
        duplicate_sites += int(key in seen_sites)
        seen_sites.add(key)
        token_mismatches += int(site.source_token != f"<{site.label}>")
        native = fine_vocab.get(site.label)
        if native is not None:
            native_rows += 1
            native_residue_mismatches += int(site.residue != native[1])
        broad_supported = canonical_ptm_family(site.label, site.residue) != "other"
        broad_rows += int(broad_supported)
        any_supported_rows += int(native is not None or broad_supported)

    label_table = []
    for label, count in labels.most_common():
        residues = label_residues[label]
        dominant_residue, dominant_count = residues.most_common(1)[0]
        native = fine_vocab.get(label)
        family = canonical_ptm_family(label, dominant_residue)
        native_token = native[0] if native else None
        native_mismatch_rows = (
            sum(count for residue, count in residues.items() if residue != native[1])
            if native
            else 0
        )
        if native_token:
            native_family = _family_from_native_token(native_token, dominant_residue)
            if native_family != "other":
                family = native_family
        label_table.append(
            {
                "source_label": label,
                "rows": count,
                "unique_sequences": len(label_sequences[label]),
                "residues": dict(sorted(residues.items())),
                "dominant_residue": dominant_residue,
                "dominant_residue_fraction": dominant_count / count,
                "proteva_native_token": native_token,
                "proteva_family": family,
                "native_supported": native is not None,
                "native_residue_mismatch_rows": native_mismatch_rows,
                "broad_supported": family != "other",
                "any_supported": native is not None or family != "other",
            }
        )

    rows = sum(labels.values())
    unique_sequences = len(sequence_hashes)
    label_probabilities = [count / rows for count in labels.values()] if rows else []
    shannon = -sum(p * math.log(p) for p in label_probabilities)
    sorted_counts = sorted(labels.values(), reverse=True)

    def top_fraction(k: int) -> float:
        return sum(sorted_counts[:k]) / rows if rows else 0.0

    return {
        "schema_version": 2,
        "source": {
            "path": str(path),
            "sha256": file_sha256(path),
            "role": "PTM-Mamba pretraining annotations; not a held-out benchmark",
            "position_indexing": "zero-based",
        },
        "rows": rows,
        "unique_accessions": len(accessions),
        "unique_sequences": unique_sequences,
        "unique_sequence_residues": sum(sequence_lengths.values()),
        "mean_unique_sequence_length": (
            sum(sequence_lengths.values()) / unique_sequences
            if unique_sequences
            else 0.0
        ),
        "records_per_unique_sequence": rows / unique_sequences
        if unique_sequences
        else 0.0,
        "source_label_types": len(labels),
        "label_diversity": {
            "shannon_nats": shannon,
            "shannon_effective_types": math.exp(shannon),
            "simpson_effective_types": (
                1.0 / sum(p * p for p in label_probabilities)
                if label_probabilities
                else 0.0
            ),
            "top_2_row_fraction": top_fraction(2),
            "top_3_row_fraction": top_fraction(3),
            "top_5_row_fraction": top_fraction(5),
        },
        "native_supported_rows": native_rows,
        "native_supported_row_fraction": native_rows / rows if rows else 0.0,
        "native_supported_types": sum(
            bool(item["native_supported"]) for item in label_table
        ),
        "broad_supported_rows": broad_rows,
        "broad_supported_row_fraction": broad_rows / rows if rows else 0.0,
        "broad_supported_types": sum(
            bool(item["broad_supported"]) for item in label_table
        ),
        "any_supported_rows": any_supported_rows,
        "any_supported_row_fraction": any_supported_rows / rows if rows else 0.0,
        "any_supported_types": sum(bool(item["any_supported"]) for item in label_table),
        "duplicate_site_rows": duplicate_sites,
        "source_token_mismatches": token_mismatches,
        "accession_sequence_conflicts": accession_sequence_conflicts,
        "native_residue_mismatches": native_residue_mismatches,
        "labels": label_table,
        "evaluation_guard": (
            "No split is present. Do not score this asset as an independent PTM test set."
        ),
    }


def write_audit(report: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ptm_label_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    labels = report["labels"]
    if not isinstance(labels, list):
        raise TypeError("report labels must be a list")
    with (out_dir / "ptm_label_mapping.csv").open("w", newline="") as handle:
        fieldnames = (
            "source_label",
            "rows",
            "unique_sequences",
            "residues",
            "dominant_residue",
            "dominant_residue_fraction",
            "proteva_native_token",
            "proteva_family",
            "native_supported",
            "native_residue_mismatch_rows",
            "broad_supported",
            "any_supported",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in labels:
            serialized = dict(row)
            serialized["residues"] = json.dumps(serialized["residues"], sort_keys=True)
            writer.writerow(serialized)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fine-vocab-py", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    vocab = load_fine_vocab(args.fine_vocab_py) if args.fine_vocab_py else {}
    report = audit_ptm_labels(args.input, fine_vocab=vocab)
    write_audit(report, args.out)
    printable = {key: value for key, value in report.items() if key != "labels"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
