"""Per-query persistence and corpus-identity stratification for CATH-EAT.

This module is deliberately model-agnostic. Embedding and nearest-neighbour work
stays in :mod:`cath_levels`; the records written here can be rescored against an
updated identity audit without loading a backbone or recomputing embeddings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CATH_LEVELS = ("cath_c", "cath_a", "cath_t", "cath_h")
STRATA = ("full", "exact", "non_exact", "ge_90", "lt_90")
QUERY_HASH_RE = re.compile(r"^cath_[0-9a-f]{16}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_IDENTITY_COUNTS = {
    "test300": {"n": 300, "ge_90": 283, "exact": 190},
    "test219": {"n": 219, "ge_90": 210, "exact": 150},
    "test_h": {"n": 150, "ge_90": 145, "exact": 102},
}
EXPECTED_H_STRATA = {
    "full": 150,
    "exact": 102,
    "non_exact": 48,
    "ge_90": 145,
    "lt_90": 5,
}


@dataclass(frozen=True)
class CathIdentity:
    """Corpus-identity metadata for one pinned CATH query."""

    query_hash: str
    domain_id: str
    sequence_sha256: str
    in_test300: bool
    in_test219: bool
    in_test_h: bool
    max_identity: float
    ge_90: bool
    exact: bool


def sequence_sha256(sequence: str) -> str:
    """Hash a normalized amino-acid sequence with SHA-256."""

    normalized = "".join(sequence.split()).upper()
    return hashlib.sha256(normalized.encode()).hexdigest()


def cath_query_hash(sequence: str) -> str:
    """Return the query identifier used by the Proteva CATH corpus audit."""

    return f"cath_{sequence_sha256(sequence)[:16]}"


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: str, field: str, line_number: int) -> bool:
    """Parse a deterministic TSV boolean encoded as zero or one."""

    if value not in {"0", "1"}:
        raise ValueError(f"identity table line {line_number}: {field} must be 0 or 1")
    return value == "1"


def _validate_identity_counts(records: Mapping[str, CathIdentity]) -> None:
    """Pin the current CATH audit denominators and identity strata."""

    selectors = {
        "test300": lambda row: row.in_test300,
        "test219": lambda row: row.in_test219,
        "test_h": lambda row: row.in_test_h,
    }
    for split, select in selectors.items():
        rows = [row for row in records.values() if select(row)]
        observed = {
            "n": len(rows),
            "ge_90": sum(row.ge_90 for row in rows),
            "exact": sum(row.exact for row in rows),
        }
        if observed != EXPECTED_IDENTITY_COUNTS[split]:
            raise ValueError(
                f"identity table count drift for {split}: "
                f"expected {EXPECTED_IDENTITY_COUNTS[split]}, observed {observed}"
            )


def load_identity_table(
    path: Path,
    *,
    enforce_expected_counts: bool = True,
) -> dict[str, CathIdentity]:
    """Load and validate Proteva's per-query CATH identity table.

    The returned mapping contains all 300 searched queries. Consumers select
    ``in_test219`` or ``in_test_h`` explicitly, preventing a silent 300/219/150
    split substitution.
    """

    required = {
        "query_hash",
        "domain_id",
        "sequence_sha256",
        "in_test300",
        "in_test219",
        "in_test_h",
        "max_identity",
        "ge_90",
        "exact",
    }
    records: dict[str, CathIdentity] = {}
    domain_ids: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"identity table missing columns: {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            query_hash = raw["query_hash"]
            domain_id = raw["domain_id"]
            sequence_hash = raw["sequence_sha256"]
            if QUERY_HASH_RE.fullmatch(query_hash) is None:
                raise ValueError(
                    f"identity table line {line_number}: invalid query hash"
                )
            if SHA256_RE.fullmatch(sequence_hash) is None:
                raise ValueError(
                    f"identity table line {line_number}: invalid sequence SHA-256"
                )
            if query_hash in records or domain_id in domain_ids:
                raise ValueError(
                    f"identity table line {line_number}: duplicate query/domain"
                )
            max_identity = float(raw["max_identity"])
            if not 0.0 <= max_identity <= 1.0:
                raise ValueError(
                    f"identity table line {line_number}: identity outside [0, 1]"
                )
            row = CathIdentity(
                query_hash=query_hash,
                domain_id=domain_id,
                sequence_sha256=sequence_hash,
                in_test300=_parse_bool(raw["in_test300"], "in_test300", line_number),
                in_test219=_parse_bool(raw["in_test219"], "in_test219", line_number),
                in_test_h=_parse_bool(raw["in_test_h"], "in_test_h", line_number),
                max_identity=max_identity,
                ge_90=_parse_bool(raw["ge_90"], "ge_90", line_number),
                exact=_parse_bool(raw["exact"], "exact", line_number),
            )
            if row.ge_90 != (row.max_identity >= 0.90):
                raise ValueError(
                    f"identity table line {line_number}: inconsistent ge_90 flag"
                )
            if row.exact != math.isclose(row.max_identity, 1.0, abs_tol=1e-12):
                raise ValueError(
                    f"identity table line {line_number}: inconsistent exact flag"
                )
            if row.in_test_h and not row.in_test219:
                raise ValueError(
                    f"identity table line {line_number}: test_h is not in test219"
                )
            records[query_hash] = row
            domain_ids.add(domain_id)
    if enforce_expected_counts:
        _validate_identity_counts(records)
    return records


def answerable_mask(truth: Sequence[str], lookup: Sequence[str]) -> np.ndarray:
    """Return EAT's per-level singleton mask over lookup union query labels."""

    counts = Counter(lookup) + Counter(truth)
    return np.asarray([counts[label] > 1 for label in truth], dtype=bool)


def bootstrap_ci(correct: np.ndarray, n_boot: int = 1000, seed: int = 42) -> float:
    """Return the 95% bootstrap confidence-interval half-width for accuracy."""

    values = np.asarray(correct, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    return float(1.96 * values[indices].mean(axis=1).std(ddof=1))


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percentile interval for a paired mean difference."""

    values = np.asarray(differences, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def build_prediction_records(
    *,
    model_tag: str,
    model_name: str,
    query_ids: Sequence[str],
    query_sequences: Sequence[str],
    query_labels: Mapping[str, Sequence[str]],
    lookup_ids: Sequence[str],
    lookup_labels: Mapping[str, Sequence[str]],
    nearest_indices: np.ndarray,
    identities: Mapping[str, CathIdentity] | None = None,
) -> list[dict[str, Any]]:
    """Create complete per-query truth/prediction records for one model."""

    n_query = len(query_sequences)
    if len(query_ids) != n_query or len(nearest_indices) != n_query:
        raise ValueError(
            "query ids, sequences, and nearest-neighbour indices differ in length"
        )
    if any(len(query_labels[level]) != n_query for level in CATH_LEVELS):
        raise ValueError("query label columns differ in length")
    if any(len(lookup_labels[level]) != len(lookup_ids) for level in CATH_LEVELS):
        raise ValueError("lookup ids and label columns differ in length")
    if np.any(nearest_indices < 0) or np.any(nearest_indices >= len(lookup_ids)):
        raise ValueError("nearest-neighbour index outside lookup range")

    predictions = {
        level: np.asarray(lookup_labels[level], dtype=str)[nearest_indices]
        for level in CATH_LEVELS
    }
    answerable = {
        level: answerable_mask(query_labels[level], lookup_labels[level])
        for level in CATH_LEVELS
    }
    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for index, (domain_id, sequence) in enumerate(zip(query_ids, query_sequences)):
        sequence_hash = sequence_sha256(sequence)
        query_hash = f"cath_{sequence_hash[:16]}"
        if query_hash in seen_hashes:
            raise ValueError(f"duplicate query sequence hash {query_hash}")
        seen_hashes.add(query_hash)
        identity = identities.get(query_hash) if identities is not None else None
        if identities is not None and identity is None:
            raise ValueError(f"query {query_hash} missing from identity table")
        if identity is not None:
            if (
                identity.domain_id != domain_id
                or identity.sequence_sha256 != sequence_hash
            ):
                raise ValueError(f"dataset/identity metadata mismatch for {query_hash}")
            if bool(answerable["cath_h"][index]) != identity.in_test_h:
                raise ValueError(f"H-answerable membership mismatch for {query_hash}")

        truth = {level: str(query_labels[level][index]) for level in CATH_LEVELS}
        predicted = {level: str(predictions[level][index]) for level in CATH_LEVELS}
        is_answerable = {level: bool(answerable[level][index]) for level in CATH_LEVELS}
        records.append(
            {
                "schema_version": 1,
                "model_tag": model_tag,
                "model": model_name,
                "query_index": index,
                "domain_id": domain_id,
                "query_hash": query_hash,
                "sequence_sha256": sequence_hash,
                "nearest_lookup_index": int(nearest_indices[index]),
                "nearest_lookup_domain_id": str(
                    lookup_ids[int(nearest_indices[index])]
                ),
                "truth": truth,
                "prediction": predicted,
                "answerable": is_answerable,
                "correct": {
                    level: predicted[level] == truth[level] for level in CATH_LEVELS
                },
                "max_corpus_identity": identity.max_identity
                if identity is not None
                else None,
                "corpus_exact": identity.exact if identity is not None else None,
                "corpus_ge_90": identity.ge_90 if identity is not None else None,
                "in_test_h": identity.in_test_h if identity is not None else None,
            }
        )
    return records


def write_prediction_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write deterministic per-query JSON lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def read_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load per-query predictions, rejecting duplicate or mixed-model rows."""

    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"prediction file is empty: {path}")
    tags = {record.get("model_tag") for record in records}
    hashes = [record.get("query_hash") for record in records]
    if len(tags) != 1 or None in tags:
        raise ValueError(f"prediction file mixes model tags: {path}")
    if len(set(hashes)) != len(hashes) or None in hashes:
        raise ValueError(f"prediction file has duplicate/missing query hashes: {path}")
    return records


def _identity_for_record(
    record: Mapping[str, Any],
    identities: Mapping[str, CathIdentity] | None,
) -> tuple[float, bool, bool]:
    """Resolve identity fields from an external audit or the persisted row."""

    if identities is not None:
        query_hash = str(record["query_hash"])
        identity = identities.get(query_hash)
        if identity is None:
            raise ValueError(
                f"prediction query {query_hash} missing from identity table"
            )
        if identity.domain_id != record["domain_id"]:
            raise ValueError(f"prediction/identity domain mismatch for {query_hash}")
        if identity.sequence_sha256 != record["sequence_sha256"]:
            raise ValueError(f"prediction/identity sequence mismatch for {query_hash}")
        return identity.max_identity, identity.exact, identity.ge_90
    max_identity = record.get("max_corpus_identity")
    exact = record.get("corpus_exact")
    ge_90 = record.get("corpus_ge_90")
    if max_identity is None or exact is None or ge_90 is None:
        raise ValueError(
            "predictions contain no identity fields; pass --identity-table"
        )
    return float(max_identity), bool(exact), bool(ge_90)


def score_identity_strata(
    records: Sequence[Mapping[str, Any]],
    *,
    identities: Mapping[str, CathIdentity] | None = None,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Score C/A/T/H accuracy over full, exact, and >=90% identity strata."""

    if not records:
        raise ValueError("cannot score an empty prediction set")
    if identities is not None and len(records) == 219:
        expected_hashes = {
            query_hash
            for query_hash, identity in identities.items()
            if identity.in_test219
        }
        observed_hashes = {str(record["query_hash"]) for record in records}
        if observed_hashes != expected_hashes:
            raise ValueError(
                "prediction rows do not exactly match identity-table test219 queries"
            )
    resolved = [_identity_for_record(record, identities) for record in records]
    selectors = {
        "full": np.ones(len(records), dtype=bool),
        "exact": np.asarray([exact for _, exact, _ in resolved]),
        "non_exact": np.asarray([not exact for _, exact, _ in resolved]),
        "ge_90": np.asarray([ge_90 for _, _, ge_90 in resolved]),
        "lt_90": np.asarray([not ge_90 for _, _, ge_90 in resolved]),
    }
    levels: dict[str, Any] = {}
    for level in CATH_LEVELS:
        answerable = np.asarray(
            [bool(record["answerable"][level]) for record in records]
        )
        correct = np.asarray([bool(record["correct"][level]) for record in records])
        level_scores: dict[str, Any] = {}
        for stratum, selector in selectors.items():
            selected = answerable & selector
            values = correct[selected].astype(np.float64)
            n = int(selected.sum())
            level_scores[stratum] = {
                "accuracy": float(values.mean()) if n else None,
                "ci95": bootstrap_ci(values, n_boot=n_boot) if n else None,
                "n": n,
                "n_queries_in_stratum": int(selector.sum()),
                "diagnostic": n < 20,
            }
        levels[level] = level_scores

    h_counts = {stratum: levels["cath_h"][stratum]["n"] for stratum in STRATA}
    if len(records) == 219 and h_counts != EXPECTED_H_STRATA:
        raise ValueError(
            f"H-level identity strata drift: expected {EXPECTED_H_STRATA}, observed {h_counts}"
        )
    return {
        "schema_version": 1,
        "model_tag": str(records[0]["model_tag"]),
        "model": str(records[0]["model"]),
        "n_predictions": len(records),
        "levels": levels,
    }


def score_paired_identity_strata(
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    identities: Mapping[str, CathIdentity] | None = None,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Compare two models on the same queries with a paired bootstrap.

    Pairing matters because both models answer the same CATH queries. Treating
    their accuracies as independent throws away that information and produces
    the wrong uncertainty estimate for the model-to-model difference.
    """

    if not baseline_records or not candidate_records:
        raise ValueError("cannot compare an empty prediction set")
    baseline_by_hash = {
        str(record["query_hash"]): record for record in baseline_records
    }
    candidate_by_hash = {
        str(record["query_hash"]): record for record in candidate_records
    }
    if len(baseline_by_hash) != len(baseline_records):
        raise ValueError("baseline predictions contain duplicate query hashes")
    if len(candidate_by_hash) != len(candidate_records):
        raise ValueError("candidate predictions contain duplicate query hashes")
    if baseline_by_hash.keys() != candidate_by_hash.keys():
        raise ValueError("paired predictions do not contain the same query hashes")

    ordered_hashes = sorted(baseline_by_hash)
    baseline = [baseline_by_hash[query_hash] for query_hash in ordered_hashes]
    candidate = [candidate_by_hash[query_hash] for query_hash in ordered_hashes]
    for left, right in zip(baseline, candidate):
        query_hash = str(left["query_hash"])
        for field in ("domain_id", "sequence_sha256", "truth", "answerable"):
            if left[field] != right[field]:
                raise ValueError(
                    f"paired prediction metadata mismatch for {query_hash}/{field}"
                )

    resolved = [_identity_for_record(record, identities) for record in baseline]
    candidate_resolved = [
        _identity_for_record(record, identities) for record in candidate
    ]
    if resolved != candidate_resolved:
        raise ValueError("paired predictions contain different identity metadata")
    selectors = {
        "full": np.ones(len(baseline), dtype=bool),
        "exact": np.asarray([exact for _, exact, _ in resolved]),
        "non_exact": np.asarray([not exact for _, exact, _ in resolved]),
        "ge_90": np.asarray([ge_90 for _, _, ge_90 in resolved]),
        "lt_90": np.asarray([not ge_90 for _, _, ge_90 in resolved]),
    }

    levels: dict[str, Any] = {}
    for level in CATH_LEVELS:
        answerable = np.asarray(
            [bool(record["answerable"][level]) for record in baseline]
        )
        baseline_correct = np.asarray(
            [bool(record["correct"][level]) for record in baseline]
        )
        candidate_correct = np.asarray(
            [bool(record["correct"][level]) for record in candidate]
        )
        level_scores: dict[str, Any] = {}
        for stratum, selector in selectors.items():
            selected = answerable & selector
            left = baseline_correct[selected]
            right = candidate_correct[selected]
            differences = right.astype(np.float64) - left.astype(np.float64)
            n = int(selected.sum())
            if n:
                low, high = paired_bootstrap_interval(differences, n_boot=n_boot)
            else:
                low, high = None, None
            level_scores[stratum] = {
                "accuracy_delta": float(differences.mean()) if n else None,
                "ci95_low": low,
                "ci95_high": high,
                "n": n,
                "candidate_only_correct": int((right & ~left).sum()),
                "baseline_only_correct": int((left & ~right).sum()),
                "diagnostic": n < 20,
            }
        levels[level] = level_scores

    return {
        "schema_version": 1,
        "baseline_tag": str(baseline[0]["model_tag"]),
        "candidate_tag": str(candidate[0]["model_tag"]),
        "n_predictions": len(baseline),
        "levels": levels,
    }


def render_strata_markdown(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Render compact C/A/T/H identity-stratified accuracy tables."""

    lines = [
        "# CATH-EAT corpus-identity strata",
        "",
        "Existing checkpoints are corpus-contaminated. These strata are diagnostic and do",
        "not turn any subset into a clean absolute result. Values are accuracy +/- 95%",
        "bootstrap-CI half-width; `*` marks n < 20.",
    ]
    for tag, result in results.items():
        lines += [
            "",
            f"## {tag}",
            "",
            "| Level | Full | Exact | Non-exact | >=90% | <90% |",
        ]
        lines.append("|---|---:|---:|---:|---:|---:|")
        for level in CATH_LEVELS:
            cells = []
            for stratum in STRATA:
                metric = result["levels"][level][stratum]
                if metric["accuracy"] is None:
                    cells.append("-- (n=0)")
                    continue
                marker = "*" if metric["diagnostic"] else ""
                cells.append(
                    f"{100 * metric['accuracy']:.1f} +/- {100 * metric['ci95']:.1f} "
                    f"(n={metric['n']}){marker}"
                )
            lines.append(f"| {level[-1].upper()} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
