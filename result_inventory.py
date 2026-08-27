"""Normalize benchmark CSVs from multiple trees without hiding provenance.

Three historical schemas are accepted: ResultTracker's wide CSV, the unified
long CSV, and Proteva's legacy ``experiment/task/metric/value`` CSV. Every
numeric metric becomes one canonical row. Missing protocol fields are written
as ``unknown``; they are never guessed, and conflicting measurements are never
silently collapsed.

Example:
    python result_inventory.py \
      --source current=/path/to/results \
      --source archive=/path/to/archive \
      --out /private/results/canonical_inventory.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

UNKNOWN = "unknown"

REQUIRED_PROTOCOL_FIELDS = (
    "checkpoint_hash",
    "task",
    "probe",
    "split",
    "sample_cap",
    "pooling",
    "normalization",
    "dataset_revision",
    "code_version",
)

OUTPUT_COLUMNS = (
    *REQUIRED_PROTOCOL_FIELDS,
    "metric_name",
    "metric_value",
    "model",
    "checkpoint_hash_kind",
    "date",
    "eval_mode",
    "eval_strategy",
    "benchmark_seed",
    "notes",
    "source_name",
    "source_file",
    "source_row",
    "source_schema",
    "protocol_key",
    "metric_key",
    "provenance_complete",
    "missing_provenance",
)

WIDE_META = {
    "Model",
    "Task",
    "Samples",
    "Date",
    "Probe",
    "EvalMode",
    "EvalSplit",
    "EvalStrategy",
    "EmbeddingNorm",
    "Pooling",
    "DatasetRevision",
    "CheckpointHash",
    "CheckpointHashKind",
    "CodeVersion",
    "BenchmarkSeed",
    "Error",
    "Notes",
    "ProbeFitSec",
    "n_queries",
    "n_eligible_queries",
    "Proteins_Scored",
}


@dataclass(frozen=True)
class Source:
    name: str
    path: Path


def _clean(value: object, default: str = UNKNOWN) -> str:
    text = "" if value is None else str(value).strip()
    return text if text and text.lower() != "nan" else default


def _number(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _manifest_hash(path: Path) -> tuple[str, str]:
    """Cheap deterministic local checkpoint fingerprint, explicitly not content hash."""

    entries = []
    patterns = ("config.json", "*.safetensors", "*.bin", "*.index.json")
    for pattern in patterns:
        for item in sorted(path.glob(pattern)):
            stat = item.stat()
            entries.append((item.name, stat.st_size, stat.st_mtime_ns))
    if not entries:
        return "ref-sha256:" + hashlib.sha256(str(path).encode()).hexdigest(), "reference"
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=True)
    return "manifest-sha256:" + hashlib.sha256(payload.encode()).hexdigest(), "local-manifest"


def checkpoint_fingerprint(
    model: str,
    *,
    explicit_hash: str = "",
    explicit_kind: str = "",
    model_bases: Iterable[Path] = (),
) -> tuple[str, str]:
    """Return an honest fingerprint and its strength; never label a ref as content."""

    if _clean(explicit_hash) != UNKNOWN:
        return _clean(explicit_hash), _clean(explicit_kind, "recorded")
    snapshot = re.search(r"(?:^|/)snapshots/([0-9a-fA-F]{7,64})(?:/|$)", model)
    if snapshot:
        return "hf-revision:" + snapshot.group(1).lower(), "hf-revision"
    raw = Path(model).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(base / raw for base in model_bases)
    for candidate in candidates:
        if candidate.is_dir():
            return _manifest_hash(candidate.resolve())
    digest = hashlib.sha256(model.encode()).hexdigest()
    return "ref-sha256:" + digest, "reference"


def _keys(row: dict[str, object]) -> tuple[str, str]:
    protocol_payload = {key: row[key] for key in REQUIRED_PROTOCOL_FIELDS}
    protocol = json.dumps(protocol_payload, sort_keys=True, separators=(",", ":"))
    metric = json.dumps(
        {**protocol_payload, "metric_name": row["metric_name"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return protocol, hashlib.sha256(metric.encode()).hexdigest()


def _base_record(
    raw: dict[str, str],
    *,
    source: Source,
    source_file: Path,
    source_row: int,
    schema: str,
    model_bases: Iterable[Path],
) -> dict[str, object]:
    if schema == "wide":
        model = _clean(raw.get("Model"))
        checkpoint_hash, hash_kind = checkpoint_fingerprint(
            model,
            explicit_hash=raw.get("CheckpointHash", ""),
            explicit_kind=raw.get("CheckpointHashKind", ""),
            model_bases=model_bases,
        )
        return {
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_hash_kind": hash_kind,
            "model": model,
            "task": _clean(raw.get("Task")),
            "probe": _clean(raw.get("Probe"), "linear"),
            "split": _clean(raw.get("EvalSplit")),
            "sample_cap": _clean(raw.get("Samples")),
            "pooling": _clean(raw.get("Pooling")),
            "normalization": _clean(raw.get("EmbeddingNorm")),
            "dataset_revision": _clean(raw.get("DatasetRevision")),
            "code_version": _clean(raw.get("CodeVersion")),
            "date": _clean(raw.get("Date"), ""),
            "eval_mode": _clean(raw.get("EvalMode")),
            "eval_strategy": _clean(raw.get("EvalStrategy")),
            "benchmark_seed": _clean(raw.get("BenchmarkSeed"), ""),
            "notes": _clean(raw.get("Notes"), ""),
            "source_name": source.name,
            "source_file": str(source_file),
            "source_row": source_row,
            "source_schema": schema,
        }
    if schema == "long":
        model = _clean(raw.get("model"))
        checkpoint_hash, hash_kind = checkpoint_fingerprint(
            model,
            explicit_hash=raw.get("checkpoint_hash", ""),
            explicit_kind=raw.get("checkpoint_hash_kind", ""),
            model_bases=model_bases,
        )
        return {
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_hash_kind": hash_kind,
            "model": model,
            "task": _clean(raw.get("task")),
            "probe": _clean(raw.get("probe_type"), "linear"),
            "split": _clean(raw.get("split")),
            "sample_cap": _clean(raw.get("n_samples")),
            "pooling": _clean(raw.get("pooling")),
            "normalization": _clean(raw.get("normalization")),
            "dataset_revision": _clean(raw.get("dataset_revision")),
            "code_version": _clean(raw.get("code_version")),
            "date": _clean(raw.get("date"), ""),
            "eval_mode": _clean(raw.get("eval_mode")),
            "eval_strategy": _clean(raw.get("eval_strategy")),
            "benchmark_seed": _clean(raw.get("benchmark_seed"), ""),
            "notes": _clean(raw.get("notes"), ""),
            "source_name": source.name,
            "source_file": str(source_file),
            "source_row": source_row,
            "source_schema": schema,
        }
    model = _clean(raw.get("experiment"))
    checkpoint_hash, hash_kind = checkpoint_fingerprint(model, model_bases=model_bases)
    return {
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_hash_kind": hash_kind,
        "model": model,
        "task": _clean(raw.get("task")),
        "probe": _clean(raw.get("probe"), "linear"),
        "split": _clean(raw.get("split")),
        "sample_cap": _clean(raw.get("n_samples")),
        "pooling": _clean(raw.get("pooling")),
        "normalization": _clean(raw.get("normalization")),
        "dataset_revision": _clean(raw.get("dataset_revision")),
        "code_version": _clean(raw.get("code_version")),
        "date": _clean(raw.get("date"), ""),
        "eval_mode": _clean(raw.get("eval_mode")),
        "eval_strategy": _clean(raw.get("eval_strategy")),
        "benchmark_seed": _clean(raw.get("seed"), ""),
        "notes": _clean(raw.get("notes"), ""),
        "source_name": source.name,
        "source_file": str(source_file),
        "source_row": source_row,
        "source_schema": schema,
    }


def _detect_schema(fieldnames: Sequence[str]) -> str | None:
    fields = set(fieldnames)
    if {"Model", "Task"} <= fields:
        return "wide"
    if {"model", "task", "metric_name", "metric_value"} <= fields:
        return "long"
    if {"experiment", "task", "metric", "value"} <= fields:
        return "legacy"
    return None


def read_result_file(
    path: Path,
    source: Source,
    *,
    model_bases: Iterable[Path] = (),
) -> Iterator[dict[str, object]]:
    """Yield canonical metric rows from one supported CSV."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        schema = _detect_schema(reader.fieldnames or [])
        if schema is None:
            return
        for source_row, raw in enumerate(reader, start=2):
            base = _base_record(
                raw,
                source=source,
                source_file=path,
                source_row=source_row,
                schema=schema,
                model_bases=model_bases,
            )
            metrics: list[tuple[str, object]]
            if schema == "wide":
                metrics = [(key, value) for key, value in raw.items() if key not in WIDE_META]
            elif schema == "long":
                metrics = [(raw.get("metric_name", ""), raw.get("metric_value", ""))]
            else:
                metrics = [(raw.get("metric", ""), raw.get("value", ""))]
            for metric_name, value in metrics:
                number = _number(value)
                if number is None or not str(metric_name).strip():
                    continue
                record = {**base, "metric_name": str(metric_name), "metric_value": number}
                missing = [
                    field for field in REQUIRED_PROTOCOL_FIELDS if record[field] == UNKNOWN
                ]
                protocol_key, metric_key = _keys(record)
                record.update(
                    {
                        "protocol_key": protocol_key,
                        "metric_key": metric_key,
                        "provenance_complete": not missing,
                        "missing_provenance": ";".join(missing),
                    }
                )
                yield record


def _result_files(source: Source) -> Iterator[Path]:
    if source.path.is_file():
        yield source.path
        return
    if not source.path.is_dir():
        return
    for path in sorted(source.path.rglob("*")):
        if path.is_file() and ".csv" in path.name.lower():
            yield path


def build_inventory(
    sources: list[Source],
    *,
    model_bases: Iterable[Path] = (),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Read all sources, assigning overlapping files to the most specific source."""

    rows: list[dict[str, object]] = []
    files_seen: set[Path] = set()
    files_by_source: Counter[str] = Counter()
    unsupported_by_source: Counter[str] = Counter()
    for source in sorted(sources, key=lambda item: len(str(item.path)), reverse=True):
        for path in _result_files(source):
            resolved = path.resolve()
            if resolved in files_seen:
                continue
            files_seen.add(resolved)
            files_by_source[source.name] += 1
            before = len(rows)
            try:
                rows.extend(read_result_file(path, source, model_bases=model_bases))
            except (OSError, UnicodeError, csv.Error):
                unsupported_by_source[source.name] += 1
                continue
            if len(rows) == before:
                unsupported_by_source[source.name] += 1

    rows.sort(
        key=lambda row: (
            str(row["model"]),
            str(row["task"]),
            str(row["probe"]),
            str(row["metric_name"]),
            str(row["source_file"]),
            int(str(row["source_row"])),
        )
    )
    key_counts = Counter(str(row["metric_key"]) for row in rows)
    conflicts = 0
    for key, count in key_counts.items():
        if count < 2:
            continue
        values = {row["metric_value"] for row in rows if row["metric_key"] == key}
        conflicts += int(len(values) > 1)
    summary = {
        "schema_version": 1,
        "sources": [
            {
                "name": source.name,
                "path": str(source.path),
                "exists": source.path.exists(),
                "files_seen": files_by_source[source.name],
                "unsupported_or_empty_files": unsupported_by_source[source.name],
                "metric_rows": sum(row["source_name"] == source.name for row in rows),
            }
            for source in sources
        ],
        "metric_rows": len(rows),
        "unique_metric_keys": len(key_counts),
        "duplicate_metric_keys": sum(count > 1 for count in key_counts.values()),
        "conflicting_metric_keys": conflicts,
        "complete_provenance_rows": sum(bool(row["provenance_complete"]) for row in rows),
    }
    return rows, summary


def write_inventory(
    rows: list[dict[str, object]], summary: dict[str, object], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in OUTPUT_COLUMNS} for row in rows)
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def _parse_source(value: str) -> Source:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("source must be NAME=PATH")
    return Source(name=name, path=Path(path).expanduser())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--model-base", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows, summary = build_inventory(args.source, model_bases=args.model_base)
    write_inventory(rows, summary, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
