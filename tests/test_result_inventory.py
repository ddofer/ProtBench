from __future__ import annotations

import csv
from pathlib import Path

from result_inventory import Source, build_inventory, read_result_file, write_inventory


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_reads_wide_long_and_legacy_without_guessing_missing_provenance(
    tmp_path: Path,
) -> None:
    wide = tmp_path / "wide.csv"
    _write_csv(
        wide,
        [
            "Model",
            "Task",
            "Samples",
            "Probe",
            "EvalSplit",
            "Pooling",
            "EmbeddingNorm",
            "DatasetRevision",
            "CodeVersion",
            "AUC",
            "ProbeFitSec",
        ],
        [
            {
                "Model": "org/model",
                "Task": "Solubility",
                "Samples": "Full",
                "Probe": "linear",
                "EvalSplit": "test",
                "Pooling": "mean",
                "EmbeddingNorm": "none",
                "DatasetRevision": "rev1",
                "CodeVersion": "abc123",
                "AUC": 0.8,
                "ProbeFitSec": 1.5,
            }
        ],
    )
    rows = list(read_result_file(wide, Source("wide", wide)))
    assert len(rows) == 1
    assert rows[0]["metric_name"] == "AUC"
    assert rows[0]["provenance_complete"] is True
    assert "ProbeFitSec" not in {row["metric_name"] for row in rows}

    long = tmp_path / "long.csv"
    _write_csv(
        long,
        ["model", "task", "probe_type", "split", "metric_name", "metric_value"],
        [
            {
                "model": "org/model",
                "task": "Stability",
                "probe_type": "linear",
                "split": "test",
                "metric_name": "Spearman",
                "metric_value": 0.5,
            }
        ],
    )
    long_row = next(read_result_file(long, Source("long", long)))
    assert long_row["pooling"] == "unknown"
    assert "pooling" in str(long_row["missing_provenance"])
    assert long_row["provenance_complete"] is False

    legacy = tmp_path / "legacy.csv"
    _write_csv(
        legacy,
        ["experiment", "task", "probe", "split", "metric", "value", "seed"],
        [
            {
                "experiment": "old-arm",
                "task": "EC",
                "probe": "linear",
                "split": "test",
                "metric": "F1_Macro",
                "value": 0.4,
                "seed": 42,
            }
        ],
    )
    legacy_row = next(read_result_file(legacy, Source("legacy", legacy)))
    assert legacy_row["source_schema"] == "legacy"
    assert legacy_row["metric_value"] == 0.4


def test_inventory_assigns_nested_file_to_most_specific_source_and_keeps_conflicts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    archive = root / "archive"
    fields = [
        "model",
        "task",
        "probe_type",
        "split",
        "metric_name",
        "metric_value",
        "code_version",
    ]
    _write_csv(
        root / "live.csv",
        fields,
        [
            {
                "model": "m",
                "task": "t",
                "probe_type": "linear",
                "split": "test",
                "metric_name": "AUC",
                "metric_value": 0.7,
                "code_version": "v1",
            }
        ],
    )
    _write_csv(
        archive / "old.csv",
        fields,
        [
            {
                "model": "m",
                "task": "t",
                "probe_type": "linear",
                "split": "test",
                "metric_name": "AUC",
                "metric_value": 0.6,
                "code_version": "v1",
            }
        ],
    )
    sources = [Source("root", root), Source("archive", archive)]
    rows, summary = build_inventory(sources)
    assert len(rows) == 2
    assert {row["source_name"] for row in rows} == {"root", "archive"}
    assert summary["metric_rows"] == 2
    assert summary["conflicting_metric_keys"] == 1

    out = root / "canonical_inventory.csv"
    write_inventory(rows, summary, out)
    assert out.exists()
    assert out.with_suffix(".summary.json").exists()
    repeated_rows, repeated_summary = build_inventory(sources)
    assert len(repeated_rows) == 2
    assert repeated_summary["metric_rows"] == 2
