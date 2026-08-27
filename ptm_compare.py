"""Consolidate PTM validation reports and verify a protocol-matched panel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def flatten_report(report: dict[str, object]) -> list[dict[str, object]]:
    tag = str(report["model_tag"])
    rows: list[dict[str, object]] = []
    modes = report.get("modes")
    if not isinstance(modes, dict):
        raise ValueError(f"{tag}: modes is not an object")
    for input_mode, methods_value in modes.items():
        if not isinstance(methods_value, dict):
            raise ValueError(f"{tag}/{input_mode}: methods is not an object")
        for method, tasks_value in methods_value.items():
            if method == "direct_ptm_any":
                task_metrics = {"site": tasks_value}
            else:
                task_metrics = tasks_value
            if not isinstance(task_metrics, dict):
                raise ValueError(f"{tag}/{input_mode}/{method}: metrics is not an object")
            for task, metrics in task_metrics.items():
                if not isinstance(metrics, dict):
                    raise ValueError(
                        f"{tag}/{input_mode}/{method}/{task}: metrics is not an object"
                    )
                row: dict[str, object] = {
                    "model_tag": tag,
                    "input_mode": str(input_mode),
                    "method": str(method),
                    "task": str(task),
                    "fit_rows": report.get("fit_rows"),
                    "validation_rows": report.get("validation_rows"),
                    "seed": report.get("seed"),
                }
                row.update(metrics)
                rows.append(row)
    return rows


def build_comparison(
    reports: list[dict[str, object]],
    *,
    baselines: dict[str, str] | None = None,
) -> dict[str, object]:
    if not reports:
        raise ValueError("no PTM reports supplied")
    protocols = {str(report.get("protocol")) for report in reports}
    seeds = {report.get("seed") for report in reports}
    fit_rows = {report.get("fit_rows") for report in reports}
    validation_rows = {report.get("validation_rows") for report in reports}
    if len(protocols) != 1 or len(seeds) != 1 or len(fit_rows) != 1 or len(validation_rows) != 1:
        raise ValueError("PTM reports do not share protocol, seed, and split sizes")

    rows = [row for report in reports for row in flatten_report(report)]
    canonical_sites = [
        row
        for row in rows
        if row["input_mode"] == "canonical"
        and row["method"] == "frozen_probe"
        and row["task"] == "site"
    ]
    panels = {(row.get("n_sites"), row.get("n_positive")) for row in canonical_sites}
    if len(panels) != 1:
        raise ValueError(f"canonical PTM residue panels differ: {sorted(panels)}")

    index = {
        (row["model_tag"], row["input_mode"], row["method"], row["task"]): row
        for row in rows
    }
    for row in rows:
        baseline_tag = (baselines or {}).get(str(row["model_tag"]))
        if not baseline_tag:
            continue
        baseline = index.get(
            (baseline_tag, row["input_mode"], row["method"], row["task"])
        )
        if baseline is None:
            continue
        row["baseline_tag"] = baseline_tag
        for metric in ("auprc", "auroc", "mcc", "f1", "top1", "top3", "mrr"):
            value = row.get(metric)
            baseline_value = baseline.get(metric)
            if isinstance(value, (int, float)) and isinstance(
                baseline_value, (int, float)
            ):
                row[f"delta_{metric}"] = float(value) - float(baseline_value)

    return {
        "schema_version": 1,
        "protocol": next(iter(protocols)),
        "seed": next(iter(seeds)),
        "fit_rows": next(iter(fit_rows)),
        "validation_rows": next(iter(validation_rows)),
        "canonical_site_panel": {
            "n_sites": next(iter(panels))[0],
            "n_positive": next(iter(panels))[1],
        },
        "rows": rows,
    }


def write_comparison(comparison: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ptm_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    rows = comparison["rows"]
    if not isinstance(rows, list):
        raise TypeError("comparison rows must be a list")
    fieldnames = sorted({key for row in rows for key in row})
    with (out_dir / "ptm_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--baseline", action="append", default=[], metavar="MODEL=BASELINE")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    baselines = {}
    for value in args.baseline:
        model, separator, baseline = value.partition("=")
        if not separator or not model or not baseline:
            parser.error(f"invalid --baseline {value!r}; expected MODEL=BASELINE")
        baselines[model] = baseline
    reports = [json.loads(path.read_text()) for path in args.reports]
    comparison = build_comparison(reports, baselines=baselines)
    write_comparison(comparison, args.out)
    print(json.dumps({key: value for key, value in comparison.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
