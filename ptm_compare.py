"""Consolidate PTM validation reports and verify a protocol-matched panel."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from ptm_benchmark import PTMSitePrediction


def _weighted_average_precision(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float:
    """Average precision with each protein duplicated by its bootstrap count."""
    if not (labels * weights).any():
        return float("nan")
    return float(average_precision_score(labels, scores, sample_weight=weights))


def paired_site_auprc_bootstrap(
    candidate: Sequence[PTMSitePrediction],
    baseline: Sequence[PTMSitePrediction],
    *,
    n_boot: int = 1000,
    seed: int = 1337,
    group_key: Callable[[str], str] | None = None,
    resampling_unit: str = "row_id",
) -> dict[str, object]:
    """Paired protein-cluster bootstrap CI for a residue-level AUPRC delta.

    Predictions are aligned by ``(row_id, position)`` and whole groups are
    resampled, so residues from one protein never masquerade as independent
    observations. ``group_key`` maps a row_id to its resampling group; pass a
    matching ``resampling_unit`` so the report says what was resampled.
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    def index(
        records: Sequence[PTMSitePrediction], name: str
    ) -> dict[tuple[str, int], PTMSitePrediction]:
        result: dict[tuple[str, int], PTMSitePrediction] = {}
        for record in records:
            key = (record.row_id, record.position)
            if key in result:
                raise ValueError(f"duplicate {name} prediction key: {key}")
            result[key] = record
        return result

    candidate_index = index(candidate, "candidate")
    baseline_index = index(baseline, "baseline")
    if candidate_index.keys() != baseline_index.keys():
        raise ValueError("candidate and baseline prediction keys differ")
    if not candidate_index:
        raise ValueError("prediction panels are empty")

    keys = sorted(candidate_index)
    labels = np.asarray([candidate_index[key].label for key in keys], dtype=np.int8)
    baseline_labels = np.asarray(
        [baseline_index[key].label for key in keys], dtype=np.int8
    )
    if not np.array_equal(labels, baseline_labels):
        raise ValueError("candidate and baseline labels differ")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("PTM site labels must be binary")

    resolved_group_key = group_key or (lambda row_id: row_id)
    grouped_row_ids = [resolved_group_key(key[0]) for key in keys]
    if any(not group_id for group_id in grouped_row_ids):
        raise ValueError("bootstrap group keys must be non-empty")
    group_ids = {
        group_id: index for index, group_id in enumerate(sorted(set(grouped_row_ids)))
    }
    group_codes = np.asarray(
        [group_ids[group_id] for group_id in grouped_row_ids], dtype=np.int32
    )
    candidate_scores = np.asarray(
        [candidate_index[key].score for key in keys], dtype=np.float64
    )
    baseline_scores = np.asarray(
        [baseline_index[key].score for key in keys], dtype=np.float64
    )
    if (
        not np.isfinite(candidate_scores).all()
        or not np.isfinite(baseline_scores).all()
    ):
        raise ValueError("PTM site scores must all be finite")

    n_groups = len(group_ids)
    unit_weights = np.ones(len(keys), dtype=np.int64)
    candidate_auprc = _weighted_average_precision(labels, candidate_scores, unit_weights)
    baseline_auprc = _weighted_average_precision(labels, baseline_scores, unit_weights)

    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(n_boot):
        counts = np.bincount(
            rng.integers(0, n_groups, size=n_groups), minlength=n_groups
        )[group_codes]
        candidate_boot = _weighted_average_precision(labels, candidate_scores, counts)
        baseline_boot = _weighted_average_precision(labels, baseline_scores, counts)
        if np.isfinite(candidate_boot) and np.isfinite(baseline_boot):
            replicates.append(candidate_boot - baseline_boot)
    if not replicates:
        raise ValueError("no bootstrap replicate contained a positive label")
    deltas = np.asarray(replicates)

    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "metric": "auprc",
        "resampling_unit": resampling_unit,
        "n_sites": len(keys),
        "n_positive": int(labels.sum()),
        "n_groups": n_groups,
        "candidate_auprc": candidate_auprc,
        "baseline_auprc": baseline_auprc,
        "delta_auprc": candidate_auprc - baseline_auprc,
        "delta_ci_low": float(low),
        "delta_ci_high": float(high),
        "bootstrap_fraction_delta_gt_zero": float(np.mean(deltas > 0.0)),
        "n_boot_requested": n_boot,
        "n_boot_effective": int(deltas.size),
        "seed": seed,
    }


def flatten_report(report: dict[str, object]) -> list[dict[str, object]]:
    tag = str(report["model_tag"])
    rows: list[dict[str, object]] = []
    modes = report.get("modes")
    if not isinstance(modes, dict):
        raise TypeError(f"{tag}: modes is not an object")
    for input_mode, methods_value in modes.items():
        if not isinstance(methods_value, dict):
            raise TypeError(f"{tag}/{input_mode}: methods is not an object")
        for method, tasks_value in methods_value.items():
            if method == "direct_ptm_any":
                task_metrics = {"site": tasks_value}
            else:
                task_metrics = tasks_value
            if not isinstance(task_metrics, dict):
                raise TypeError(
                    f"{tag}/{input_mode}/{method}: metrics is not an object"
                )
            for task, metrics in task_metrics.items():
                if not isinstance(metrics, dict):
                    raise TypeError(
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
    if (
        len(protocols) != 1
        or len(seeds) != 1
        or len(fit_rows) != 1
        or len(validation_rows) != 1
    ):
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


def flatten_external_report(report: dict[str, object]) -> list[dict[str, object]]:
    """Flatten phosphosite/NHAC frozen-probe reports into comparable rows."""

    tag = str(report["model_tag"])
    rows: list[dict[str, object]] = []
    for split in (
        "validation",
        "author_test",
        "repository_test",
        "exact_dedup_test",
    ):
        metrics = report.get(split)
        if not isinstance(metrics, dict):
            continue
        row: dict[str, object] = {
            "model_tag": tag,
            "task": report.get("task"),
            "split": split,
            "method": "frozen_probe",
            "seed": report.get("seed"),
        }
        row.update(metrics)
        rows.append(row)
    residue_baseline = report.get("residue_only_baseline")
    if isinstance(residue_baseline, dict):
        for split, metrics in residue_baseline.items():
            if not isinstance(metrics, dict):
                continue
            row = {
                "model_tag": tag,
                "task": report.get("task"),
                "split": str(split),
                "method": "residue_only_baseline",
                "seed": report.get("seed"),
            }
            row.update(metrics)
            rows.append(row)
    return rows


def build_external_comparison(
    reports: list[dict[str, object]],
    *,
    baselines: dict[str, str] | None = None,
) -> dict[str, object]:
    if not reports:
        raise ValueError("no external PTM reports supplied")
    tasks = {str(report.get("task")) for report in reports}
    protocols = {str(report.get("protocol")) for report in reports}
    seeds = {report.get("seed") for report in reports}
    if len(tasks) != 1 or len(protocols) != 1 or len(seeds) != 1:
        raise ValueError("external PTM reports do not share task, protocol, and seed")
    rows = [row for report in reports for row in flatten_external_report(report)]
    frozen_rows = [row for row in rows if row["method"] == "frozen_probe"]
    panels: dict[str, set[tuple[object, object]]] = {}
    for row in frozen_rows:
        panels.setdefault(str(row["split"]), set()).add(
            (row.get("n_sites"), row.get("n_positive"))
        )
    mismatched = {split: values for split, values in panels.items() if len(values) != 1}
    if mismatched:
        raise ValueError(f"external PTM panels differ: {mismatched}")

    index = {(row["model_tag"], row["split"], row["method"]): row for row in rows}
    for row in rows:
        baseline_tag = (baselines or {}).get(str(row["model_tag"]))
        if not baseline_tag:
            continue
        baseline = index.get((baseline_tag, row["split"], row["method"]))
        if baseline is None:
            continue
        row["baseline_tag"] = baseline_tag
        for metric in ("auprc", "auroc", "mcc", "f1"):
            value = row.get(metric)
            baseline_value = baseline.get(metric)
            if isinstance(value, (int, float)) and isinstance(
                baseline_value, (int, float)
            ):
                row[f"delta_{metric}"] = float(value) - float(baseline_value)
    return {
        "schema_version": 1,
        "task": next(iter(tasks)),
        "protocol": next(iter(protocols)),
        "seed": next(iter(seeds)),
        "panels": {
            split: {
                "n_sites": next(iter(values))[0],
                "n_positive": next(iter(values))[1],
            }
            for split, values in sorted(panels.items())
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
    parser.add_argument(
        "--baseline", action="append", default=[], metavar="MODEL=BASELINE"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    baselines = {}
    for value in args.baseline:
        model, separator, baseline = value.partition("=")
        if not separator or not model or not baseline:
            parser.error(f"invalid --baseline {value!r}; expected MODEL=BASELINE")
        baselines[model] = baseline
    reports = [json.loads(path.read_text()) for path in args.reports]
    comparison = (
        build_comparison(reports, baselines=baselines)
        if "modes" in reports[0]
        else build_external_comparison(reports, baselines=baselines)
    )
    write_comparison(comparison, args.out)
    print(
        json.dumps(
            {key: value for key, value in comparison.items() if key != "rows"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
