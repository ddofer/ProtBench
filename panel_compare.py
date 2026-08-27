"""Compare complete benchmark panels with transparent task-level aggregation.

This is intentionally a descriptive *task bootstrap*: it resamples the observed
task deltas, not examples within a task and not training seeds. It is useful for
showing how fragile a panel aggregate is, but is not a substitute for per-task
prediction bootstraps or replicate training runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from benchmark_tasks import TASK_NAME_TO_KEY, TASKS, metric_greater_is_better
from benchmark_utils import METRIC_PRIORITY

PROTOCOL_COLUMNS = (
    "Task",
    "Samples",
    "Probe",
    "EvalMode",
    "EvalSplit",
    "EvalStrategy",
    "EmbeddingNorm",
    "Pooling",
)

DEFAULTS = {
    "Samples": "Full",
    "Probe": "linear",
    "EvalMode": "standard",
    "EvalSplit": "test",
    "EvalStrategy": "test_split",
    "EmbeddingNorm": "unknown",
    "Pooling": "unknown",
    "CodeVersion": "unknown",
}


def _prepare(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or "Task" not in frame or "Model" not in frame:
        raise ValueError(f"{path} is not a non-empty ResultTracker CSV")
    for column, default in DEFAULTS.items():
        if column not in frame:
            frame[column] = default
        frame[column] = frame[column].fillna(default).astype(str)
    duplicates = frame.duplicated(list(PROTOCOL_COLUMNS), keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, list(PROTOCOL_COLUMNS)].to_dict("records")
        raise ValueError(f"{path} has duplicate protocol rows: {keys[:3]}")
    return frame


def _metric_for(task: str, candidate: pd.Series, baseline: pd.Series) -> str:
    task_key = TASK_NAME_TO_KEY.get(task, task)
    cfg = TASKS.get(task_key)
    preferred = getattr(cfg, "main_metric", None)
    ordered = [preferred] if preferred else []
    ordered.extend(metric for metric in METRIC_PRIORITY if metric != preferred)
    for metric in ordered:
        if metric not in candidate.index or metric not in baseline.index:
            continue
        try:
            values = float(candidate[metric]), float(baseline[metric])
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values):
            return str(metric)
    raise ValueError(f"no common finite headline metric for task {task!r}")


def _interval(samples: np.ndarray) -> list[float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def compare_panel(
    candidate_path: Path,
    baseline_path: Path,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
    allow_baseline_superset: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return every matched task and a deterministic task-bootstrap summary."""

    candidate = _prepare(candidate_path)
    baseline = _prepare(baseline_path)
    key_columns = list(PROTOCOL_COLUMNS)
    candidate_map = {
        tuple(row[column] for column in key_columns): row
        for _, row in candidate.iterrows()
    }
    baseline_map = {
        tuple(row[column] for column in key_columns): row
        for _, row in baseline.iterrows()
    }
    candidate_keys = set(candidate_map)
    baseline_keys = set(baseline_map)
    if allow_baseline_superset and candidate_keys <= baseline_keys:
        baseline_map = {key: baseline_map[key] for key in candidate_keys}
    elif candidate_keys != baseline_keys:
        only_candidate = sorted(set(candidate_map) - set(baseline_map))
        only_baseline = sorted(set(baseline_map) - set(candidate_map))
        raise ValueError(
            "panel protocol mismatch; "
            f"candidate-only={only_candidate[:3]}, baseline-only={only_baseline[:3]}"
        )

    rows = []
    for key in sorted(candidate_map):
        cand = candidate_map[key]
        base = baseline_map[key]
        task = str(cand["Task"])
        metric = _metric_for(task, cand, base)
        candidate_value = float(cand[metric])
        baseline_value = float(base[metric])
        raw_delta = candidate_value - baseline_value
        signed_delta = raw_delta if metric_greater_is_better(metric) else -raw_delta
        rows.append(
            {
                "task": task,
                "metric": metric,
                "candidate": candidate_value,
                "baseline": baseline_value,
                "raw_delta": raw_delta,
                "signed_delta": signed_delta,
                "winner": "candidate" if signed_delta > 0 else "baseline" if signed_delta < 0 else "tie",
                "probe": cand["Probe"],
                "split": cand["EvalSplit"],
                "sample_cap": cand["Samples"],
                "pooling": cand["Pooling"],
                "normalization": cand["EmbeddingNorm"],
            }
        )
    table = pd.DataFrame(rows)
    deltas = table["signed_delta"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(deltas, size=(n_boot, len(deltas)), replace=True)
    summary = {
        "schema_version": 1,
        "candidate_model": str(candidate["Model"].iloc[0]),
        "baseline_model": str(baseline["Model"].iloc[0]),
        "candidate_code_versions": sorted(set(candidate["CodeVersion"])),
        "baseline_code_versions": sorted(set(baseline["CodeVersion"])),
        "tasks": len(table),
        "wins": int((deltas > 0).sum()),
        "losses": int((deltas < 0).sum()),
        "ties": int((deltas == 0).sum()),
        "median_signed_delta": float(np.median(deltas)),
        "median_task_bootstrap_ci95": _interval(np.median(draws, axis=1)),
        "mean_signed_delta": float(np.mean(deltas)),
        "mean_task_bootstrap_ci95": _interval(np.mean(draws, axis=1)),
        "bootstrap": {
            "unit": "task",
            "resamples": n_boot,
            "seed": seed,
            "interpretation": "descriptive panel stability; not a seed or per-example CI",
        },
        "panel_scope": {
            "candidate_rows": len(candidate_keys),
            "baseline_rows": len(baseline_keys),
            "baseline_superset_allowed": allow_baseline_superset,
            "interpretation": (
                "complete candidate panel matched against the same baseline protocols; "
                "not a claim about tasks absent from the candidate"
            ),
        },
    }
    return table, summary


def _markdown(table: pd.DataFrame, summary: dict[str, object]) -> str:
    lines = [
        "# Benchmark panel comparison",
        "",
        f"Candidate: `{summary['candidate_model']}`  ",
        f"Baseline: `{summary['baseline_model']}`",
        "",
        (
            f"Wins/losses/ties: **{summary['wins']}/{summary['losses']}/{summary['ties']}** "
            f"over {summary['tasks']} tasks.  "
        ),
        (
            f"Median signed delta: **{summary['median_signed_delta']:+.5f}**, "
            f"task-bootstrap 95% CI {summary['median_task_bootstrap_ci95']}.  "
        ),
        (
            f"Mean signed delta: **{summary['mean_signed_delta']:+.5f}**, task-bootstrap "
            f"95% CI {summary['mean_task_bootstrap_ci95']}."
        ),
        "",
        "The bootstrap resamples tasks; it describes panel sensitivity and is not a",
        "replacement for per-example confidence intervals or training replicates.",
        "",
        "| Task | Metric | Candidate | Baseline | Signed delta | Winner |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in table.to_dict("records"):
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['candidate']:.5f} | "
            f"{row['baseline']:.5f} | {row['signed_delta']:+.5f} | {row['winner']} |"
        )
    return "\n".join(lines) + "\n"


def write_comparison(table: pd.DataFrame, summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "per_task.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "REPORT.md").write_text(_markdown(table, summary))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-baseline-superset", action="store_true")
    args = parser.parse_args(argv)
    table, summary = compare_panel(
        args.candidate,
        args.baseline,
        n_boot=args.n_boot,
        seed=args.seed,
        allow_baseline_superset=args.allow_baseline_superset,
    )
    write_comparison(table, summary, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
