"""Shared benchmark utilities for result processing and comparison.

This module centralizes common constants, data-prep functions, and metric
utilities used across:
  - benchmark_comparison.py
  - benchmark_comparison.py
  - collect_bench_results.py
  - protein_benchmark_suite.py

Usage:
    from benchmark_utils import (
        METRIC_PRIORITY,
        RESULT_IDENTITY_COLUMNS,
        prepare_result_df,
        relative_delta_pct,
        find_result_file,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_RESULT_PROBE = "linear"
DEFAULT_RESULT_EVAL_MODE = "standard"
DEFAULT_RESULT_EVAL_SPLIT = "test"
DEFAULT_RESULT_EVAL_STRATEGY = "test_split"

RESULT_IDENTITY_COLUMNS = (
    "Task",
    "Samples",
    "Probe",
    "EvalMode",
    "EvalSplit",
    "EvalStrategy",
    "CodeVersion",
)

# Rows written before the stamp existed. They are NOT interchangeable with
# stamped rows: the residue CV fallback, for one, moved from residue-level KFold
# (which leaked residues of a protein across folds) to protein-level GroupKFold.
UNKNOWN_CODE_VERSION = "unknown"

# Ordered metric priority for selection: first valid metric in both runs is used.
METRIC_PRIORITY = [
    "AUC",
    "F1_Macro",
    "F1_Weighted",
    "F1",
    "AP",
    "Accuracy",
    "Spearman",
    "MSE",
    "Recall@10",
    "Recall@1",
    "Recall@30",
    # Contact prediction. Long-range precision-at-L/5 is the conventional
    # headline; without it a contact row matches nothing here and --compare
    # silently reports no metric for the task.
    "P@L/5_long",
    "F1_Micro",
]

# Task group colors for visualization.
TASK_GROUP_COLORS = {
    "Binary": "#1f77b4",
    "Multiclass": "#ff7f0e",
    "Multilabel": "#2ca02c",
    "Regression": "#d62728",
    "Retrieval": "#9467bd",
    "Residue": "#8c564b",
    "Contact": "#17becf",
    "Other": "#7f7f7f",
}

EPSILON = 1e-12


# ============================================================================
# File Resolution
# ============================================================================


@lru_cache(maxsize=1)
def code_version() -> str:
    """Short git description of the benchmark code that produced a result.

    Two rows can share model, task, probe and split and still be different
    measurements, because the code between them changed -- the residue CV
    fallback moved from residue-level KFold to protein-level GroupKFold, and the
    indel RED arm had its sign corrected. ``Date`` cannot express that; this can.
    ``-dirty`` marks uncommitted changes. Falls back to ``unknown`` outside a git
    checkout, which is also what old, unstamped rows read as.
    """
    import subprocess

    repo = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return UNKNOWN_CODE_VERSION
    if not sha:
        return UNKNOWN_CODE_VERSION
    return f"{sha}-dirty" if dirty else sha


def find_result_file(model_or_dir: str, output_dir: str = "results/benchmarks") -> Path:
    """Find the most recent result CSV file for a model.

    Args:
        model_or_dir:
            - Direct path to CSV file
            - Path to output directory containing CSV files
            - Model name (will search in output_dir for matching CSV)
        output_dir: Where to search for CSVs if model_or_dir is a model name

    Returns:
        Path to the result CSV file

    Raises:
        FileNotFoundError: If no suitable CSV is found
    """
    path = Path(model_or_dir)

    # If it's a direct CSV file, return it
    if path.suffix == ".csv":
        if path.exists():
            return path
        raise FileNotFoundError(f"Result CSV not found: {path}")

    # If it's a directory, find the most recent CSV in it
    if path.is_dir():
        csv_files = sorted(
            path.glob("bench_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if csv_files:
            logger.info(f"Found result file: {csv_files[0]}")
            return csv_files[0]
        raise FileNotFoundError(
            f"No benchmark CSV files found in {path}. Expected pattern: bench_*.csv"
        )

    # If it's a model name, search in output_dir
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(
            f"Output directory not found while searching for model '{model_or_dir}': "
            f"{output_path}"
        )

    # Create a safe version of the model name for matching
    safe_model = model_or_dir.replace("/", "_").replace("\\", "_")

    # Find CSV files matching this model (stable and legacy formats)
    exact_file = output_path / f"bench_{safe_model}.csv"
    legacy_files = list(output_path.glob(f"bench_{safe_model}_*.csv"))

    csv_files = []
    if exact_file.exists():
        csv_files.append(exact_file)
    csv_files.extend(legacy_files)
    csv_files = sorted(csv_files, key=lambda p: p.stat().st_mtime, reverse=True)

    if csv_files:
        logger.info(f"Found result file: {csv_files[0]}")
        return csv_files[0]

    raise FileNotFoundError(
        f"No result CSV found for model '{model_or_dir}'. "
        "Expected one of: "
        f"{output_path}/bench_{safe_model}.csv or "
        f"{output_path}/bench_{safe_model}_*.csv"
    )


# ============================================================================
# Result DataFrame Preparation
# ============================================================================


def prepare_result_df(
    df: pd.DataFrame,
    *,
    extra_defaults: dict[str, str] | None = None,
    extra_dedup_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize result rows and keep the latest row per result identity.

    Handles missing columns, converts types, deduplicates by identity columns,
    and keeps the last (most recent) row for each unique combination.

    Args:
        df: Raw benchmark dataframe
        extra_defaults: Additional column defaults beyond standard ones
        extra_dedup_columns: Additional columns to consider in deduplication
            (e.g., ["experiment", "benchmark_seed"] for ablation reports)

    Returns:
        Normalized dataframe with one row per benchmark identity

    Raises:
        ValueError: If dataframe is empty or missing critical columns
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    prepared = df.copy()

    if "Task" not in prepared.columns:
        prepared = prepared.reset_index().rename(
            columns={prepared.index.name or "index": "Task"}
        )

    # Standard defaults
    defaults = {
        "Samples": "Full",
        "Probe": DEFAULT_RESULT_PROBE,
        "EvalMode": DEFAULT_RESULT_EVAL_MODE,
        "EvalSplit": DEFAULT_RESULT_EVAL_SPLIT,
        "EvalStrategy": DEFAULT_RESULT_EVAL_STRATEGY,
        "Date": "",
        "CodeVersion": UNKNOWN_CODE_VERSION,
    }

    # Merge any extra defaults provided
    if extra_defaults:
        defaults.update(extra_defaults)

    # Apply defaults and handle legacy column names
    for column, default_value in defaults.items():
        if column not in prepared.columns:
            prepared[column] = default_value
        prepared[column] = prepared[column].fillna(default_value)

    # Handle legacy BenchmarkSeed -> benchmark_seed
    if "benchmark_seed" not in prepared.columns and "BenchmarkSeed" in prepared.columns:
        prepared["benchmark_seed"] = prepared["BenchmarkSeed"]

    # Type conversions
    prepared["Samples"] = prepared["Samples"].astype(str)
    prepared["Probe"] = prepared["Probe"].astype(str)
    prepared["EvalMode"] = prepared["EvalMode"].astype(str)
    prepared["EvalSplit"] = prepared["EvalSplit"].astype(str)
    prepared["EvalStrategy"] = prepared["EvalStrategy"].astype(str)
    prepared["Date"] = prepared["Date"].astype(str)

    # Date-based sorting
    prepared["_date_sort"] = pd.to_datetime(prepared["Date"], errors="coerce")
    prepared["_row_order"] = np.arange(len(prepared))
    prepared = prepared.sort_values(["_date_sort", "_row_order"])

    # Deduplication: keep last (most recent) row for each identity
    dedup_cols = list(RESULT_IDENTITY_COLUMNS)
    if extra_dedup_columns:
        dedup_cols.extend(extra_dedup_columns)
    prepared = prepared.drop_duplicates(subset=dedup_cols, keep="last")

    return prepared.drop(columns=["_date_sort", "_row_order"])


# ============================================================================
# Metric Selection and Conversion
# ============================================================================


def get_best_metric_for_task(row: pd.Series) -> Tuple[Optional[str], float]:
    """Determine the best metric value for a task based on METRIC_PRIORITY.

    Returns:
        Tuple of (metric_name, value) where value is normalized such that
        higher is always better. MSE values are negated (since lower is better).
    """
    for metric in METRIC_PRIORITY:
        if metric in row.index and pd.notna(row[metric]):
            val = row[metric]
            # For MSE, lower is better, so return negative value for comparison
            if metric == "MSE":
                return (metric, -val)
            return (metric, val)

    return (None, float("-inf"))


def comparison_value(metric: str, value: float) -> float:
    """Normalize metric direction so larger is always better for comparisons.

    Args:
        metric: Metric name (e.g., "AUC", "MSE")
        value: Metric value

    Returns:
        Normalized value where higher is always better
    """
    return -float(value) if metric == "MSE" else float(value)


def relative_delta_pct(
    metric: str, baseline_value: float, experiment_value: float
) -> float:
    """Compute relative delta percentage where positive means better than baseline.

    For higher-is-better metrics (AUC, F1, etc.):
        delta_pct = 100 * (experiment - baseline) / abs(baseline)

    For lower-is-better metrics (MSE):
        delta_pct = 100 * (baseline - experiment) / abs(baseline)

    Args:
        metric: Metric name
        baseline_value: Baseline metric value
        experiment_value: Experiment/trained metric value

    Returns:
        Relative change as a percentage. Positive means improvement over baseline.
    """
    denom = max(abs(baseline_value), EPSILON)
    if metric == "MSE":
        return 100.0 * (baseline_value - experiment_value) / denom
    return 100.0 * (experiment_value - baseline_value) / denom


def first_common_metric(cols_a: Iterable[str], cols_b: Iterable[str]) -> Optional[str]:
    """Find the first comparable metric present in both sets of columns.

    Uses METRIC_PRIORITY to determine which metric to select.

    Args:
        cols_a: Column names from first dataframe/series
        cols_b: Column names from second dataframe/series

    Returns:
        First metric in METRIC_PRIORITY that exists in both, or None
    """
    cols_a_set = set(cols_a)
    cols_b_set = set(cols_b)
    for metric in METRIC_PRIORITY:
        if metric in cols_a_set and metric in cols_b_set:
            return metric
    return None


def format_result_key(result_key: tuple[str, str, str, str, str, str]) -> str:
    """Render a result identity key for warnings and output.

    Args:
        result_key: Tuple of (Task, Samples, Probe, EvalMode, EvalSplit, EvalStrategy)

    Returns:
        Human-readable key string, e.g., "Task | probe=knn | eval=proteingym"
    """
    task, samples, probe, eval_mode, eval_split, eval_strategy = result_key
    parts = [task]
    if probe != DEFAULT_RESULT_PROBE:
        parts.append(f"probe={probe}")
    if eval_mode != DEFAULT_RESULT_EVAL_MODE:
        parts.append(f"eval={eval_mode}")
    if eval_split != DEFAULT_RESULT_EVAL_SPLIT:
        parts.append(f"split={eval_split}")
    if eval_strategy != DEFAULT_RESULT_EVAL_STRATEGY:
        parts.append(f"strategy={eval_strategy}")
    if samples != "Full":
        parts.append(f"samples={samples}")
    return " | ".join(parts)


# ============================================================================
# Task Grouping
# ============================================================================


def task_group_map(include_non_standard: bool = False) -> dict[str, str]:
    """Build mapping from task display names to coarse task groups.

    Args:
        include_non_standard: If True, include tasks with eval_mode != "standard"

    Returns:
        Dict mapping task name to group (e.g., {"solubility": "Binary", ...})
    """
    from benchmark_tasks import TASKS

    group_name_map = {
        "binary": "Binary",
        "multiclass": "Multiclass",
        "multilabel": "Multilabel",
        "regression": "Regression",
        "retrieval": "Retrieval",
        "token_classification": "Residue",
        "contact_prediction": "Contact",
    }
    return {
        cfg.name: group_name_map.get(cfg.problem_type, "Other")
        for cfg in TASKS.values()
        if include_non_standard or cfg.eval_mode == "standard"
    }
