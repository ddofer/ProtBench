"""
Benchmark comparison utilities.

Compare two benchmark runs and show which model is better for each task,
with performance deltas and winner breakdown.

Usage:
    from benchmark_comparison import compare_benchmarks, display_comparison

    df = compare_benchmarks(
        "results/model1_bench",
        "results/model2_bench",
        output_dir="results/benchmarks"
    )
    display_comparison(df)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_utils import (
    DEFAULT_RESULT_EVAL_MODE,
    DEFAULT_RESULT_EVAL_SPLIT,
    DEFAULT_RESULT_EVAL_STRATEGY,
    DEFAULT_RESULT_PROBE,
    RESULT_IDENTITY_COLUMNS,
    comparison_value,
    find_result_file,
    first_common_metric,
    format_result_key,
    prepare_result_df,
)

logger = logging.getLogger(__name__)


def compare_benchmarks(
    model1_or_dir: str,
    model2_or_dir: str,
    output_dir: str = "results/benchmarks",
    sort_by_task: bool = True,
    round_digits: int = 5,
) -> pd.DataFrame:
    """
    Compare two benchmark runs and show which model is better for each task.

    Args:
        model1_or_dir: First model name, directory, or CSV file path
        model2_or_dir: Second model name, directory, or CSV file path
        output_dir: Default directory to search for result CSVs
        sort_by_task: Whether to sort by task name
        round_digits: Number of decimal places to round to

    Returns:
        DataFrame with comparison results (task, best model, metrics, delta)
    """
    # Find and load result files
    file1 = find_result_file(model1_or_dir, output_dir)
    file2 = find_result_file(model2_or_dir, output_dir)

    logger.info(f"Loading results from:\n  {file1}\n  {file2}")

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df1 = prepare_result_df(df1)
    df2 = prepare_result_df(df2)

    # Extract model names from filename if not in dataframe
    model1_name = df1["Model"].iloc[0] if "Model" in df1.columns else file1.stem
    model2_name = df2["Model"].iloc[0] if "Model" in df2.columns else file2.stem

    # Simplify model names for display
    def simplify_name(name: str) -> str:
        # Remove full paths and file prefixes
        name = str(name).replace("/", "_")
        # Try to extract meaningful part
        if "models_" in name:
            return name.split("models_")[-1]
        if "checkpoint" in name:
            return name.split("/")[-1] if "/" in str(name) else name
        return name

    model1_display = simplify_name(model1_name)
    model2_display = simplify_name(model2_name)

    logger.info(f"Comparing: {model1_name} vs {model2_name}")

    # Get comparable row identities and build comparison.
    row_map1 = {
        tuple(row[column] for column in RESULT_IDENTITY_COLUMNS): row
        for _, row in df1.iterrows()
    }
    row_map2 = {
        tuple(row[column] for column in RESULT_IDENTITY_COLUMNS): row
        for _, row in df2.iterrows()
    }
    result_keys = sorted(set(row_map1) | set(row_map2))
    comparison_rows = []
    missing_in_model1: list[str] = []
    missing_in_model2: list[str] = []
    no_common_metric: list[str] = []

    include_samples = any(key[1] != "Full" for key in result_keys)
    include_probe = any(key[2] != DEFAULT_RESULT_PROBE for key in result_keys)
    include_eval_mode = any(key[3] != DEFAULT_RESULT_EVAL_MODE for key in result_keys)
    include_eval_split = any(key[4] != DEFAULT_RESULT_EVAL_SPLIT for key in result_keys)
    include_eval_strategy = any(
        key[5] != DEFAULT_RESULT_EVAL_STRATEGY for key in result_keys
    )

    for result_key in result_keys:
        row1 = row_map1.get(result_key)
        if row1 is None:
            missing_in_model1.append(format_result_key(result_key))
            continue

        row2 = row_map2.get(result_key)
        if row2 is None:
            missing_in_model2.append(format_result_key(result_key))
            continue

        metric_name = first_common_metric(row1.index, row2.index)
        if metric_name is None:
            no_common_metric.append(format_result_key(result_key))
            continue

        actual_metric1_val = float(row1[metric_name])
        actual_metric2_val = float(row2[metric_name])
        metric1_val = comparison_value(metric_name, actual_metric1_val)
        metric2_val = comparison_value(metric_name, actual_metric2_val)

        # Guard against non-finite metric values
        if not np.isfinite(metric1_val) or not np.isfinite(metric2_val):
            logger.warning(
                "Non-finite metric values for %s (%s: %.4g vs %.4g) - skipping",
                format_result_key(result_key),
                metric_name,
                metric1_val,
                metric2_val,
            )
            continue

        # Determine winner
        if metric1_val > metric2_val:
            best_model = model1_display
            best_val = actual_metric1_val
            other_val = actual_metric2_val
            delta_val = actual_metric1_val - actual_metric2_val
        elif metric2_val > metric1_val:
            best_model = model2_display
            best_val = actual_metric2_val
            other_val = actual_metric1_val
            delta_val = actual_metric2_val - actual_metric1_val
        else:
            best_model = "tie"
            best_val = actual_metric1_val
            other_val = actual_metric2_val
            delta_val = 0.0

        # Build comparison row
        comp_row = {
            "Task": result_key[0],
            "Winner": best_model,
            "Metric": metric_name,
            f"Best_{metric_name}": round(float(best_val), round_digits),
            f"Other_{metric_name}": round(float(other_val), round_digits),
            f"Δ_{metric_name}": round(float(delta_val), round_digits),
        }
        if include_samples:
            comp_row["Samples"] = result_key[1]
        if include_probe:
            comp_row["Probe"] = result_key[2]
        if include_eval_mode:
            comp_row["EvalMode"] = result_key[3]
        if include_eval_split:
            comp_row["EvalSplit"] = result_key[4]
        if include_eval_strategy:
            comp_row["EvalStrategy"] = result_key[5]

        comparison_rows.append(comp_row)

    if missing_in_model1:
        logger.warning(
            "Rows present only in %s: %s",
            file2.name,
            "; ".join(missing_in_model1),
        )
    if missing_in_model2:
        logger.warning(
            "Rows present only in %s: %s",
            file1.name,
            "; ".join(missing_in_model2),
        )
    if no_common_metric:
        logger.warning(
            "Rows skipped because no comparable metric existed: %s",
            "; ".join(no_common_metric),
        )

    result_df = pd.DataFrame(comparison_rows)

    if sort_by_task and not result_df.empty and "Task" in result_df.columns:
        result_df = result_df.sort_values("Task").reset_index(drop=True)

    return result_df


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to a simple Markdown table."""
    if df is None or df.empty:
        return "| Result |\n| --- |\n| No comparison data available. |\n"

    columns = [str(col) for col in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            cell = "" if pd.isna(value) else str(value)
            values.append(cell.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows]) + "\n"


def save_comparison_output(comparison_df: pd.DataFrame, output_file: str) -> Path:
    """Save comparison results to CSV or Markdown based on file extension."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        comparison_df.to_csv(output_path, index=False)
    elif suffix == ".md":
        output_path.write_text(_to_markdown_table(comparison_df), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported output format: {output_path.suffix}. Use .csv or .md"
        )

    return output_path


def display_comparison(comparison_df: pd.DataFrame):
    """Display comparison results in a readable format."""
    print("\n" + "=" * 130)
    print("BENCHMARK COMPARISON - Task Winners (compact views)")
    print("=" * 130)

    # Split into regression (Spearman, MSE) and other metrics
    reg_metrics = {"Spearman", "MSE"}
    if comparison_df is None or comparison_df.empty:
        print("No comparison data available.")
        return

    reg_df = comparison_df[comparison_df["Metric"].isin(reg_metrics)].copy()
    other_df = comparison_df[~comparison_df["Metric"].isin(reg_metrics)].copy()

    # Drop columns that are entirely NA to make tables compact
    if not reg_df.empty:
        reg_df = reg_df.dropna(axis=1, how="all")
    if not other_df.empty:
        other_df = other_df.dropna(axis=1, how="all")

    # Pretty print both tables with pandas display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    if not other_df.empty:
        print("\n-- Classification / Other Metrics --")
        print(other_df.to_string(index=False))
    else:
        print("\n-- Classification / Other Metrics --\n  (no rows)\n")

    print("\n" + "-" * 130 + "\n")

    if not reg_df.empty:
        print("-- Regression Metrics (Spearman / MSE) --")
        print(reg_df.to_string(index=False))
    else:
        print("-- Regression Metrics (Spearman / MSE) --\n  (no rows)\n")

    print("=" * 130)

    # Summary statistics
    print("\nSUMMARY:")
    winners = comparison_df["Winner"].value_counts()
    for winner, count in winners.items():
        pct = 100 * count / len(comparison_df)
        print(f"  {winner:30s}: {count:2d} tasks ({pct:5.1f}%)")

    # Reset pandas display options
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.max_colwidth")


def main() -> None:
    """CLI entrypoint for benchmark comparison."""
    parser = argparse.ArgumentParser(description="Compare two benchmark result runs.")
    parser.add_argument(
        "--model1_dir", required=True, help="First model name, directory, or CSV path"
    )
    parser.add_argument(
        "--model2_dir", required=True, help="Second model name, directory, or CSV path"
    )
    parser.add_argument(
        "--output_dir",
        default="results/benchmarks",
        help="Directory used when model names are provided",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Optional output file (.csv or .md) for saving the comparison",
    )
    parser.add_argument(
        "--round_digits",
        type=int,
        default=5,
        help="Number of decimal places to round to",
    )

    args = parser.parse_args()

    try:
        comparison_df = compare_benchmarks(
            args.model1_dir,
            args.model2_dir,
            output_dir=args.output_dir,
            round_digits=args.round_digits,
        )
        display_comparison(comparison_df)

        if args.output_file:
            saved_path = save_comparison_output(comparison_df, args.output_file)
            print(f"\nSaved comparison to: {saved_path}")

    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(2)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
