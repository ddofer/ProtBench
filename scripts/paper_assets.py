"""Generate lightweight manuscript assets from the live ProtBench registry.

The script intentionally imports only ``benchmark_tasks`` so it can run in a
minimal Python environment without installing the full benchmark stack.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_tasks import (
    DEFAULT_TASKS,
    FAST_TASKS,
    PROTEINGYM_TASKS,
    RETRIEVAL_TASKS,
    TASKS,
    VERY_FAST_TASKS,
)


DISPLAY_TYPE = {
    "binary": "Binary classification",
    "multiclass": "Multiclass classification",
    "multilabel": "Multilabel classification",
    "regression": "Regression",
    "retrieval": "Retrieval",
    "token_classification": "Residue-level token classification",
}

SAFE_RESULT_SOURCES = [
    (
        "ProtBench in-repo baselines",
        "results/benchmarks/hmmer_baseline.json; results/benchmarks/mmseqs_baseline.json",
        "Alignment baseline summaries already tracked in this repository.",
    ),
    (
        "ProtJepa vanilla ESM-2",
        "../protJepa/results/bench_pb/vanilla*/bench_facebook_esm2_*.csv",
        "Use only off-the-shelf ESM-2 rows in the main paper unless trained-model rows are explicitly approved for the supplement.",
    ),
    (
        "ProtSent public benchmark outputs",
        "../ProtSent/results/benchmarks/v3; ../ProtSent/results/benchmarks/v2_150m; ../ProtSent/results/benchmarks/COMPARISON.md",
        "Published/public ProtSent and vanilla comparison artifacts.",
    ),
    (
        "ProtEva vanilla baseline candidates",
        "../proteva/results/soup_pareto_20260816/vanilla_esmc/bench_Synthyra_ESMplusplus_small.csv",
        "Use only clearly vanilla/common-model rows; exclude private continued-pretraining checkpoints.",
    ),
]

EXCLUDED_RESULT_SOURCES = [
    (
        "ProtEva continued-pretraining and soup outputs",
        "../proteva/results/benchmarks; ../proteva/results/soup_pareto_*/",
        "Private in-progress results; do not publish without explicit approval.",
    ),
    (
        "ProtJepa trained-model grids",
        "../protJepa/results/bench_pb/*_jepa*; ../protJepa/results/bench_pb/*_mlm*; ../protJepa/results/bench_pb/*_ladder*",
        "Supplement-only candidates; main text should not depend on them unless the relevant manuscript is public-ready.",
    ),
]

VANILLA_ESM2_FILES = {
    "ESM-2 35M": REPO_ROOT.parent
    / "protJepa/results/bench_pb/vanilla/bench_facebook_esm2_t12_35M_UR50D.csv",
    "ESM-2 150M": REPO_ROOT.parent
    / "protJepa/results/bench_pb/vanilla_150m/bench_facebook_esm2_t30_150M_UR50D.csv",
}

PROTSENT_FILES = {
    "ESM-2 35M": REPO_ROOT.parent
    / "ProtSent/results/benchmarks/v3/esm2_35m_linear/bench__storage_models_ESM2-35M.csv",
    "ProtSent-V1 35M": REPO_ROOT.parent
    / "ProtSent/results/benchmarks/v3/protsent_old_linear/bench_oriel9p_protsent-esm2-35M.csv",
    "ProtSent-V2 35M": REPO_ROOT.parent
    / "ProtSent/results/benchmarks/v3/protsent_v3_linear/bench_models_protsent_esm2_35m_v3_final.csv",
}

VANILLA_ESM2_TASKS = {
    "Remote Homology (Fold)": "Accuracy",
    "Solubility (DeepSol)": "AUC",
    "Signal Peptide Prediction (SignalP/ProteinBERT)": "AUC",
    "Neuropeptide Precursor Prediction (ProFET/NeuroPID)": "AUC",
    "beta-lactamase-PEER": "Spearman",
    "Metal Ion Binding": "AUC",
    "Variant Effect (GB1)": "Spearman",
    "Fluorescence (TAPE)": "Spearman",
    "Stability (Biomap)": "Spearman",
    "SCOPe-40 Structural Retrieval": "Recall@10",
}

PROTSENT_TASKS = {
    "Remote Homology (Fold)": "Accuracy",
    "Solubility (DeepSol)": "AUC",
    "Signal Peptide Prediction (SignalP/ProteinBERT)": "AUC",
    "Metal Ion Binding": "AUC",
    "Variant Effect (GB1)": "Spearman",
    "Fluorescence (TAPE)": "Spearman",
    "Stability (Biomap)": "Spearman",
}


def preset_for_task(task_key: str) -> str:
    if task_key in VERY_FAST_TASKS:
        return "very-fast"
    if task_key in FAST_TASKS or task_key in RETRIEVAL_TASKS:
        return "fast"
    if task_key in PROTEINGYM_TASKS:
        return "proteingym"
    if task_key in DEFAULT_TASKS:
        return "no-fast/default"
    return "explicit-only"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def task_coverage_markdown() -> str:
    type_counts = Counter(cfg.problem_type for cfg in TASKS.values())
    rows = []
    for problem_type in sorted(type_counts, key=lambda key: DISPLAY_TYPE.get(key, key)):
        metrics = sorted({cfg.main_metric for cfg in TASKS.values() if cfg.problem_type == problem_type})
        rows.append(
            [
                DISPLAY_TYPE.get(problem_type, problem_type),
                str(type_counts[problem_type]),
                ", ".join(metrics),
            ]
        )

    preset_rows = [
        ["All registered tasks", str(len(TASKS)), "Full registry, including opt-in tasks."],
        ["Standard probe tasks", str(sum(1 for cfg in TASKS.values() if cfg.eval_mode == "standard")), "Non-ProteinGym tasks."],
        ["ProteinGym tasks", str(len(PROTEINGYM_TASKS)), "Four supervised and four zero-shot variant-effect tasks."],
        ["--very-fast", str(len(VERY_FAST_TASKS)), "Curated scout subset."],
        ["--fast", str(len(FAST_TASKS) + len(RETRIEVAL_TASKS)), "Default sweep: FAST_TASKS plus retrieval tasks."],
        ["--no-fast/default", str(len(DEFAULT_TASKS)), "Broad non-ProteinGym sweep, excluding retrieval and very large multilabel tasks."],
    ]

    return "\n".join(
        [
            "# ProtBench Task Coverage",
            "",
            "Generated from `benchmark_tasks.py`.",
            "",
            markdown_table(["Problem type", "Tasks", "Main metric(s)"], rows),
            "",
            markdown_table(["Preset/scope", "Tasks", "Notes"], preset_rows),
        ]
    )


def task_inventory_tsv() -> str:
    lines = ["task_key\tdisplay_name\tproblem_type\tmain_metric\tpreset\tdataset"]
    for task_key in sorted(TASKS):
        cfg = TASKS[task_key]
        lines.append(
            "\t".join(
                [
                    task_key,
                    cfg.name,
                    cfg.problem_type,
                    cfg.main_metric,
                    preset_for_task(task_key),
                    cfg.dataset,
                ]
            )
        )
    return "\n".join(lines) + "\n"


def result_source_manifest() -> str:
    def rows(entries: list[tuple[str, str, str]]) -> list[list[str]]:
        return [[name, path, note] for name, path, note in entries]

    return "\n".join(
        [
            "# Paper Result Source Manifest",
            "",
            "This manifest separates result artifacts that may feed the ProtBench paper from artifacts that require explicit approval before use.",
            "",
            "## Safe candidates",
            "",
            markdown_table(["Source", "Path pattern", "Use"], rows(SAFE_RESULT_SOURCES)),
            "",
            "## Excluded unless explicitly approved",
            "",
            markdown_table(["Source", "Path pattern", "Reason"], rows(EXCLUDED_RESULT_SOURCES)),
        ]
    )


def read_result_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("EvalSplit") == "test" and row.get("Probe") == "linear"
        ]
    return {row["Task"]: row for row in rows}


def format_metric(value: str) -> str:
    if not value:
        return ""
    return f"{float(value):.3f}"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT.parent))
    except ValueError:
        return str(path)


def results_table(title: str, files: dict[str, Path], tasks: dict[str, str]) -> str:
    missing = [path for path in files.values() if not path.exists()]
    if missing:
        missing_lines = "\n".join(f"- `{path}`" for path in missing)
        return "\n".join(
            [
                f"## {title}",
                "",
                "Not generated because these local result files were not found:",
                "",
                missing_lines,
                "",
            ]
        )

    rows_by_model = {model: read_result_rows(path) for model, path in files.items()}
    rows = []
    for task, metric in tasks.items():
        values = [
            format_metric(rows_by_model[model].get(task, {}).get(metric, ""))
            for model in files
        ]
        rows.append([task, metric, *values])
    source_lines = [f"- {model}: `../{display_path(path)}`" for model, path in files.items()]
    return "\n".join(
        [
            f"## {title}",
            "",
            "Rows are linear-probe test-split results. Empty cells mean the metric was not present in the source row.",
            "",
            markdown_table(["Task", "Metric", *files.keys()], rows),
            "",
            "Sources:",
            "",
            "\n".join(source_lines),
            "",
        ]
    )


def representative_results_markdown() -> str:
    return "\n".join(
        [
            "# Representative Public Result Tables",
            "",
            "Generated from local sibling-project result artifacts. These tables intentionally exclude private ProtEva continued-pretraining and model-soup outputs.",
            "",
            results_table(
                "Vanilla ESM-2 Scale Slice",
                VANILLA_ESM2_FILES,
                VANILLA_ESM2_TASKS,
            ),
            results_table(
                "Published ProtSent 35M Slice",
                PROTSENT_FILES,
                PROTSENT_TASKS,
            ),
        ]
    )


def write_assets(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "task_coverage.md": task_coverage_markdown(),
        "task_inventory.tsv": task_inventory_tsv(),
        "result_source_manifest.md": result_source_manifest(),
        "representative_results.md": representative_results_markdown(),
    }
    paths = []
    for filename, content in outputs.items():
        path = out_dir / filename
        path.write_text(content.rstrip() + "\n")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="paper/generated", type=Path)
    args = parser.parse_args()
    paths = write_assets(args.out_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())