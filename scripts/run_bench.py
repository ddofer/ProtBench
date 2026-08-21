#!/usr/bin/env python
"""Benchmark one model / checkpoint end to end: probes, optional fine-tuning, a readable summary.

    python scripts/run_bench.py -m Synthyra/ESMplusplus_small
    python scripts/run_bench.py -m /path/to/checkpoint --preset no-fast --finetune lora

Runs every task of the preset that this model has not already been scored on
(same probe, same eval split), picking the probe that is fastest for each task
shape, then refreshes `results/bench_results_all.csv` and writes a markdown
summary. Re-running is cheap: finished tasks are skipped, so an interrupted
sweep resumes where it stopped (`--force` re-runs everything).

Probe choice: sklearn's linear probe is fastest for most tasks, but scales
badly in the number of outputs -- OvR multilabel fits one model per label
(EC, 572 labels: 156s vs 22s) and lbfgs over 1000+ classes is ~30x slower than
the torch head (remote_homology: 61s vs 2s). Those get `torch_linear`; the CSV
records which, so rows stay comparable.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parent.parent
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from benchmark_tasks import DEFAULT_TASKS, FAST_TASKS, TASKS, VERY_FAST_TASKS  # noqa: E402
from protein_benchmark_suite import safe_model_name  # noqa: E402

logger = logging.getLogger("run_bench")

PRESETS = {"very-fast": VERY_FAST_TASKS, "fast": FAST_TASKS, "no-fast": DEFAULT_TASKS}
# 1195 and ~6500 classes: lbfgs over that many classes is far slower than the
# torch head at the same accuracy. Every other multiclass task here is <= ~15
# classes, where sklearn wins.
MANY_CLASS_TASKS = {"remote_homology", "cath_eat"}
# Fine-tuning goes through finetune_sequence.py, which is sequence-level only.
FINETUNABLE = {"binary", "multiclass", "regression"}


def choose_probe(task: str) -> str:
    """Return the probe that is fastest for this task's shape."""
    cfg = TASKS[task]
    if cfg.problem_type == "multilabel" or task in MANY_CLASS_TASKS:
        return "torch_linear"
    return "linear"


def pending_tasks(tasks, csv_path: Path, eval_split: str) -> dict[str, list[str]]:
    """Group not-yet-run tasks by probe. A task counts as done only when the CSV
    holds a row for it with the SAME probe and eval split."""
    done: set[tuple[str, str, str]] = set()
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        done = {
            (str(r["Task"]), str(r["Probe"]), str(r["EvalSplit"]))
            for _, r in df.iterrows()
        }
    pending: dict[str, list[str]] = {}
    for task in tasks:
        probe = choose_probe(task)
        if (TASKS[task].name, probe, eval_split) in done:
            continue
        pending.setdefault(probe, []).append(task)
    return pending


def summarize(csv_path: Path, out_path: Path) -> Path:
    """Write one markdown row per task: main metric, value, probe, fit seconds."""
    df = pd.read_csv(csv_path)
    key_by_name = {cfg.name: key for key, cfg in TASKS.items()}
    lines = [
        f"# {df['Model'].iloc[0] if 'Model' in df and len(df) else out_path.stem}",
        "",
        "| task | type | metric | value | probe | split | eval | n | fit s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in df.sort_values("Task").iterrows():
        task = key_by_name.get(str(r["Task"]), str(r["Task"]))
        cfg = TASKS.get(task)
        metric = getattr(cfg, "main_metric", "") if cfg else ""
        if metric not in df.columns or pd.isna(r.get(metric)):
            metric = next(
                (m for m in ("AUC", "Spearman", "F1_Micro", "F1_Macro", "Accuracy", "MCC")
                 if m in df.columns and pd.notna(r.get(m))),
                "",
            )
        value = r.get(metric)
        fit = r.get("ProbeFitSec")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                task,
                getattr(cfg, "problem_type", "") if cfg else "",
                metric,
                "" if metric == "" or pd.isna(value) else round(float(value), 4),
                r.get("Probe", ""),
                r.get("EvalSplit", ""),
                r.get("EvalStrategy", ""),
                r.get("Samples", ""),
                "" if pd.isna(fit) else round(float(fit), 1),
            )
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=BENCH)
    logger.info("  exit=%d in %.1fs", result.returncode, time.perf_counter() - start)
    if result.returncode != 0:
        # A task that dies (missing local dataset, OOM) must not take the sweep
        # with it; the CSV keeps whatever finished and a re-run retries the rest.
        logger.warning("  command failed; continuing")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", "-m", required=True, help="HF id or local checkpoint path")
    p.add_argument("--preset", default="fast", choices=tuple(PRESETS))
    p.add_argument("--tasks", "-t", nargs="+", help="Explicit task keys (overrides --preset)")
    p.add_argument("--eval_split", default="test", choices=("test", "validation"))
    p.add_argument("--finetune", default="none", choices=("none", "lora", "last_n", "full"),
                   help="Also fine-tune on sequence-level tasks (default: probes only).")
    p.add_argument("--output_dir", "-o", default="results/benchmarks")
    p.add_argument("--force", action="store_true", help="Re-run tasks that already have results")
    p.add_argument("--probe_args", nargs=argparse.REMAINDER, default=[],
                   help="Everything after this flag is passed to protein_benchmark_suite.py")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tasks = args.tasks or PRESETS[args.preset]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"Unknown tasks: {unknown}")

    out_dir = Path(args.output_dir)
    csv_path = out_dir / f"bench_{safe_model_name(args.model_name)}.csv"
    groups = (
        {probe: [t for t in tasks if choose_probe(t) == probe] for probe in ("linear", "torch_linear")}
        if args.force
        else pending_tasks(tasks, csv_path, args.eval_split)
    )
    groups = {probe: task_list for probe, task_list in groups.items() if task_list}
    if not groups:
        logger.info("Nothing to run: every task already has %s results in %s", args.eval_split, csv_path)
    for probe, task_list in groups.items():
        logger.info("=== %d task(s) with probe=%s: %s", len(task_list), probe, " ".join(task_list))
        _run([sys.executable, "protein_benchmark_suite.py", "-m", args.model_name,
              "--tasks", *task_list, "-p", probe, "--eval_split", args.eval_split,
              "--cache_embeddings", "-o", str(out_dir), *args.probe_args])

    if args.finetune != "none":
        ft_tasks = [t for t in tasks if TASKS[t].problem_type in FINETUNABLE]
        logger.info("=== fine-tuning (%s) on %d task(s)", args.finetune, len(ft_tasks))
        for task in ft_tasks:
            _run([sys.executable, "finetune_sequence.py", "--model_name", args.model_name,
                  "--task", task, "--mode", args.finetune, "--early_stop", "-o", str(out_dir)])

    # Readable outputs: the long-format CSV across every model, plus a summary
    # of this model alone.
    collect = [sys.executable, "collect_bench_results.py", "--probe-csv", str(csv_path)]
    ft_jsonl = sorted(out_dir.glob(f"finetune_*_{safe_model_name(args.model_name)}.jsonl"))
    if ft_jsonl:
        collect += ["--ft-jsonl", *map(str, ft_jsonl)]
    _run(collect)
    if csv_path.exists():
        summary = summarize(csv_path, out_dir / f"SUMMARY_{safe_model_name(args.model_name)}.md")
        logger.info("Wrote %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
