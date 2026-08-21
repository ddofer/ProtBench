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

from benchmark_tasks import (  # noqa: E402
    DEFAULT_TASKS,
    FAST_MAX_SAMPLES,
    FAST_TASKS,
    RETRIEVAL_TASKS,
    TASK_NAME_TO_KEY,
    TASKS,
    VERY_FAST_TASKS,
)
from protein_benchmark_suite import effective_probe_type, safe_model_name  # noqa: E402

logger = logging.getLogger("run_bench")

# Mirrors protein_benchmark_suite.main's own preset resolution exactly -- used
# only to decide what is still pending; the suite is told the preset by FLAG
# (see preset_flag) so it resolves the list itself and no drift is possible.
PRESETS = {
    "very-fast": list(VERY_FAST_TASKS),
    "fast": list(FAST_TASKS) + list(RETRIEVAL_TASKS),
    "no-fast": list(DEFAULT_TASKS),
}
_PRESET_FLAGS = {"very-fast": "--very-fast", "fast": "--fast", "no-fast": "--no-fast"}
# Fine-tuning goes through finetune_sequence.py, which is sequence-level only.
FINETUNABLE = {"binary", "multiclass", "regression"}


def preset_flag(preset: str) -> str:
    """The suite's own flag for this preset. Passing the flag (rather than an
    expanded --tasks list) is what keeps the two in step -- including the sample
    cap, which the suite applies only when it resolves --very-fast itself."""
    return _PRESET_FLAGS[preset]


def preset_max_samples(preset: str) -> int | None:
    """Sequence cap this preset implies, or None for full data.

    Only --very-fast subsamples. Capped rows are scout results: they are recorded
    in the CSV's Samples column, they do not satisfy a full run (see
    pending_tasks), and mixing them with full rows would invalidate cross-model
    comparison.
    """
    return FAST_MAX_SAMPLES if preset == "very-fast" else None


def _samples_key(samples) -> str:
    """Normalise the CSV's Samples cell: uncapped runs write 'Full', capped ones an int."""
    if samples is None or pd.isna(samples) or str(samples) in ("Full", "", "nan"):
        return "Full"
    return str(int(float(samples)))


def choose_probe(task: str) -> str:
    """Return the probe that is fastest for this task's shape.

    Thin delegate: the rule lives in the suite's ``effective_probe_type`` so
    ``-p auto`` on the plain CLI routes identically. Kept as a named function
    because the runner groups tasks by probe before spawning.
    """
    return effective_probe_type(TASKS[task], "auto")


def pending_tasks(
    tasks, csv_path: Path, eval_split: str, max_samples: int | None = None
) -> dict[str, list[str]]:
    """Group not-yet-run tasks by probe.

    A task counts as done only when the CSV holds a row for it with the same
    probe, eval split AND sample count -- and only when that row is a real
    result. The suite writes a row for a FAILED task too
    (``EvalStrategy='task_exception'``); counting those as done would make a
    transient failure (a missing local dataset, an OOM) permanent until
    ``--force``. Sample count is part of the identity because a capped scout run
    (``--very-fast``) must never make a full sweep look complete: mixing
    subsampled and full rows in one CSV invalidates cross-model comparison.
    """
    done: set[tuple[str, str, str, str]] = set()
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            if str(r.get("EvalStrategy", "")) == "task_exception":
                continue
            done.add((str(r["Task"]), str(r["Probe"]), str(r["EvalSplit"]),
                      _samples_key(r.get("Samples"))))
    want_samples = _samples_key(max_samples)
    pending: dict[str, list[str]] = {}
    for task in tasks:
        probe = choose_probe(task)
        if (TASKS[task].name, probe, eval_split, want_samples) in done:
            continue
        pending.setdefault(probe, []).append(task)
    return pending


def summarize(csv_path: Path, out_path: Path) -> Path:
    """Write one markdown row per task: main metric, value, probe, fit seconds.

    Metric choice and markdown rendering both come from the shared helpers, so
    this table headlines the same metric as ``bench_results_all.csv`` and
    ``COMPARISON.md`` for a given row instead of a fourth private priority list.
    """
    from benchmark_comparison import _to_markdown_table
    from benchmark_utils import get_best_metric_for_task

    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.sort_values("Task").iterrows():
        task = TASK_NAME_TO_KEY.get(str(r["Task"]), str(r["Task"]))
        cfg = TASKS.get(task)
        metric, _ = get_best_metric_for_task(r)
        fit = r.get("ProbeFitSec")
        rows.append({
            "task": task,
            "type": getattr(cfg, "problem_type", "") if cfg else "",
            "metric": metric or "",
            "value": "" if metric is None else round(float(r[metric]), 4),
            "probe": r.get("Probe", ""),
            "split": r.get("EvalSplit", ""),
            "eval": r.get("EvalStrategy", ""),
            "n": r.get("Samples", ""),
            "fit s": "" if pd.isna(fit) else round(float(fit), 1),
        })

    model = df["Model"].iloc[0] if "Model" in df and len(df) else out_path.stem
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"# {model}\n\n" + _to_markdown_table(pd.DataFrame(rows)))
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

    explicit_tasks = bool(args.tasks)
    tasks = args.tasks or PRESETS[args.preset]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"Unknown tasks: {unknown}")

    max_samples = None if explicit_tasks else preset_max_samples(args.preset)
    if max_samples is not None:
        logger.warning(
            "--preset %s SUBSAMPLES to %d sequences per task (residue tasks tighter still). "
            "These are scout rows: recorded as Samples=%d, and NOT comparable with full-data "
            "results for another model.",
            args.preset, max_samples, max_samples,
        )

    out_dir = Path(args.output_dir)
    csv_path = out_dir / f"bench_{safe_model_name(args.model_name)}.csv"
    groups = (
        {probe: [t for t in tasks if choose_probe(t) == probe] for probe in ("linear", "torch_linear")}
        if args.force
        else pending_tasks(tasks, csv_path, args.eval_split, max_samples=max_samples)
    )
    groups = {probe: task_list for probe, task_list in groups.items() if task_list}
    if not groups:
        logger.info("Nothing to run: every task already has %s results in %s", args.eval_split, csv_path)
    for probe, task_list in groups.items():
        logger.info("=== %d task(s) with probe=%s: %s", len(task_list), probe, " ".join(task_list))
        # A preset is passed to the suite as its own FLAG whenever the whole
        # preset is still pending, so the suite resolves the task list and the
        # sample cap itself -- expanding --tasks here silently dropped the cap.
        # A partial resume names the remaining tasks and re-adds the cap by hand.
        whole_preset = not explicit_tasks and len(task_list) == len(
            [t for t in tasks if choose_probe(t) == probe]
        )
        if whole_preset and len(groups) == 1:
            selector = [preset_flag(args.preset)]
        else:
            selector = ["--tasks", *task_list]
            if max_samples is not None:
                selector += ["--max_samples", str(max_samples)]
        _run([sys.executable, "protein_benchmark_suite.py", "-m", args.model_name,
              *selector, "-p", probe, "--eval_split", args.eval_split,
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
