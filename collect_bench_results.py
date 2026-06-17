"""Fold probe CSVs (ResultTracker) + FT JSONLs into one long-format CSV.

Single human-readable results file across models / probes / splits / tasks,
with a free-text `notes` column. Append + dedup (latest wins per key)."""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path

# Robust import: try package-style first, fall back to adding bench dir to sys.path
try:
    from plm.bench.benchmark_tasks import TASKS
except ImportError:
    _BENCH_DIR = str(Path(__file__).resolve().parent)
    if _BENCH_DIR not in sys.path:
        sys.path.insert(0, _BENCH_DIR)
    from benchmark_tasks import TASKS  # type: ignore[import]

OUT_COLUMNS = ["date", "model", "notes", "probe_type", "task", "task_type",
               "split", "metric_name", "metric_value", "n_samples",
               "runtime_s", "source_file"]

def _task_type(task: str) -> str:
    cfg = TASKS.get(task)
    return "residue" if (cfg and getattr(cfg, "problem_type", "") == "token_classification") else "sequence"

def _main_metric(task: str, fallback: str = "AUC") -> str:
    cfg = TASKS.get(task)
    return getattr(cfg, "main_metric", fallback) if cfg else fallback

def normalize_probe_row(row: dict, main_metric: str, task_type: str,
                        notes: str, runtime_s: float | None, source_file: str = "") -> dict:
    val = row.get(main_metric, "")
    if val in ("", None):
        # Probe CSV uses display names (e.g. "Remote Homology (Fold)") but TASKS keys
        # are short ("remote_homology"), so _main_metric may fall back to "AUC" for
        # non-AUC tasks.  Scan known metric columns and pick the first non-empty one.
        for _col in ("AUC", "Spearman", "F1_Macro", "Accuracy", "MCC", "Recall@10", "MSE"):
            _v = row.get(_col, "")
            if _v not in ("", None):
                val = _v; main_metric = _col; break
    return {
        "date": row.get("Date", ""), "model": row.get("Model", ""), "notes": notes,
        "probe_type": (row.get("Probe", "") or "linear").lower(),
        "task": row.get("Task", ""), "task_type": task_type,
        "split": (row.get("EvalSplit", "") or "").lower(),
        "metric_name": main_metric,
        "metric_value": float(val) if val not in ("", None) else None,
        "n_samples": row.get("Samples", ""), "runtime_s": runtime_s,
        "source_file": source_file,
    }

def normalize_ft_record(rec: dict, main_metric: str, task_type: str,
                        runtime_s: float | None, source_file: str = "") -> dict:
    m = rec.get("metric", {}) or {}
    # HF Trainer writes eval_<MainMetric> with the task's EXACT metric casing
    # (e.g. eval_AUC, eval_F1_Macro, eval_Accuracy, eval_spearman) — so match
    # case-insensitively, not just lower(). Build a lower->value index once.
    m_ci = {str(k).lower(): v for k, v in m.items()}
    value = m_ci.get(f"eval_{main_metric}".lower())
    if value is None:
        for k in (f"eval_{main_metric.replace('_Macro','').replace('_macro','')}".lower(),
                  "eval_spearman", "eval_auc", "eval_accuracy",
                  "eval_f1_macro", "eval_matthews_correlation", "eval_mcc"):
            if k in m_ci:
                value = m_ci[k]; break
    return {
        "date": (rec.get("timestamp_iso", "") or "")[:10],
        "model": rec.get("checkpoint", ""), "notes": rec.get("notes", ""),
        "probe_type": rec.get("mode", ""), "task": rec.get("task", ""),
        "task_type": task_type, "split": (rec.get("split", "") or "").lower(),
        "metric_name": main_metric,
        "metric_value": float(value) if value is not None else None,
        "n_samples": rec.get("n_eval", ""), "runtime_s": runtime_s,
        "source_file": source_file,
    }

def _read_existing(out_csv: Path) -> list[dict]:
    if not out_csv.exists():
        return []
    with out_csv.open() as f:
        return list(csv.DictReader(f))

def _dedup_key(r: dict) -> tuple:
    # NOT keyed on date: a corrected re-run (e.g. fixed LoRA lr) must OVERWRITE
    # the stale row, not append a second one on a new day. `notes` IS in the key
    # so two distinct runs on the SAME checkpoint stay separate (e.g. a
    # "verify-...-lora" smoke vs the real run) — notes is set once per launcher,
    # not hand-typed per row, so it is stable enough to be identity here.
    return (r["model"], r.get("notes", ""), r["task"], r["probe_type"], r["split"], r["metric_name"])

def collect(probe_csvs: list[str], ft_jsonls: list[str], out_csv: str,
            notes: str = "", runtime_map: dict | None = None) -> int:
    runtime_map = runtime_map or {}
    rows: list[dict] = []
    for pc in probe_csvs:
        p = Path(pc)
        if not p.exists():
            continue
        with p.open() as f:
            for row in csv.DictReader(f):
                task = row.get("Task", "")
                rt = runtime_map.get(("probe", row.get("EvalSplit", "").lower()))
                main = normalize_probe_row(
                    row, _main_metric(task), _task_type(task), notes, rt, p.name)
                rows.append(main)
                # Regression tasks: also emit MSE as a second row (main metric is
                # Spearman). dedup key includes metric_name, so this never collides.
                if main["metric_name"] != "MSE" and row.get("MSE", "") not in ("", None):
                    rows.append(normalize_probe_row(
                        row, "MSE", _task_type(task), notes, rt, p.name))
    for jl in ft_jsonls:
        p = Path(jl)
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task = rec.get("task", "")
                rt = runtime_map.get((rec.get("mode", ""), rec.get("split", "").lower()))
                rows.append(normalize_ft_record(
                    rec, _main_metric(task), _task_type(task), rt, p.name))
                # Regression MLM tasks also carry eval_auc (median-binarized).
                # Emit as a second row; dedup key includes metric_name → no collision.
                m = rec.get("metric", {}) or {}
                if "eval_auc" in {str(k).lower() for k in m} and _main_metric(task).lower() != "auc":
                    rows.append(normalize_ft_record(rec, "AUC", _task_type(task), rt, p.name))
    out = Path(out_csv)
    merged = {_dedup_key(r): r for r in _read_existing(out)}
    for r in rows:
        merged[_dedup_key(r)] = r
    # Collapse notes-drift: the dedup key includes `notes`, so two runs of the
    # SAME (model, task, probe, split, metric) under edited notes (e.g. a fixed
    # LoRA config, or a re-scored MLM run) would BOTH survive and double the cell.
    # Keep one row per logical cell — latest `date` wins (ties keep the row seen
    # later, i.e. the most recent collect). This is what "overwrite stale" means.
    final: dict[tuple, dict] = {}
    for r in merged.values():
        k = (r["model"], r["task"], r["probe_type"], r["split"], r["metric_name"])
        cur = final.get(k)
        if cur is None or str(r.get("date", "")) >= str(cur.get("date", "")):
            final[k] = r
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for r in final.values():
            w.writerow({k: r.get(k, "") for k in OUT_COLUMNS})
    return len(rows)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-csv", nargs="*", default=[])
    ap.add_argument("--ft-jsonl", nargs="*", default=[])
    ap.add_argument("--out", default="/data/proteva/plm/results/bench_results_all.csv")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)
    n = collect(args.probe_csv, args.ft_jsonl, args.out, notes=args.notes)
    print(f"collected {n} rows -> {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
