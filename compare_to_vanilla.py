#!/usr/bin/env python3
"""Delta-vs-vanilla view over the unified benchmark CSV.

``collect_bench_results.py`` writes ``bench_results_all.csv`` (one long row per
model × task × probe × split × metric). This pivots it to ONE row per
(probe_type, task, metric) carrying the vanilla baseline value (the untrained
base model, default AMPLIFY), every other model's value, and each model's delta
vs the baseline — the table you actually read to answer "did post-training help,
and by how much, on linear probe AND LoRA".

The baseline is matched by substring (default ``AMPLIFY``). LoRA rows have no
AMPLIFY baseline (the base model can't FT — no classification head), so their
``baseline``/``deltas`` are ``None`` but the row is still emitted so the
fine-tuned models are visible.

Pure stdlib. Read-only.

  python plm/bench/compare_to_vanilla.py                       # test split, all probes
  python plm/bench/compare_to_vanilla.py --probe lora --split test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_CSV = "results/bench_results_all.csv"


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_comparison(rows, *, baseline_substr="AMPLIFY", split="test"):
    """Pivot long rows -> per-(probe, task, metric) comparison records.

    Each record: ``{probe_type, task, metric, baseline, models:{model->val},
    deltas:{model->val-baseline or None}}``. ``baseline`` is the first model
    whose name contains ``baseline_substr`` (None if absent, e.g. LoRA rows).
    Rows for other splits or with non-numeric values are skipped. Sorted by
    (probe_type, task).
    """
    groups: dict[tuple, dict] = {}
    for r in rows:
        if (r.get("split") or "") != split:
            continue
        val = _to_float(r.get("metric_value"))
        if val is None:
            continue
        key = (r.get("probe_type", ""), r.get("task", ""), r.get("metric_name", ""))
        g = groups.setdefault(key, {})
        g[r.get("model", "")] = val

    out = []
    for (probe, task, metric), model_vals in sorted(groups.items()):
        baseline_model = next(
            (m for m in model_vals if baseline_substr in m), None
        )
        baseline = model_vals.get(baseline_model) if baseline_model else None
        models = {m: v for m, v in model_vals.items() if m != baseline_model}
        deltas = {
            m: (v - baseline if baseline is not None else None)
            for m, v in models.items()
        }
        out.append({
            "probe_type": probe, "task": task, "metric": metric,
            "baseline": baseline, "baseline_model": baseline_model,
            "models": models, "deltas": deltas,
        })
    return out


def _short(model: str) -> str:
    """Compact model label: last path component, AMPLIFY id kept readable."""
    base = model.rstrip("/").split("/")[-1]
    return base.replace("stage2_3ep_", "").replace("_step139754", "")


def format_table(records) -> str:
    if not records:
        return "(no rows)"
    # Stable model column order across all records (baseline first if present).
    model_order: list[str] = []
    for rec in records:
        for m in rec["models"]:
            if m not in model_order:
                model_order.append(m)
    lines = []
    cur_probe = None
    for rec in records:
        if rec["probe_type"] != cur_probe:
            cur_probe = rec["probe_type"]
            lines.append("")
            lines.append(f"### probe = {cur_probe}")
            header = f"{'task':40} {'metric':9} {'vanilla':>8}"
            for m in model_order:
                header += f" {_short(m)[:14]:>14} {'Δ':>8}"
            lines.append(header)
        base = rec["baseline"]
        row = f"{rec['task'][:40]:40} {rec['metric']:9} {('%.4f'%base) if base is not None else '--':>8}"
        for m in model_order:
            v = rec["models"].get(m)
            d = rec["deltas"].get(m)
            row += f" {('%.4f'%v) if v is not None else '--':>14}"
            row += f" {('%+.4f'%d) if d is not None else '--':>8}"
        lines.append(row)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--split", default="test")
    ap.add_argument("--probe", default=None, help="filter to one probe_type (linear/lora)")
    ap.add_argument("--baseline", default="AMPLIFY", help="substring identifying the vanilla baseline model")
    args = ap.parse_args(argv)

    with Path(args.csv).open() as f:
        rows = list(csv.DictReader(f))
    records = build_comparison(rows, baseline_substr=args.baseline, split=args.split)
    if args.probe:
        records = [r for r in records if r["probe_type"] == args.probe]
    print(f"# {args.csv}  (split={args.split}, baseline~={args.baseline!r})")
    print(format_table(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
