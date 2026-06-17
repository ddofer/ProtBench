#!/usr/bin/env python3
"""Humanized benchmark report over the unified CSV — one command, three views.

Reads ``results/bench_results_all.csv`` (long format from
``collect_bench_results.py``) and writes, into ``results/``:

  1. ``BENCH_REPORT.md``  — GitHub-markdown report a human reads: per-probe
     trajectory tables (vanilla → step0 → epoch1 → epoch3) with ↑/↓ arrows, plus
     an INSIGHTS section (per-model win/loss vs vanilla, top lifts, regressions).
  2. ``bench_pivot.csv``  — wide table (one row per task×metric×probe, one column
     per model) for interactive DS work: ``pandas.read_csv(...)`` and slice.

Reuses ``compare_to_vanilla.build_comparison`` so the baseline/delta logic lives
in exactly one place. Pure stdlib. Read-only on inputs.

  python plm/bench/report.py                      # all probes, test split
  python plm/bench/report.py --split test --probe linear
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from compare_to_vanilla import build_comparison, _short, DEFAULT_CSV

RESULTS_DIR = Path(DEFAULT_CSV).parent
# Trajectory order we want columns to appear in (substring match, first wins).
TRAJECTORY = ["AMPLIFY", "step0", "warminit", "epoch1", "epoch3", "final"]
# "Higher is better" for every metric we emit EXCEPT MSE.
LOWER_IS_BETTER = {"MSE"}


def _ordered_models(records):
    """Models in trajectory order (known stages first), then any extras."""
    seen = []
    for rec in records:
        for m in rec["models"]:
            if m not in seen:
                seen.append(m)
    def rank(m):
        for i, key in enumerate(TRAJECTORY):
            if key.lower() in m.lower():
                return i
        return len(TRAJECTORY)
    return sorted(seen, key=rank)


def _arrow(delta, metric):
    if delta is None:
        return ""
    better = (delta < 0) if metric in LOWER_IS_BETTER else (delta > 0)
    if abs(delta) < 1e-4:
        return "·"
    return "↑" if better else "↓"


def _md_table(records, models):
    """One markdown table for a set of records sharing a probe_type."""
    head = ["task", "metric", "vanilla"]
    for m in models:
        head += [_short(m)[:16], "Δ"]
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for rec in records:
        base = rec["baseline"]
        cells = [rec["task"][:46], rec["metric"],
                 f"{base:.4f}" if base is not None else "—"]
        for m in models:
            v, d = rec["models"].get(m), rec["deltas"].get(m)
            cells.append(f"{v:.4f}" if v is not None else "—")
            cells.append(
                (f"{d:+.4f} {_arrow(d, rec['metric'])}".strip())
                if d is not None else "—")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _insights(records, models):
    """Per-model win/loss vs vanilla + top lifts and regressions."""
    lines = ["## Insights (Δ vs vanilla AMPLIFY)\n"]
    # Per-model win/loss tally over tasks that HAVE a baseline.
    for m in models:
        lifts = []
        for rec in records:
            d = rec["deltas"].get(m)
            if d is None:
                continue
            signed = -d if rec["metric"] in LOWER_IS_BETTER else d
            lifts.append((signed, rec["task"], rec["metric"]))
        if not lifts:
            continue
        wins = sum(1 for s, *_ in lifts if s > 1e-4)
        losses = sum(1 for s, *_ in lifts if s < -1e-4)
        mean = sum(s for s, *_ in lifts) / len(lifts)
        lines.append(
            f"- **{_short(m)}**: {wins} better / {losses} worse / "
            f"{len(lifts)} tasks · mean Δ={mean:+.4f}")
    # Global top-5 lifts and regressions (best model per task).
    best = []
    for rec in records:
        for m in models:
            d = rec["deltas"].get(m)
            if d is None:
                continue
            signed = -d if rec["metric"] in LOWER_IS_BETTER else d
            best.append((signed, rec["task"], rec["metric"], _short(m)))
    best.sort(reverse=True)
    if best:
        lines.append("\n**Top lifts:**")
        for s, task, metric, m in best[:5]:
            lines.append(f"- {task} ({metric}): {s:+.4f} — {m}")
        lines.append("\n**Top regressions:**")
        for s, task, metric, m in best[-5:][::-1]:
            if s < -1e-4:
                lines.append(f"- {task} ({metric}): {s:+.4f} — {m}")
    return "\n".join(lines)


def write_pivot_csv(records, models, path: Path):
    cols = ["probe_type", "task", "metric", "vanilla"] + [_short(m) for m in models]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rec in records:
            row = [rec["probe_type"], rec["task"], rec["metric"],
                   rec["baseline"] if rec["baseline"] is not None else ""]
            for m in models:
                v = rec["models"].get(m)
                row.append(v if v is not None else "")
            w.writerow(row)


# Default views, in reading order: each is one (probe, split) the report covers.
# linear+lora live on `test`; MLM zero-shot lives on its own `zeroshot` split.
DEFAULT_VIEWS = [("linear", "test"), ("lora", "test"), ("mlm_zeroshot", "zeroshot")]


def build_report(csv_path: str, baseline: str, views=None) -> str:
    """One report covering several (probe, split) views, plus a combined pivot CSV."""
    views = views or DEFAULT_VIEWS
    with Path(csv_path).open() as f:
        rows = list(csv.DictReader(f))

    md = ["# Proteva downstream benchmark report",
          f"\n_Source: `{csv_path}` · baseline~=`{baseline}` · "
          f"↑ = better than vanilla, ↓ = worse (MSE: lower is better, sign-corrected)._\n"]
    all_records = []
    for probe, split in views:
        recs = [r for r in build_comparison(rows, baseline_substr=baseline, split=split)
                if r["probe_type"] == probe]
        if not recs:
            md.append(f"\n## {probe} ({split}) — _no rows yet_\n")
            continue
        models = _ordered_models(recs)
        md.append(f"\n## {probe} ({split} split)\n")
        md.append(_md_table(recs, models))
        md.append("\n" + _insights(recs, models))
        all_records.extend(recs)

    # Side artifact: ONE wide pivot CSV across all views for interactive DS use.
    write_pivot_csv(all_records, _ordered_models(all_records),
                    RESULTS_DIR / "bench_pivot.csv")
    return "\n".join(md)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--probe", default=None, help="restrict to one probe_type")
    ap.add_argument("--split", default=None, help="restrict to one split")
    ap.add_argument("--baseline", default="AMPLIFY")
    ap.add_argument("--out", default=str(RESULTS_DIR / "BENCH_REPORT.md"))
    args = ap.parse_args(argv)

    views = DEFAULT_VIEWS
    if args.probe or args.split:
        views = [(args.probe or p, args.split or s) for p, s in DEFAULT_VIEWS
                 if (not args.probe or p == args.probe)]
    report = build_report(args.csv, args.baseline, views)
    Path(args.out).write_text(report + "\n")
    print(report)
    print(f"\n-> wrote {args.out}")
    print(f"-> wrote {RESULTS_DIR / 'bench_pivot.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
