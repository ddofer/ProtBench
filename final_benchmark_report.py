#!/usr/bin/env python3
"""Clean, readable consolidated Proteva benchmark report.

One-time-clean: dedups the accumulated discriminative results (keep newest per
task x model x metric), reads ProteinGym from its authoritative per-run dirs
(not the polluted aggregate), folds in the new approaches (bf16+truncate indels,
trunk_aux ensemble), and prints ONE readable table per family + a delta figure.

Outputs (to results/):
  final_benchmark.csv    long form (deduped, with source + full_data flag)
  final_benchmark.txt    the printed readable tables
  final_benchmark.png    epoch3 - vanilla per task, grouped by family
"""
import glob
import json
import os
import numpy as np
import pandas as pd

RES = "/data/proteva/plm/results"
B = f"{RES}/bench"
SHORT = {
    "chandar-lab/AMPLIFY_120M": "vanilla",
    "/data/proteva/cache/ckpts/arch_warminit_step0": "step0",
    "/data/proteva/cache/ckpts/stage2_3ep_epoch1_step139754": "epoch1",
    "/data/proteva/cache/ckpts/hf_stage2_final": "epoch3",
}
ORDER = ["vanilla", "step0", "epoch1", "epoch3"]
HIGHER = {"AUC": True, "F1_Macro": True, "Spearman": True, "Recall@10": True, "MSE": False}
# pgym leaderboard-faithful dirs: tag -> model
PG = {"van": "vanilla", "st0": "step0", "ep1": "epoch1", "ep3": "epoch3"}


def _row(rows, fam, task, metric, model, val, full, src):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return
    rows.append(dict(family=fam, task=task, metric=metric, model=model,
                     value=round(float(val), 4), full_data=full, source=src))


def _read_jsonl(pat):
    for f in glob.glob(pat, recursive=True):
        for line in open(f):
            line = line.strip()
            if line:
                yield json.loads(line)


def collect():
    rows = []

    # 1) discriminative probes (linear + lora) from the aggregate CSV, DEDUPED
    #    keep-last (collect appends chronologically -> last == newest run).
    df = pd.read_csv(f"{RES}/bench_results_all.csv")
    df = df[df.model.isin(SHORT) & ~df.probe_type.eq("mlm_zeroshot")]
    df = df[~df.task.str.contains("ProteinGym|proteingym|Zero-Shot", case=False, na=False)]
    df["m"] = df.model.map(SHORT)
    df = df.drop_duplicates(["probe_type", "task", "metric_name", "m"], keep="last")
    for _, r in df.iterrows():
        fam = "discrim (linear)" if r.probe_type == "linear" else "discrim (LoRA)"
        _row(rows, fam, r.task, r.metric_name, r.m, r.metric_value, True, "bench_results_all")

    # 2) ProteinGym substitutions — authoritative leaderboard-faithful dirs
    for tag, m in PG.items():
        for r in _read_jsonl(f"{B}/pgym_{tag}_dmssub/**/*.jsonl"):
            mt = r.get("metric", {})
            _row(rows, "zero-shot (PGym)", "DMS subs (hier)", "Spearman", m, mt.get("eval_spearman"), True, "pgym")
            _row(rows, "zero-shot (PGym)", "DMS subs (hier)", "AUC", m, mt.get("eval_auc"), True, "pgym")
        for r in _read_jsonl(f"{B}/pgym_{tag}_clinsub/**/*.jsonl"):
            mt = r.get("metric", {})
            _row(rows, "zero-shot (PGym)", "Clinical subs", "AUC", m, mt.get("eval_auc"), True, "pgym")

    # 3) improved bf16+truncate indels (100% coverage)
    for tag, m in PG.items():
        d = {"van": "vanilla", "st0": "step0", "ep1": "epoch1", "ep3": "epoch3"}[tag]
        for r in _read_jsonl(f"{B}/idx_bf16_{d}/*.jsonl"):
            if "indel" not in r["task"]:
                continue
            mt = r.get("metric", {})
            nm = "DMS indels" if "dms" in r["task"] else "Clinical indels"
            _row(rows, "zero-shot (PGym)", nm, "AUC", m, mt.get("eval_auc"), True, "idx_bf16")

    # 4) trunk_aux ProteinGym ensemble (full-data dir preferred over di3 cap-250)
    for tag, m in [("ep3", "epoch3"), ("ep1", "epoch1"), ("van", "vanilla"), ("st0", "step0")]:
        full = glob.glob(f"{B}/aux_zs_{tag}_full2/*.jsonl")
        use = f"{B}/aux_zs_{tag}_full2" if full else f"{B}/aux_zs_{tag}_di3"
        is_full = bool(full)
        agg = {}
        for r in _read_jsonl(f"{use}/*.jsonl"):
            agg.setdefault(r.get("score"), []).append((r.get("spearman"), r.get("auc")))
        names = {"mlm_marginal": "aux: MLM-marginal", "cons_weighted_mlm": "aux: +cons",
                 "di3_sad": "aux: di3-SAD", "ensemble": "aux: ensemble"}
        for sc, vals in agg.items():
            sp = np.nanmean([s for s, _ in vals if s is not None]) if vals else np.nan
            au = np.nanmean([a for _, a in vals if a is not None]) if vals else np.nan
            _row(rows, "zero-shot (aux)", names.get(sc, sc), "Spearman", m, sp, is_full, os.path.basename(use))
            _row(rows, "zero-shot (aux)", names.get(sc, sc), "AUC", m, au, is_full, os.path.basename(use))
    return pd.DataFrame(rows)


def pivot(df):
    p = df.pivot_table(index=["family", "task", "metric"], columns="model",
                       values="value", aggfunc="last").reindex(columns=ORDER)
    fd = df.groupby(["family", "task", "metric"])["full_data"].min()

    def d(row):
        v, e, met = row.get("vanilla"), row.get("epoch3"), row.name[2]
        if pd.isna(v) or pd.isna(e):
            return np.nan
        return round((e - v) if HIGHER.get(met, True) else (v - e), 4)
    p["Δ_ep3_van"] = p.apply(d, axis=1)
    p["full"] = fd.reindex(p.index).map({True: "", False: "*capped"})
    return p


def figure(p, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = p.reset_index()
    d = d[d["Δ_ep3_van"].notna() & ~d.family.str.contains("aux")].copy()
    d["label"] = d.task.str.slice(0, 30) + " [" + d.metric + "]"
    d = d.sort_values(["family", "Δ_ep3_van"])
    fams = list(dict.fromkeys(d.family))
    cm = {f: plt.cm.Set2(i) for i, f in enumerate(fams)}
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.3 * len(d))))
    ax.barh(range(len(d)), d["Δ_ep3_van"], color=[cm[f] for f in d.family])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.label, fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("epoch3 − vanilla  (sign-corrected; >0 = stage-2 better)")
    ax.set_title("Proteva stage-2 vs vanilla AMPLIFY-120M")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cm[f], label=f) for f in fams], fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)


def main():
    df = collect()
    p = pivot(df)
    df.to_csv(f"{RES}/final_benchmark.csv", index=False)
    lines = []
    for fam in ["zero-shot (PGym)", "zero-shot (aux)", "discrim (linear)", "discrim (LoRA)"]:
        if fam not in p.index.get_level_values(0):
            continue
        sub = p.xs(fam, level=0)
        cols = [c for c in ORDER if c in sub.columns] + ["Δ_ep3_van", "full"]
        lines.append(f"\n===== {fam} =====")
        lines.append(sub[cols].to_string())
    txt = "\n".join(lines)
    print(txt)
    open(f"{RES}/final_benchmark.txt", "w").write(txt)
    try:
        figure(p, f"{RES}/final_benchmark.png")
        print(f"\n-> {RES}/final_benchmark.csv | .txt | .png")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
