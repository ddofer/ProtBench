"""Smoke: WT test-time training on real AMPLIFY_120M + real ProteinGym BLAT.

Runs the zero-shot embedding-cosine variant-effect readout on the β-lactamase
(TEM-1, `BLAT_ECOLX_Firnberg_2014`) assay — a single wild-type with many point
mutants — **with and without** WT test-time training, and prints both Spearman
correlations. This exercises the full real-model path that the unit tests
approximate with a stub (resolve_mlm_head -> adapt_to_wt -> embedding readout).

CPU only by default (per the GPU-safety rule). Honest expectation per the design
doc: a small/zero delta on one easy-MSA assay with few steps — this confirms the
path runs and is wired, not that it beats HEAD.

    CUDA_VISIBLE_DEVICES="" python contrib/proteva/wt_tta_smoke.py
"""

from __future__ import annotations

import argparse
import glob
import os
import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from protein_benchmark_suite import embed_sequences, load_model
from wt_test_time_training import (
    TTAConfig,
    resolve_mlm_head,
    run_tta_zeroshot,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("wt_tta_smoke")

# Reads the ProteinGym parquet straight out of the HF cache rather than
# re-downloading. $HF_HOME wins if set, else the default cache location.
_HF_HUB = os.path.join(
    os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"
)
_PG_GLOB = os.path.join(
    _HF_HUB,
    "datasets--OATML-Markslab--ProteinGym_v1",
    "snapshots", "*", "DMS_substitutions", "train-*.parquet",
)


def load_blat(n_variants: int, seed: int, assay: str = "BLAT_ECOLX_Firnberg_2014"):
    """Return (wt_seq, mutants, scores) for one β-lactamase assay from the cache."""
    import pyarrow.parquet as pq

    frames = []
    for sp in sorted(glob.glob(_PG_GLOB)):
        ids = pq.read_table(sp, columns=["DMS_id"]).to_pandas()["DMS_id"].astype(str)
        if (ids == assay).any():
            frames.append(pq.read_table(sp).to_pandas())
    if not frames:
        raise SystemExit(f"Assay {assay} not found in cached ProteinGym_v1.")
    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    df = df[df["DMS_id"].astype(str) == assay].reset_index(drop=True)
    wt = str(df["target_seq"].iloc[0])
    rng = np.random.default_rng(seed)
    take = rng.choice(len(df), size=min(n_variants, len(df)), replace=False)
    sub = df.iloc[np.sort(take)]
    mutants = [str(s) for s in sub["mutated_sequence"].tolist()]
    scores = sub["DMS_score"].to_numpy(dtype=float)
    return wt, mutants, scores


def _baseline_spearman(model_obj, device, mutants, wt, scores, **embed_kw):
    mut_embs = embed_sequences(model_obj, False, mutants, device, **embed_kw)
    wt_emb = embed_sequences(model_obj, False, [wt], device, **embed_kw)
    sims = F.cosine_similarity(
        torch.as_tensor(np.asarray(mut_embs, dtype=np.float64)),
        torch.as_tensor(np.asarray(wt_emb, dtype=np.float64)),
    ).numpy()
    corr, _ = spearmanr(scores, sims)
    return float(corr)


def run_blat_smoke(n_variants=40, iters=8, lr=4e-4, layers=2, mask_rate=0.15,
                   train_head=True, seed=0, device="cpu",
                   model_name="chandar-lab/AMPLIFY_120M"):
    wt, mutants, scores = load_blat(n_variants, seed)
    log.info("BLAT assay: WT len=%d, %d variants", len(wt), len(mutants))

    model_obj, is_sbert, device = load_model(model_name, device=device)
    assert not is_sbert, "AMPLIFY should load as a (tokenizer, model) HF tuple"
    tokenizer, model = model_obj
    embed_kw = dict(batch_size=16, max_length=1024, embed_save_path=None)

    baseline = _baseline_spearman(model_obj, device, mutants, wt, scores, **embed_kw)
    log.info("Baseline (no TTA) Spearman: %+.4f", baseline)

    refs = resolve_mlm_head(model, tokenizer)
    cfg = TTAConfig(iters=iters, lr=lr, n_layers=layers, mask_rate=mask_rate,
                    train_head=train_head, seed=seed)
    groups = np.array(["BLAT"] * len(mutants))

    def embed_fn(seqs):
        return embed_sequences(model_obj, False, seqs, device, **embed_kw)

    tta_scores = run_tta_zeroshot(
        model, refs, tokenizer, mutants, [wt] * len(mutants), groups, scores,
        problem_type="regression", embed_fn=embed_fn, tta_cfg=cfg, device=device,
        max_length=1024,
    )
    tta = float(tta_scores[0]) if tta_scores else float("nan")
    log.info("WT-TTT     (--tta) Spearman: %+.4f  [iters=%d lr=%g layers=%d head=%s]",
             tta, iters, lr, layers, train_head)
    log.info("Delta (tta - baseline):      %+.4f", tta - baseline)
    return {"baseline": baseline, "tta": tta, "delta": tta - baseline,
            "n_variants": len(mutants), "wt_len": len(wt)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-variants", type=int, default=40)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--no-train-head", dest="train_head", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="chandar-lab/AMPLIFY_120M")
    args = ap.parse_args()
    res = run_blat_smoke(
        n_variants=args.n_variants, iters=args.iters, lr=args.lr, layers=args.layers,
        train_head=args.train_head, seed=args.seed, model_name=args.model,
    )
    log.info("RESULT %s", res)


if __name__ == "__main__":
    main()
