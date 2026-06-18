"""Proteva trunk + AUX-head ProteinGym zero-shot DMS-substitution scorer.

A SEPARATE scorer that runs *alongside* the canonical MLM-marginal scorer
(:mod:`plm.bench.proteingym_mlm_zeroshot`). It reuses that module's
native-context windowing (``get_optimal_window``), masked-marginal log-prob
tables (``windowed_logp_table`` / ``masked_marginal_logprob_table``), the needed-
positions union, and the ProteinGym DMS-substitution data loading verbatim, then
adds THREE Proteva-only aux scores derived from the trained aux heads:

  1. ``mlm_marginal``     — the canonical masked-marginal LLR (leaderboard-
                            comparable baseline; identical code path to the MLM
                            scorer so aux lift = score - mlm_marginal).
  2. ``di3_sad``          — structure-disruption SAD: symmetric-KL between the WT
                            and mutant 3Di (Foldseek structure-state) distributions
                            from ``out.di3_logits``, summed over a window around
                            each mutated site, negated. One unmasked forward per
                            distinct mutant sequence (capped + windowed).
  3. ``cons_weighted_mlm``— VESPA-style conservation-weighted MLM LLR, using the
                            scalar per-position ``out.cons_pred`` (low = conserved)
                            to up-weight conserved sites. ~Zero extra forwards.

and an ``ensemble`` that z-scores the three terms per assay and sums them.

Substitutions only (DMS_substitutions). CPU- and GPU-capable; the smoke test runs
on CPU. No existing file is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# protein_benchmark_suite (and its load_model) does bench-local imports
# (`from benchmark_comparison import ...`) that assume plm/bench is on sys.path —
# true when scripts are run as files (run_full_bench.sh), but not under `-m`.
# Add it so this module works either way without touching existing files.
_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from plm.bench.benchmark_tasks import TASKS
# Reuse the canonical MLM scorer's primitives verbatim (no reimplementation).
from plm.bench.proteingym_mlm_zeroshot import (
    _detect_native_context,
    get_optimal_window,
    windowed_logp_table,
    _score_substitution_windowed,
)
# Reuse the collator-derived canonical AA->token-id lookup (the correct one;
# NOT the stale CANON_TOKEN_IDS in uc30_aux_loader.py).
from plm.bench.zero_shot_dms import aa_token_id_lookup

_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"

# Tasks this scorer supports (substitution DMS only).
SUBSTITUTION_ZS = ["proteingym_dms_substitutions_zeroshot"]


# --------------------------------------------------------------------------- #
# Aux-forward helpers (unmasked WT/MUT forwards -> di3 + cons).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _aux_forward(model, tokenizer, seqs, device, max_length):
    """One unmasked forward over a batch of (cropped) sequences.

    Returns, per sequence in the batch:
      - ``di3_logp``: (T_res, 20) log-softmax over 3Di structure states, sliced to
        the residue tokens (BOS/EOS dropped), CPU float32.
      - ``cons``: (T_res,) scalar conservation prediction (low = conserved), CPU.
      - ``mlm_logp``: (T_res, 20) log-softmax over the 20 canonical-AA decoder
        columns (the "PSSM profile"), CPU — used only for the canon sanity check.
    Sequences are right-padded; per-sequence residue lengths recover the true crop.
    """
    enc = tokenizer(list(seqs), padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    ids = enc["input_ids"]
    am = enc.get("attention_mask", None)
    out = model(input_ids=ids.to(device),
                attention_mask=am.to(device) if am is not None else None)
    di3 = torch.log_softmax(out.di3_logits.float(), dim=-1).cpu()      # (B,T,20)
    cons = out.cons_pred.float().squeeze(-1).cpu()                     # (B,T)
    mlm = torch.log_softmax(out.logits.float(), dim=-1).cpu()          # (B,T,V)
    special = set(int(x) for x in (getattr(tokenizer, "all_special_ids", []) or []))
    res = []
    ids_list = ids.tolist()
    for b, row in enumerate(ids_list):
        keep = [t for t, x in enumerate(row) if x not in special]
        res.append((di3[b, keep], cons[b, keep], mlm[b, keep]))
    return res


@torch.no_grad()
def _aux_di3_forward_gpu(model, tokenizer, seqs, device, max_length):
    """Batched unmasked forward returning di3 log-softmax ON the device.

    The GPU-resident counterpart of :func:`_aux_forward` for the di3_sad hot path:
    it skips the cons/mlm heads and the per-seq ``.cpu()`` host transfers entirely,
    returning the 3Di (Foldseek structure-state) log-probs as a single padded
    device tensor plus the per-sequence residue length, so the downstream symKL +
    window-sum can run vectorized on-device with one final scalar transfer.

    Returns:
      - ``di3``: (B, T_res_max, 20) log-softmax over 3Di states, sliced to the
        residue tokens (BOS/EOS/PAD dropped) and LEFT-aligned into a padded tensor,
        ON ``device`` (float32). Rows shorter than ``T_res_max`` are zero-padded;
        the padded tail is never read (callers gate on ``lengths``).
      - ``lengths``: (B,) int64 residue counts (== number of kept tokens per seq),
        ON ``device``.
    Identical residue selection to :func:`_aux_forward` (same special-id drop), so
    ``di3[b, :lengths[b]]`` equals that function's per-seq ``di3_logp`` exactly.
    """
    enc = tokenizer(list(seqs), padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    ids = enc["input_ids"]
    am = enc.get("attention_mask", None)
    out = model(input_ids=ids.to(device),
                attention_mask=am.to(device) if am is not None else None)
    di3_all = torch.log_softmax(out.di3_logits.float(), dim=-1)        # (B,T,20) device
    special = set(int(x) for x in (getattr(tokenizer, "all_special_ids", []) or []))
    ids_list = ids.tolist()
    keeps = [[t for t, x in enumerate(row) if x not in special] for row in ids_list]
    lengths = torch.tensor([len(k) for k in keeps], dtype=torch.long, device=device)
    B = di3_all.shape[0]
    Tmax = int(lengths.max().item()) if B else 0
    di3 = di3_all.new_zeros((B, Tmax, 20))                             # (B,Tmax,20) device
    for b, keep in enumerate(keeps):
        if keep:
            di3[b, :len(keep)] = di3_all[b, keep]
    return di3, lengths


def _sym_kl(logp, logq):
    """Symmetric KL between two distributions given as log-probs, per row.

    symKL(P,Q) = 0.5*KL(P||Q) + 0.5*KL(Q||P)
               = 0.5 * sum (p-q) * (logp - logq).  (>= 0; 0 iff P==Q.)
    ``logp``/``logq`` are (..., K) log-softmax tensors; returns (...,) symKL.
    """
    p = logp.exp()
    q = logq.exp()
    return 0.5 * ((p - q) * (logp - logq)).sum(dim=-1)


def _cons_weights(cons_vec, lo=0.2, hi=1.0):
    """VESPA-style per-position weights from the scalar conservation head.

    Standardize ``cons_pred`` within the protein (low = conserved), then
    w_p = clip(sigmoid(-z_p), lo, hi): conserved (low cons -> low z) up-weighted.
    Returns a 1-D numpy array aligned with the residue positions.
    """
    c = np.asarray(cons_vec, dtype=np.float64)
    mu, sd = c.mean(), c.std()
    z = (c - mu) / sd if sd > 1e-8 else np.zeros_like(c)
    w = 1.0 / (1.0 + np.exp(z))  # sigmoid(-z)
    return np.clip(w, lo, hi)


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-8 else np.zeros_like(x)


def _profile_llr(diffs, profile_at, aa2id):
    """Unmasked WT-marginal LLR over the PSSM-distilled MLM head ("pssm_marginal").

    The Proteva MLM head is trained to emit the MSA PSSM profile (PSSM distilled
    into ``out.logits``; no separate head). For a variant with mutated positions
    ``diffs`` = ``[(pos, wt_aa, mut_aa), ...]`` the score is the log-likelihood
    ratio summed over those positions, read from a SINGLE unmasked WT forward:

        sum_p  profile_logp[p][mut] - profile_logp[p][wt]

    (the softmax log-Z cancels in the difference, so a full-vocab log-softmax row
    is fine). ``profile_at(p)`` returns the (V,) unmasked log-prob row at residue
    ``p`` or ``None`` if unavailable (long-WT windows). Non-canonical AAs are
    skipped (contribute 0, matching cons_weighted_mlm); returns ``None`` if any
    needed position's profile row is missing.
    """
    total = 0.0
    for (p, w_aa, m_aa) in diffs:
        if w_aa not in aa2id or m_aa not in aa2id:
            continue
        row = profile_at(p)
        if row is None:
            return None
        total += float(row[aa2id[m_aa]] - row[aa2id[w_aa]])
    return total


def _combine_weighted(term_arrays, weights):
    """Weighted ensemble: z-score each per-assay term, then weighted-sum.

    ``term_arrays`` is a list of K raw per-variant score lists (same length);
    ``weights`` is K floats (need not sum to 1). Returns a python list of the
    combined per-variant scores. Each term is z-scored within the assay first so
    terms on different scales (LLR vs symKL) combine comparably.
    """
    zs = np.vstack([_zscore(t) for t in term_arrays])      # (K, n)
    w = np.asarray(weights, dtype=np.float64)
    return list(w @ zs)


def _simplex_grid(K, step=0.1):
    """All weight vectors of length ``K`` on the probability simplex, grid ``step``.

    e.g. K=3, step=0.5 -> (1,0,0),(.5,.5,0),(.5,0,.5),(0,1,0),(0,.5,.5),(0,0,1).
    Used by :func:`_best_simplex_weights` for the in-sample weight ceiling.
    """
    n = int(round(1.0 / step))
    pts = []

    def rec(k, rem, acc):
        if k == K - 1:
            pts.append([*acc, rem / n])
            return
        for i in range(rem + 1):
            rec(k + 1, rem - i, [*acc, i / n])

    rec(0, n, [])
    return pts


def _best_simplex_weights(per_assay_terms, per_assay_ys, step=0.1):
    """In-sample "cheat" weight ceiling for the seq ensemble.

    Grid-searches simplex weights (sum to 1, granularity ``step``) over the K
    terms to MAXIMISE the mean per-assay Spearman. Weights are fit on the same
    assays they're scored on (deliberately in-sample -- it is an upper bound on
    what any fixed weighting could achieve, NOT a held-out result). Each term is
    z-scored within its assay before weighting (same convention as the shipped
    ensemble). ``per_assay_terms[a]`` is a (K, n_a) raw-term array; returns
    ``(best_weights_tuple, best_mean_spearman)``.
    """
    if not per_assay_terms:
        return None, float("nan")
    K = per_assay_terms[0].shape[0]
    zt = [np.vstack([_zscore(t[k]) for k in range(K)]) for t in per_assay_terms]
    best_w, best = None, -np.inf
    for w in _simplex_grid(K, step):
        wv = np.asarray(w, dtype=np.float64)
        rs = []
        for terms_z, y in zip(zt, per_assay_ys):
            ens = wv @ terms_z
            if np.std(ens) < 1e-12:
                continue
            r, _ = spearmanr(y, ens)
            if not np.isnan(r):
                rs.append(r)
        m = float(np.mean(rs)) if rs else -np.inf
        if m > best:
            best, best_w = m, tuple(w)
    return best_w, best


def _opt_int(s):
    """argparse type: int, or None for 'None'/'none'/''/0/negative (disable cap)."""
    if s is None or str(s).strip().lower() in ("none", "", "null"):
        return None
    v = int(s)
    return None if v <= 0 else v


# --------------------------------------------------------------------------- #
# Per-assay scoring.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _score_assay(model, tokenizer, refs, wt, idx, muts, labels, bin_labels,
                 aa2id, device, max_length, model_window, batch_size,
                 di3_window, compute_di3, ensemble_weights=(1.0, 1.0, 1.0)):
    """Score one assay; return per-variant {score_name: [values]} + labels.

    All variants share one WT aux forward (di3 + cons + the unmasked profile)
    and one masked-marginal table (mlm). ``pssm_marginal`` reuses the WT aux
    forward's profile (no extra cost); ``di3_sad`` additionally does one unmasked
    forward per DISTINCT mutant sequence (batched, capped, windowed). Returns
    None if <2 scorable.
    """
    wt_bytes = np.frombuffer(wt.encode("latin-1"), dtype=np.uint8)

    # --- 1. Masked-marginal MLM table over the union of mutated positions ----
    need = set()
    diffs = {}  # i -> [(pos, wt_aa, mut_aa), ...]
    for i in idx:
        mut = muts[int(i)]
        if len(mut) != len(wt):
            continue
        mb = np.frombuffer(mut.encode("latin-1"), dtype=np.uint8)
        dpos = [int(p) for p in np.nonzero(wt_bytes != mb)[0]]
        diffs[int(i)] = [(p, wt[p], mut[p]) for p in dpos]
        need.update(dpos)
    pos_cache = windowed_logp_table(
        refs, tokenizer, wt, sorted(need), model_window, device, max_length, batch_size)

    def logp_at(p):
        return pos_cache.get(p)

    # --- 2. WT aux forward (di3 + cons), windowed if WT is long ---------------
    # For short WT (<= model_window) the whole sequence is one crop; for long WT
    # we crop a window per needed position (same policy as the MLM table). For
    # cons/di3 of a SHORT protein the single full forward suffices; for long WT
    # we fall back to per-position windows only where needed.
    # The same unmasked WT forward that yields di3/cons also yields the per-
    # position profile (out.logits = the PSSM-distilled head); keep it for the
    # fast ``pssm_marginal`` (WT-marginal LLR). ``prof_at(p)`` -> (V,) log-prob
    # row at residue p, or None where unavailable (long-WT windows).
    wt_cons = None
    if len(wt) <= model_window:
        di3_full, cons_full, mlm_full = _aux_forward(model, tokenizer, [wt], device, max_length)[0]
        wt_cons = cons_full.numpy()
        wt_di3_full = di3_full  # (L,20)
        _Lp = mlm_full.shape[0]

        def prof_at(p, _prof=mlm_full, _Lp=_Lp):
            return _prof[p] if 0 <= p < _Lp else None
    else:
        wt_di3_full = None  # long-WT: di3 windows computed lazily per mutant below
        # cons weights AND the profile need a per-position value; compute via
        # windows over `need`.
        cons_arr = np.zeros(len(wt), dtype=np.float64)
        seen = np.zeros(len(wt), dtype=bool)
        prof_rows = {}
        triples = [(p, *get_optimal_window(p, len(wt), model_window)) for p in sorted(need)]
        for b0 in range(0, len(triples), batch_size):
            chunk = triples[b0:b0 + batch_size]
            res = _aux_forward(model, tokenizer, [wt[st:en] for _, st, en in chunk],
                               device, max_length)
            for (p, st, en), (_d, c, m) in zip(chunk, res):
                cons_arr[p] = float(c[p - st]); seen[p] = True
                prof_rows[p] = m[p - st]
        cons_arr[~seen] = cons_arr[seen].mean() if seen.any() else 0.0
        wt_cons = cons_arr

        def prof_at(p, _rows=prof_rows):
            return _rows.get(p)

    cons_w = _cons_weights(wt_cons)

    # --- 3. di3_sad: per distinct-mutant unmasked forward (GPU-vectorized) ----
    di3_sad = {}  # i -> score
    if compute_di3:
        # Group variants by distinct mutant sequence so each mutant forwards once.
        seq2idxs = defaultdict(list)
        for i in idx:
            if int(i) in diffs:
                seq2idxs[muts[int(i)]].append(int(i))
        distinct = list(seq2idxs.keys())

        # Short-WT path keeps the full WT di3 ON the device once and slices it per
        # crop; long-WT path batches the WT re-forward (same policy as before).
        wt_di3_dev = None
        if wt_di3_full is not None:
            # wt_di3_full is the short-WT full forward; promote it to a device tensor
            # once (same numbers, no .cpu() round-trips in the loop).
            wt_di3_dev = wt_di3_full.to(device).float()  # (L,20) on device

        # One unmasked forward per distinct mutant sequence (batched). The whole
        # symKL + per-variant window-sum runs on-device; only the final per-variant
        # scalar scores are pulled to CPU (once per assay, below).
        for b0 in range(0, len(distinct), batch_size):
            chunk = distinct[b0:b0 + batch_size]
            crops = []
            for mseq in chunk:
                i0 = seq2idxs[mseq][0]
                first_pos = diffs[i0][0][0] if diffs[i0] else 0
                st, en = get_optimal_window(first_pos, len(wt), model_window)
                crops.append((mseq, st, en))

            # Mutant di3 over each crop, ON device: (B, Tm, 20) + residue lengths.
            q_mut, len_mut = _aux_di3_forward_gpu(
                model, tokenizer, [m[st:en] for m, st, en in crops], device, max_length)
            B, Tm, _ = q_mut.shape

            # WT di3 over the SAME crops, aligned into (B, Tw, 20) + lengths. Short
            # WT slices the cached full forward; long WT batches the re-forward.
            if wt_di3_dev is not None:
                len_wt = torch.tensor([en - st for (_m, st, en) in crops],
                                      dtype=torch.long, device=device)
                Tw = int(len_wt.max().item()) if B else 0
                q_wt = q_mut.new_zeros((B, Tw, 20))
                for b, (_m, st, en) in enumerate(crops):
                    q_wt[b, :en - st] = wt_di3_dev[st:en]
            else:
                q_wt, len_wt = _aux_di3_forward_gpu(
                    model, tokenizer, [wt[st:en] for (_m, st, en) in crops],
                    device, max_length)
                Tw = q_wt.shape[1]

            # Effective per-row length = min(mut, wt) (the old code truncated to the
            # shorter crop before symKL). symKL per position on a common T.
            n = torch.minimum(len_mut, len_wt)                         # (B,)
            T = min(Tm, Tw)
            sk = _sym_kl(q_wt[:, :T], q_mut[:, :T])                    # (B,T) device
            # Mask positions beyond the effective length so window sums match the
            # old `kl = _sym_kl(...[:n])` (anything >= n[b] contributes 0).
            ar = torch.arange(T, device=device).unsqueeze(0)          # (1,T)
            valid = ar < n.unsqueeze(1).clamp(max=T)                  # (B,T)
            sk = sk * valid
            # Prefix sum: window [lo, hi) sum = csum[b,hi] - csum[b,lo].
            csum = torch.zeros((B, T + 1), dtype=sk.dtype, device=device)
            csum[:, 1:] = sk.cumsum(dim=1)

            # Build index tensors for ALL (variant, mutated-position) pairs in this
            # batch: row b, lo = max(0, lp-k), hi = min(n_eff, lp+k+1), keeping only
            # pairs with 0 <= lp < n_eff (== the old `0 <= lp < len(kl)` guard).
            # Hoist the per-row effective lengths to host once (one sync per batch,
            # not one per crop) before the pure-Python index assembly.
            neff_list = torch.minimum(len_mut, len_wt).tolist()
            rows, los, his, var_slot = [], [], [], []
            batch_vars = []  # variant ids in this batch, in slot order
            for b, (mseq, st, en) in enumerate(crops):
                neff = int(neff_list[b])
                for i in seq2idxs[mseq]:
                    slot = len(batch_vars)
                    batch_vars.append(i)
                    for (p, _w, _m) in diffs[i]:
                        lp = p - st
                        if 0 <= lp < neff:
                            lo = lp - di3_window
                            if lo < 0:
                                lo = 0
                            hi = lp + di3_window + 1
                            if hi > neff:
                                hi = neff
                            rows.append(b)
                            los.append(lo)
                            his.append(hi)
                            var_slot.append(slot)

            nvar = len(batch_vars)
            acc = torch.zeros(nvar, dtype=csum.dtype, device=device)
            if rows:
                rows_t = torch.tensor(rows, dtype=torch.long, device=device)
                los_t = torch.tensor(los, dtype=torch.long, device=device)
                his_t = torch.tensor(his, dtype=torch.long, device=device)
                slot_t = torch.tensor(var_slot, dtype=torch.long, device=device)
                wsum = csum[rows_t, his_t] - csum[rows_t, los_t]      # per-pair sums
                acc.index_add_(0, slot_t, wsum)                      # sum per variant
            acc = (-acc).cpu()                                        # negate; one xfer
            for slot, i in enumerate(batch_vars):
                di3_sad[i] = float(acc[slot])

    # --- 4. assemble per-variant scores --------------------------------------
    out_scores = {"mlm_marginal": [], "cons_weighted_mlm": [], "di3_sad": [],
                  "pssm_marginal": []}
    ys, ys_bin, kept = [], [], []
    for i in idx:
        ii = int(i)
        if ii not in diffs:
            continue
        # mlm_marginal (canonical LLR) — identical code path to the MLM scorer.
        mlm = _score_substitution_windowed(wt, muts[ii], logp_at, aa2id, wt_bytes=wt_bytes)
        if mlm is None:
            continue
        # cons-weighted MLM (per-position weighted LLR over the same rows).
        cw = 0.0
        ok = True
        for (p, w_aa, m_aa) in diffs[ii]:
            if w_aa not in aa2id or m_aa not in aa2id:
                continue
            row = logp_at(p)
            if row is None:
                ok = False
                break
            cw += cons_w[p] * float(row[aa2id[m_aa]] - row[aa2id[w_aa]])
        if not ok:
            continue
        # pssm_marginal (unmasked WT-marginal LLR over the PSSM-distilled head).
        pssm = _profile_llr(diffs[ii], prof_at, aa2id)
        if pssm is None:
            continue
        out_scores["mlm_marginal"].append(mlm)
        out_scores["cons_weighted_mlm"].append(cw)
        out_scores["di3_sad"].append(di3_sad.get(ii, np.nan) if compute_di3 else np.nan)
        out_scores["pssm_marginal"].append(pssm)
        ys.append(labels[ii])
        ys_bin.append(bin_labels[ii] if bin_labels is not None else None)
        kept.append(ii)

    if len(kept) < 2:
        return None

    # ensemble: z-score each term across this assay's variants, sum (incl di3).
    terms = [_zscore(out_scores["mlm_marginal"]),
             _zscore(out_scores["cons_weighted_mlm"])]
    if compute_di3 and np.isfinite(out_scores["di3_sad"]).all():
        terms.append(_zscore(out_scores["di3_sad"]))
    out_scores["ensemble"] = list(np.sum(terms, axis=0))

    # ensemble_seq: the FAST seq-only alternate — mlm + cons + pssm, no di3
    # (weighted z-score sum). One masked table + one unmasked WT forward; cost is
    # independent of #variants, so it runs at mlm speed even with di3 disabled.
    seq_terms = np.vstack([out_scores["mlm_marginal"],
                           out_scores["cons_weighted_mlm"],
                           out_scores["pssm_marginal"]])
    out_scores["ensemble_seq"] = _combine_weighted(
        [out_scores["mlm_marginal"], out_scores["cons_weighted_mlm"],
         out_scores["pssm_marginal"]], ensemble_weights)

    return {"scores": out_scores, "ys": ys, "ys_bin": ys_bin, "n": len(kept),
            "seq_terms": seq_terms}


def _assay_metrics(name, score_vals, ys, ys_bin, problem_type, bin_present,
                   negate_auc=False):
    """Per-assay Spearman (+ optional AUC) for one score over one assay.

    ``negate_auc`` negates the score before the AUC (clinical pathogenicity: bin
    label 1 = pathogenic = deleterious = LOW LLR, so pathogenic must rank HIGH —
    matches proteingym_mlm_zeroshot). DMS bin label 1 = high fitness, no flip.
    """
    sv = np.asarray(score_vals, dtype=np.float64)
    if not np.isfinite(sv).all() or np.unique(sv[np.isfinite(sv)]).size < 2:
        return None
    r, _ = spearmanr(ys, sv)
    primary = float(r) if not np.isnan(r) else 0.0
    auc = None
    if bin_present:
        sv_auc = -sv if negate_auc else sv
        pairs = [(b, s) for b, s in zip(ys_bin, sv_auc)
                 if b is not None and not np.isnan(float(b))]
        if len(pairs) == len(sv):
            yb = [int(round(float(b))) for b, _ in pairs]
            if len(set(yb)) == 2:
                try:
                    auc = float(roc_auc_score(yb, [s for _, s in pairs]))
                except ValueError:
                    auc = None
    return {"assay": name, "score": name, "spearman": primary, "auc": auc}


# --------------------------------------------------------------------------- #
# Task driver.
# --------------------------------------------------------------------------- #
def _eval_task(task_key, model, tokenizer, refs, device, batch_size, max_length,
               model_window, max_assays, max_variants_per_assay, di3_window,
               only_assays=None, compute_di3=True, ensemble_weights=(1.0, 1.0, 1.0)):
    cfg = TASKS[task_key]
    from datasets import load_dataset

    load_kwargs = {"data_dir": cfg.data_dir} if cfg.data_dir else {}
    try:
        ds = load_dataset(cfg.dataset, **load_kwargs)
    except Exception:
        ds = load_dataset(cfg.dataset, trust_remote_code=True, **load_kwargs)
    split = cfg.train_split if cfg.train_split in ds else list(ds.keys())[0]
    data = ds[split]

    mut_col, wt_col = cfg.input_map["mutant"], cfg.input_map["wt"]
    muts = list(data[mut_col])
    wts = data[wt_col]
    # clinical_substitutions has a STRING annotation label -> map to {0,1}; DMS
    # ships a float DMS_score directly.
    label_map = getattr(cfg, "label_map", None)
    if label_map:
        labels = np.asarray([label_map.get(str(x), np.nan) for x in data[cfg.label_col]],
                            dtype=float)
    else:
        labels = np.asarray(data[cfg.label_col], dtype=object).astype(float)
    groups = np.asarray(data[cfg.group_by])
    bin_labels = None
    if cfg.bin_col and cfg.bin_col in data.column_names:
        bin_labels = np.asarray(data[cfg.bin_col], dtype=float)
    elif getattr(cfg, "problem_type", None) == "binary":
        # clinical: the {0,1} annotation IS the binary label -> enables per-assay AUC.
        bin_labels = labels

    aa2id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in _AA_LETTERS}

    g2idx = defaultdict(list)
    for _i, _g in enumerate(groups.tolist()):
        g2idx[str(_g)].append(_i)
    assays = list(np.unique(groups))
    if only_assays:
        want = set(only_assays)
        assays = [a for a in assays if str(a) in want]
    if max_assays:
        assays = assays[:max_assays]

    # pssm_marginal + ensemble_seq are the new fast seq-only scores.
    SCORE_NAMES = ["mlm_marginal", "di3_sad", "cons_weighted_mlm", "pssm_marginal",
                   "ensemble", "ensemble_seq"]
    recs = {s: [] for s in SCORE_NAMES}
    timings = {s: 0.0 for s in SCORE_NAMES}
    per_assay_n = []
    seq_terms_all, seq_ys_all = [], []  # for the in-sample weight ceiling

    n_assays_total = len(assays)
    for k_assay, g in enumerate(assays, start=1):
        idx = np.asarray(g2idx[str(g)], dtype=int)
        if max_variants_per_assay and idx.size > max_variants_per_assay:
            idx = idx[:max_variants_per_assay]
        wt = wts[int(idx[0])]

        t_all0 = time.time()
        res = _score_assay(model, tokenizer, refs, wt, idx, muts, labels, bin_labels,
                           aa2id, device, max_length, model_window, batch_size,
                           di3_window, compute_di3, ensemble_weights=ensemble_weights)
        elapsed = time.time() - t_all0
        # Lightweight progress so long full-data runs aren't opaque.
        print(f"[aux] assay {k_assay}/{n_assays_total} {str(g):<45s} "
              f"nvar={idx.size:>5d} wtlen={len(wt):>5d} {elapsed:>7.1f}s",
              flush=True)
        if res is None:
            continue
        # crude attribution: di3 dominates the per-mutant forwards; split the
        # measured wall-clock by counting distinct mutants vs the shared tables.
        timings["di3_sad"] += elapsed  # whole-assay (di3 is the marginal extra cost)
        for s in ["mlm_marginal", "cons_weighted_mlm", "pssm_marginal",
                  "ensemble", "ensemble_seq"]:
            timings[s] += 0.0
        per_assay_n.append(res["n"])
        seq_terms_all.append(res["seq_terms"])
        seq_ys_all.append(np.asarray(res["ys"], dtype=np.float64))

        for s in SCORE_NAMES:
            m = _assay_metrics(s, res["scores"][s], res["ys"], res["ys_bin"],
                               cfg.problem_type, bin_labels is not None,
                               negate_auc=(cfg.problem_type == "binary"))
            if m is not None:
                m["assay"] = str(g)
                m["n"] = res["n"]
                recs[s].append(m)

    # "Cheat" ceiling: best fixed simplex weights over (mlm,cons,pssm), fit
    # in-sample on these assays (upper bound, NOT held out — see docstring).
    best_w, best_sp = _best_simplex_weights(seq_terms_all, seq_ys_all, step=0.1)

    return {"task": task_key, "recs": recs, "timings": timings,
            "n_assays": len(per_assay_n),
            "ensemble_weights": list(ensemble_weights),
            "best_seq_weights": list(best_w) if best_w is not None else None,
            "best_seq_spearman": best_sp}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_name", default="/data/proteva/cache/ckpts/hf_stage2_final")
    ap.add_argument("--tasks", nargs="*", default=SUBSTITUTION_ZS)
    # Canonical aux dirs are aux_zs_<tag>_full2/ (what final_benchmark_report reads);
    # the orchestrator passes the per-model dir explicitly. This bare default follows
    # the _full2 convention so a one-off run never resurrects the old (deleted) aux_zs/.
    ap.add_argument("--output_dir", default="/data/proteva/plm/results/bench/aux_zs_full2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--model_window", type=int, default=None)
    ap.add_argument("--max_assays", type=int, default=None)
    ap.add_argument("--assays", nargs="*", default=None,
                    help="optional explicit DMS_id list to score (e.g. for a fast "
                         "smoke test on small assays); default scores all assays.")
    ap.add_argument("--max_variants_per_assay", type=_opt_int, default=2000,
                    help="cap per-mutant di3 forwards (default 2000). Pass "
                         "'None'/'none'/0/-1 to DISABLE the cap and score full data "
                         "(now fast enough thanks to the GPU-vectorized di3 path).")
    ap.add_argument("--di3_window", type=int, default=4,
                    help="half-width k of the symKL window around each mutated site.")
    ap.add_argument("--no_di3", action="store_true",
                    help="skip the expensive di3_sad per-mutant forwards; run only "
                         "the cheap mlm_marginal + cons_weighted_mlm + pssm_marginal "
                         "+ ensemble_seq path (fully fast, mlm-speed).")
    ap.add_argument("--ensemble_weights", type=str, default="1,1,1",
                    help="3 comma-separated weights (mlm,cons,pssm) for the fast "
                         "ensemble_seq (terms z-scored first; need not sum to 1). "
                         "Default equal. The report also prints the in-sample best "
                         "weights as a ceiling.")
    ap.add_argument("--sanity_check", action="store_true", default=True,
                    help="verify the canon AA token-id map recovers WT residues.")
    args = ap.parse_args(argv)
    ens_w = tuple(float(x) for x in str(args.ensemble_weights).split(","))
    if len(ens_w) != 3:
        raise SystemExit(f"--ensemble_weights needs 3 values (mlm,cons,pssm), got {ens_w}")

    from plm.bench.protein_benchmark_suite import load_model
    from plm.bench.wt_test_time_training import resolve_mlm_head

    model_obj, is_sbert, device = load_model(args.model_name, device=args.device)
    if is_sbert:
        raise SystemExit("model loaded as SentenceTransformer; need ProtevaForPretraining.")
    tokenizer, model = model_obj
    model.eval()
    # Ensure CPU-safe dense attention (the spec's flash_attn_mode='off').
    try:
        model.encoder.config.flash_attn_mode = "off"
    except Exception:
        pass

    refs = resolve_mlm_head(model, tokenizer)
    if refs is None:
        raise SystemExit("no MLM head resolved for model.")

    native_max = _detect_native_context(model, tokenizer)
    if args.max_length is None:
        args.max_length = native_max
    if native_max and args.max_length > native_max:
        args.max_length = int(native_max)
    model_window = args.model_window or (native_max - 2)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- canon-id sanity check (one short WT) --------------------------------
    if args.sanity_check:
        _canon_sanity_check(model, tokenizer, device, args.max_length)

    for task_key in args.tasks:
        if task_key not in TASKS:
            print(f"skip unknown task {task_key}")
            continue
        result = _eval_task(task_key, model, tokenizer, refs, device,
                            args.batch_size, args.max_length, model_window,
                            args.max_assays, args.max_variants_per_assay,
                            args.di3_window, only_assays=args.assays,
                            compute_di3=not args.no_di3, ensemble_weights=ens_w)
        _report(task_key, result, args, out)
    return 0


def _canon_sanity_check(model, tokenizer, device, max_length):
    """Verify the collator-derived AA->token-id map recovers WT residues.

    Runs one unmasked forward over a short WT and checks the decoder argmax over
    the 20 canonical-AA columns recovers the input residues (>=90% expected;
    the stale CANON_TOKEN_IDS gives ~20%).
    """
    from plm.hf.collator import ProteinPackedCollator
    collator = ProteinPackedCollator()
    collator._ensure_ready()
    aa_to_id = aa_token_id_lookup(collator)            # AA letter -> token id (correct)
    aa_ids = [aa_to_id[a] for a in _AA_LETTERS]
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    res = _aux_forward(model, tokenizer, [seq], device, max_length)[0]
    mlm = res[2]  # (L, V) log-probs
    cols = mlm[:, aa_ids]                              # (L, 20) over canonical AAs
    pred = cols.argmax(dim=-1).tolist()
    recovered = "".join(_AA_LETTERS[p] for p in pred)
    n = min(len(seq), len(recovered))
    acc = sum(1 for a, b in zip(seq[:n], recovered[:n]) if a == b) / n
    print(f"[sanity] di3_logits shape={tuple(res[0].shape)} "
          f"cons_pred shape={tuple(res[1].shape)} logits(20-slice) shape={tuple(cols.shape)}")
    print(f"[sanity] canon-id WT recovery (unmasked decoder argmax): {acc:.1%} over {n} residues")
    if acc < 0.5:
        print("[sanity] WARNING: low recovery — canon token-id map likely WRONG.")


def _report(task_key, result, args, out):
    recs = result["recs"]
    timings = result["timings"]
    summary = {}
    print(f"\n=== {task_key}  ({result['n_assays']} assays scored) ===")
    print(f"{'score':<20} {'mean_spearman':>14} {'n_assays':>9} {'wall_s':>9}")
    jsonl_path = out / f"aux_zs_{Path(args.model_name).name}__{task_key}.jsonl"
    with open(jsonl_path, "w") as fh:
        for score_name, rows in recs.items():
            sps = [r["spearman"] for r in rows if r["spearman"] is not None]
            mean_sp = float(np.mean(sps)) if sps else float("nan")
            aucs = [r["auc"] for r in rows if r.get("auc") is not None]
            mean_auc = float(np.mean(aucs)) if aucs else None
            wall = timings.get(score_name, 0.0)
            summary[score_name] = {"mean_spearman": mean_sp, "mean_auc": mean_auc,
                                   "n_assays": len(rows), "wall_s": wall}
            print(f"{score_name:<20} {mean_sp:>14.4f} {len(rows):>9} {wall:>9.1f}")
            for r in rows:
                fh.write(json.dumps({"task": task_key, **r}) + "\n")
    # aux lift vs mlm_marginal baseline
    base = summary.get("mlm_marginal", {}).get("mean_spearman", float("nan"))
    print(f"  (baseline mlm_marginal mean Spearman = {base:.4f}; "
          f"aux lift = score - baseline)")
    # ensemble_seq weight ceiling (in-sample upper bound; the shipped ensemble_seq
    # uses --ensemble_weights, default equal).
    bw, bsp = result.get("best_seq_weights"), result.get("best_seq_spearman")
    if bw is not None:
        print(f"  ensemble_seq weights used (mlm,cons,pssm) = "
              f"{tuple(result.get('ensemble_weights'))}; in-sample BEST = "
              f"{tuple(round(x, 2) for x in bw)} -> Spearman {bsp:.4f} (ceiling, not held out)")
    (out / f"summary_{Path(args.model_name).name}__{task_key}.json").write_text(
        json.dumps({"task": task_key, "model": args.model_name,
                    "summary": summary, "di3_window": args.di3_window,
                    "max_variants_per_assay": args.max_variants_per_assay,
                    "ensemble_weights": result.get("ensemble_weights"),
                    "best_seq_weights": bw, "best_seq_spearman": bsp}, indent=2))
    print(f"  -> JSONL: {jsonl_path}")


if __name__ == "__main__":
    raise SystemExit(main())
