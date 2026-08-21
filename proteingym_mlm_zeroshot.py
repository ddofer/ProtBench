"""Canonical MLM ProteinGym zero-shot (ESM-1v style).

Scores all four ProteinGym ZS tasks: substitutions via masked-marginal
(sum of per-position logP(mut)-logP(wt) from the WT masked table) and indels
via length-normalized pseudo-log-likelihood delta (mean PLL(mut) - PLL(WT)).
Long sequences use a mutation-centered optimal window to stay in-distribution.
Output JSONL is read by collect_bench_results.

Uses load_model() from protein_benchmark_suite (returns (model_obj, is_sbert,
device); for AMPLIFY model_obj=(tokenizer, model)) and resolve_mlm_head(model,
tokenizer) -> MLMHeadRefs(.forward_logits, .mask_token_id). Data loading mirrors
protein_benchmark_suite.prepare_data for proteingym_zeroshot.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, torch
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from benchmark_tasks import TASKS
from _hf_finetune_common import write_jsonl_record, safe_ckpt

SUBSTITUTION_ZS = [
    "proteingym_dms_substitutions_zeroshot",
    "proteingym_clinical_substitutions_zeroshot",
]
INDEL_ZS = [
    "proteingym_dms_indels_zeroshot",
    "proteingym_clinical_indels_zeroshot",
]
ALL_ZS = SUBSTITUTION_ZS + INDEL_ZS


def _detect_native_context(model, tokenizer, fallback=1024):
    """The model's trained context length (tokens), read from config — NOT
    hardcoded — so each encoder windows at ITS OWN native length: Proteva 1024
    (encoder.config.max_position, its continued-pretraining length), vanilla
    AMPLIFY 2048 (config.max_length). Tries, in priority order, the encoder's
    max_position, then the model config's max_position_embeddings / n_positions /
    max_length, then the tokenizer's model_max_length. Each candidate is sanity-
    bounded to [16, 1e6] to reject HF's int-sentinel (1e30) model_max_length and
    bogus values; falls back to ``fallback`` only if nothing sane is found.
    """
    cands = []
    enc = getattr(model, "encoder", None)
    ecfg = getattr(enc, "config", None)
    cfg = getattr(model, "config", None)
    for obj, attr in [(ecfg, "max_position"),
                      (cfg, "max_position_embeddings"),
                      (cfg, "n_positions"),
                      (cfg, "max_length"),
                      (cfg, "max_position"),
                      (tokenizer, "model_max_length")]:
        v = getattr(obj, attr, None) if obj is not None else None
        if isinstance(v, int) and 16 <= v <= 1_000_000:
            cands.append(int(v))
    return cands[0] if cands else fallback


def score_substitution(wt, mut, logP, aa2id):
    """Sum of per-position log-prob deltas; None if lengths differ (indel)."""
    if len(wt) != len(mut):
        return None
    if len(wt) > logP.shape[0]:
        return None  # truncated; skip variant
    s = 0.0
    for i, (w, m) in enumerate(zip(wt, mut)):
        if w == m or w not in aa2id or m not in aa2id:
            continue
        s += float(logP[i, aa2id[m]] - logP[i, aa2id[w]])
    return s


def pll_from_table(seq, logP, aa2id):
    """Length-normalized pseudo-log-likelihood of ``seq`` from its masked table.

    mean-PLL = (1/L) * sum_i log P(seq[i] | seq with position i masked), over the
    L canonical-AA positions. The encoder/masked-LM analogue of a sequence
    likelihood (ESM-1v / ESM2; pseudo-perplexity = -mean-PLL). It needs no
    position alignment, so an indel score is ``mean_PLL(mut) - mean_PLL(WT)``
    even when lengths differ. The 1/L normalization is what makes the comparison
    valid across different-length sequences — without it longer variants get a
    systematically more-negative raw sum, conflating length with fitness
    (Engelberg/Frey et al., PRX Life 2025, "Pseudo-perplexity in One Fell Swoop";
    cf. Tranception's additive length term, Notin et al. ICML 2022).
    Returns None if the sequence was truncated past the table or has no scorable
    positions.
    """
    if len(seq) > logP.shape[0]:
        return None  # truncated; skip variant
    if hasattr(logP, "numpy"):
        logP = logP.numpy()  # torch->numpy: per-position float() is ~100x cheaper
    s, n = 0.0, 0
    for i, a in enumerate(seq):
        if a in aa2id:
            s += float(logP[i, aa2id[a]])
            n += 1
    return (s / n) if n else None


def get_optimal_window(mut_pos, seq_len, model_window):
    """ProteinGym/ESM mutation-centered window in RESIDUE space (scoring_utils.py).

    Returns [start, end) residue indices so the mutated residue keeps ~half the
    window of context on each side, clamped at the sequence ends. ``model_window``
    is the model's residue capacity (native context minus the 2 special tokens),
    so the cropped sequence always tokenizes within the trained length — no RoPE
    extrapolation. Each mutation is windowed independently, so multi-mutants that
    span >model_window are still scored correctly per-site.
    """
    half = model_window // 2
    if seq_len <= model_window:
        return 0, seq_len
    if mut_pos < half:
        return 0, model_window
    if mut_pos >= seq_len - half:
        return seq_len - model_window, seq_len
    return max(0, mut_pos - half), min(seq_len, mut_pos + half)


@torch.no_grad()
def windowed_logp_table(refs, tokenizer, wt, positions, model_window, device,
                        max_length, batch_size=32):
    """Batched mutation-centered windowed masked-marginals for the long-WT path.

    For each residue position in ``positions``, score it from a window centered on
    it (``get_optimal_window``) with that residue masked, batching B windows per
    forward. The long-WT path is the ProteinGym bottleneck: ~WT-length batch-1
    forwards per assay otherwise (16/217 DMS-subs assays exceed the context).
    Returns ``{pos: [V] log-prob row (cpu)}``; positions outside the (truncated)
    token range are omitted, so a later ``logp_at`` lookup returns None and the
    variant is skipped.
    """
    out = {}
    special_set = set(refs.special_ids)
    triples = [(p, *get_optimal_window(p, len(wt), model_window)) for p in positions]
    for b0 in range(0, len(triples), batch_size):
        chunk = triples[b0:b0 + batch_size]
        seqs = [wt[st:en] for (_p, st, en) in chunk]
        enc = tokenizer(seqs, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        ids = enc["input_ids"]
        am = enc.get("attention_mask", None)
        keep = []  # (batch_row, pos, tpos)
        for j, (p, st, en) in enumerate(chunk):
            local_idx = p - st
            res_tok = [t for t, x in enumerate(ids[j].tolist()) if x not in special_set]
            if local_idx >= len(res_tok):
                continue  # outside truncated range -> omit (logp_at -> None)
            tpos = res_tok[local_idx]
            ids[j, tpos] = refs.mask_token_id
            keep.append((j, p, tpos))
        if not keep:
            continue
        lg = refs.forward_logits(ids.to(device),
                                 am.to(device) if am is not None else None)
        lsm = torch.log_softmax(lg.float(), dim=-1).cpu()
        for (j, p, tpos) in keep:
            out[p] = lsm[j, tpos].numpy()  # numpy row: fast scalar indexing in the scorer
    return out


@torch.no_grad()
def masked_marginal_logprob_table(refs, tokenizer, wt, device, max_length=1024, batch_size=64):
    """[L,V] log-softmax table; row i = log p(aa | WT with residue i masked).

    Uses refs.forward_logits (MLMHeadRefs callable) instead of a bare model
    so that AMPLIFY-specific input prep is handled transparently.
    """
    enc = tokenizer(wt, truncation=True, max_length=max_length, return_tensors="pt")
    ids = enc["input_ids"][0]
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask[0]

    mask_id = refs.mask_token_id
    special_set = set(refs.special_ids)
    res_tok = [t for t, x in enumerate(ids.tolist()) if x not in special_set]
    L = len(res_tok)
    vocab_size = tokenizer.vocab_size
    logP = torch.full((L, vocab_size), float("-inf"))

    for start in range(0, L, batch_size):
        chunk = res_tok[start:start + batch_size]
        B = len(chunk)
        batch_ids = ids.unsqueeze(0).repeat(B, 1).clone()
        for j, tpos in enumerate(chunk):
            batch_ids[j, tpos] = mask_id

        batch_ids = batch_ids.to(device)
        if attention_mask is not None:
            batch_mask = attention_mask.unsqueeze(0).repeat(B, 1).to(device)
        else:
            batch_mask = None

        # forward_logits returns [B, T, V]
        lg = refs.forward_logits(batch_ids, batch_mask)
        lsm = torch.log_softmax(lg.float(), dim=-1)
        # Gather each row's masked position on-GPU, then ONE transfer to CPU
        # (a per-position .cpu() does B syncs/batch -> dominates the build).
        idx_b = torch.arange(B, device=lsm.device)
        idx_t = torch.as_tensor(chunk, device=lsm.device)
        logP[start:start + B] = lsm[idx_b, idx_t].cpu()

    return logP


def _score_substitution_windowed(wt, mut, logp_at, aa2id, wt_bytes=None):
    """Substitution score = sum over mutated positions of logP(mut) - logP(wt).

    ``logp_at(p)`` returns the [V] masked log-prob row for residue position ``p``
    (a full-table slice for short WT, or a mutation-centered window for long WT);
    returns None if that position is unscorable. Mutated positions are found by
    diffing wt vs mut, so it works for single- and multi-mutants. Returns None
    if any required position is unscorable.

    The diff is vectorized (numpy byte compare) so the per-variant cost is
    O(#mutations), not O(len(wt)) — critical when scoring all ~2.5M ProteinGym
    variants. ``wt_bytes`` (the WT as uint8, constant per assay) is precomputed by
    the caller to avoid re-encoding the WT for every variant.
    """
    if len(wt) != len(mut):
        return None
    if wt_bytes is None:
        wt_bytes = np.frombuffer(wt.encode("latin-1"), dtype=np.uint8)
    mut_bytes = np.frombuffer(mut.encode("latin-1"), dtype=np.uint8)
    s = 0.0
    for p in np.nonzero(wt_bytes != mut_bytes)[0].tolist():
        w, m = wt[p], mut[p]
        if w not in aa2id or m not in aa2id:
            continue
        row = logp_at(p)
        if row is None:
            return None
        s += float(row[aa2id[m]] - row[aa2id[w]])
    return s


def _filter_huge_assays(assays, g2idx, thresh):
    """Drop assays with more than ``thresh`` variants (indel runtime scales with
    variant count). Returns (kept_assays, n_dropped)."""
    kept = np.asarray([a for a in assays if len(g2idx[str(a)]) <= thresh])
    return kept, len(assays) - len(kept)


def _crop_or_skip(seq, model_window, long_policy):
    """Center-crop an over-length sequence (truncate policy) or signal skip.

    Returns (seq, ok). ok=False (skip -> None) when too long under any policy
    other than "truncate". Center-crop mirrors the exact path's _center_crop.
    Shared by the single_pass and strided per-sequence scorers.
    """
    if len(seq) <= model_window:
        return seq, True
    if long_policy != "truncate":
        return seq, False
    st = (len(seq) - model_window) // 2
    return seq[st:st + model_window], True


@torch.no_grad()
def single_pass_pll_table(refs, tokenizer, seqs, aa2id, model_window, device,
                          max_length, batch_size=32, long_policy="skip"):
    """Batched SINGLE-PASS (unmasked) mean log-prob per sequence.

    score(seq) = mean_i log p(seq[i] | full UNMASKED seq), i over canonical-AA
    token positions. ONE forward per sequence instead of L masked forwards
    (masked_marginal_logprob_table). Indel score = score(mut) - score(WT).

    WARNING — REFERENCE-ONLY, validated BAD. This is the naive unmasked baseline,
    NOT OFS. Real OFS (Kantroo et al. 2024, arXiv:2407.07265) trains an MLP
    ensemble to predict the *masked* profile from unmasked embeddings; this reads
    the raw unmasked diagonal, so the model copies the true residue (leakage) and
    ranking collapses (validated 2026-06-18: flat Spearman 0.498->0.313, AMFR went
    anti-correlated). Use `strided` instead. INVARIANT if ever used: *mean* over
    positions, not sum — CAPSD_AAV2S has corr(length, DMS) ~ -0.53, so a sum scorer
    injects a spurious length term. Specials/pads excluded via aa2id + attn mask.

    Returns a list aligned with ``seqs``; entry is a float, or None when a
    sequence is over-length under long_policy="skip" (matching the masked path)
    or has no canonical positions.
    """
    canon_ids = set(aa2id.values())
    out = [None] * len(seqs)
    prepared = [_crop_or_skip(s, model_window, long_policy) for s in seqs]
    work = [(k, s) for k, (s, ok) in enumerate(prepared) if ok]
    canon_t = torch.tensor(sorted(canon_ids), device=device)

    for b0 in range(0, len(work), batch_size):
        chunk = work[b0:b0 + batch_size]
        ks = [k for k, _ in chunk]
        enc = tokenizer([s for _, s in chunk], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        ids = enc["input_ids"]
        am = enc.get("attention_mask", None)
        lg = refs.forward_logits(ids.to(device),
                                 am.to(device) if am is not None else None)
        lsm = torch.log_softmax(lg.float(), dim=-1)            # [B,T,V]
        tok = ids.to(lsm.device)                               # [B,T]
        true_lp = lsm.gather(-1, tok.unsqueeze(-1)).squeeze(-1)   # [B,T]
        is_canon = torch.isin(tok, canon_t)                   # specials excluded
        if am is not None:
            is_canon &= am.to(lsm.device).bool()              # drop pad
        sum_lp = (true_lp * is_canon).sum(dim=-1)             # [B]
        n_pos = is_canon.sum(dim=-1)                          # [B]
        s_cpu = sum_lp.cpu().tolist(); n_cpu = n_pos.cpu().tolist()
        for j, k in enumerate(ks):
            out[k] = (s_cpu[j] / n_cpu[j]) if n_cpu[j] > 0 else None
    return out


@torch.no_grad()
def strided_masked_pll_table(refs, tokenizer, seqs, aa2id, model_window, device,
                             max_length, n_passes=8, batch_size=16, long_policy="skip",
                             progress_label="", progress_every=20):
    """Leakage-free few-pass masked pseudo-log-likelihood.

    Exact masked-PLL masks ONE position per forward -> L forwards/seq. Here we
    mask a STRIDED group of positions simultaneously: in pass g (g=0..n_passes-1)
    every residue whose index % n_passes == g is masked at once, scored from the
    (still-unmasked) rest. After n_passes forwards every position has been masked
    exactly once, with its nearest co-masked neighbour n_passes residues away, so
    local context is intact. n_passes forwards/seq instead of L (~L/n_passes
    speedup); n_passes >= L reproduces exact masked-PLL. Leakage-free because the
    scored residue is always [MASK] in its own forward (unlike single_pass).
    score(seq) = mean over canonical positions of log p(true residue | strided-masked seq).
    Returns a list aligned with ``seqs`` (float or None).
    """
    mask_id = refs.mask_token_id
    special_set = set(refs.special_ids)
    canon_ids = set(aa2id.values())
    out = [None] * len(seqs)
    prepared = [_crop_or_skip(s, model_window, long_policy) for s in seqs]
    work = [(k, s) for k, (s, ok) in enumerate(prepared) if ok]

    import time as _t
    nb = (len(work) + batch_size - 1) // batch_size
    _t0 = _t.time()
    for bi, b0 in enumerate(range(0, len(work), batch_size)):
        if progress_label and (bi % progress_every == 0 or bi == nb - 1):
            done = min(b0 + batch_size, len(work)); el = _t.time() - _t0
            tail = ""
            if el > 2:                      # skip rate on first batch (el~=0 -> bogus)
                rate = done / el
                tail = f" {rate:.0f} seq/s ETA {(len(work)-done)/rate/60:.0f}m"
            print(f"    [{progress_label}] {done}/{len(work)} seqs "
                  f"({100*done//max(1,len(work))}%){tail}", flush=True)
        chunk = work[b0:b0 + batch_size]
        ks = [k for k, _ in chunk]
        enc = tokenizer([s for _, s in chunk], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        ids0 = enc["input_ids"]
        am = enc.get("attention_mask", None)
        B, T = ids0.shape
        # residue-token positions per row, and their running residue index
        res_tok = [[t for t in range(T) if ids0[j, t].item() not in special_set]
                   for j in range(B)]
        sum_lp = [0.0] * B
        n_pos = [0] * B
        for g in range(n_passes):
            ids = ids0.clone()
            masked = [[] for _ in range(B)]   # token-positions masked this pass, per row
            any_masked = False
            for j in range(B):
                for ri, tpos in enumerate(res_tok[j]):
                    if ri % n_passes == g:
                        ids[j, tpos] = mask_id
                        masked[j].append(tpos)
                        any_masked = True
            if not any_masked:
                continue
            lg = refs.forward_logits(ids.to(device),
                                     am.to(device) if am is not None else None)
            lsm = torch.log_softmax(lg.float(), dim=-1).cpu()
            for j in range(B):
                for tpos in masked[j]:
                    tok = int(ids0[j, tpos])
                    if tok in canon_ids:                       # score canonical only
                        sum_lp[j] += float(lsm[j, tpos, tok])
                        n_pos[j] += 1
        for j, k in enumerate(ks):
            out[k] = (sum_lp[j] / n_pos[j]) if n_pos[j] > 0 else None
    return out


INDEL_SCORE_MODES = ("strided", "single_pass", "masked_pll", "embedding_span", "embedding_red")


def _make_residue_embedder(model, tokenizer, device, batch_size, max_length):
    """Adapt a loaded model to the ``seqs -> [(T, d), ...]`` contract the embedding arms need.

    Reuses ``token_classification_probe.iter_residue_embeddings`` (length-sorted
    batching, on-device masking, special/pad tokens excluded). A ``*ForMaskedLM``
    wrapper returns logits from ``outputs[0]``, so hand it the base encoder.
    """
    from token_classification_probe import iter_residue_embeddings

    encoder = getattr(model, "base_model", model)

    def embed(seqs):
        return list(iter_residue_embeddings(
            encoder=encoder, tokenizer=tokenizer, sequences=list(seqs),
            device=device, batch_size=batch_size, max_length=max_length))

    return embed


def _eval_task(task_key, refs, tokenizer, device, batch_size, max_length,
               max_assays=None, max_variants_per_assay=None,
               model_window=1022, indel_long_policy="skip",
               skip_huge_assays=None, indel_score_mode="strided",
               indel_pll_passes=16, assay_shard=0, assay_num_shards=1,
               residue_embedder=None):
    cfg = TASKS[task_key]
    is_indel = task_key in INDEL_ZS
    from datasets import load_dataset

    # Mirror protein_benchmark_suite.prepare_data: data_dir kwarg + train split.
    load_kwargs = {}
    if cfg.data_dir:
        load_kwargs["data_dir"] = cfg.data_dir
    try:
        ds = load_dataset(cfg.dataset, **load_kwargs)
    except Exception as e:
        try:
            ds = load_dataset(cfg.dataset, trust_remote_code=True, **load_kwargs)
        except Exception as e2:
            raise RuntimeError(f"Failed to load dataset {cfg.dataset}: {e2}") from e

    all_keys = list(ds.keys())
    # Take train split (ProteinGym is distributed as a single "train" split)
    train_split = cfg.train_split if cfg.train_split in all_keys else all_keys[0]
    data = ds[train_split]

    mut_col = cfg.input_map["mutant"]   # "mutated_sequence"
    wt_col  = cfg.input_map["wt"]       # "target_seq"
    # Materialize muts to a Python list ONCE. data[col] is a lazy HF Arrow Column;
    # per-element column[i] is ~3ms (Arrow lookup), and muts is indexed per variant
    # (millions of times) -> it dominated runtime. list() pays ~20s once, then each
    # access is ~1us (2500x faster). wts is accessed only once per assay (217x), so
    # it stays lazy (materializing it would waste ~20s for no gain).
    muts    = list(data[mut_col])
    wts     = data[wt_col]
    labels_raw = np.asarray(data[cfg.label_col], dtype=object)
    groups  = np.asarray(data[cfg.group_by])

    # Apply label map (clinical: Pathogenic->1, Benign->0)
    if cfg.label_map:
        labels = np.array([cfg.label_map.get(str(x), x) for x in labels_raw], dtype=float)
    else:
        labels = labels_raw.astype(float)

    # Official ProteinGym binary labels for DMS AUC (the leaderboard ground truth);
    # None for clinical (which uses its own annotation) or if the column is absent.
    bin_labels = None
    if cfg.bin_col and cfg.bin_col in data.column_names:
        bin_labels = np.asarray(data[cfg.bin_col], dtype=float)

    # Build aa->token_id map for the 20 canonical amino acids
    aa2id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    # Per-assay records (keyed by assay id) so aggregation is a separate, offline
    # step — flat vs ProteinGym hierarchical vs pooled can be recomputed without
    # rescoring. pool_* accumulate raw (label, -score) across genes for the pooled
    # clinical-indel AUC (ProteinGym pools genes there; per-gene counts too small).
    recs, pool_ys, pool_scores, n_skipped = [], [], [], 0
    # Build the assay -> row-indices map in ONE pass (O(N)); a per-assay
    # np.where(groups==g) is O(N) each -> O(N*assays) over ~2.5M rows.
    from collections import defaultdict
    g2idx = defaultdict(list)
    for _i, _g in enumerate(groups.tolist()):
        g2idx[str(_g)].append(_i)
    assays = np.unique(groups)
    if max_assays:
        assays = assays[:max_assays]
    if is_indel and max_variants_per_assay is None and skip_huge_assays:
        # Only meaningful for indels (subs share one WT table, so variant count
        # doesn't drive runtime).
        assays, n_dropped = _filter_huge_assays(assays, g2idx, skip_huge_assays)
        if n_dropped:
            print(f"skip_huge_assays={skip_huge_assays}: dropped {n_dropped} "
                  f"indel assay(s) over the variant cap")

    # Data-parallel: this process scores only its strided 1/N slice of assays.
    assays = _shard_assays(assays, assay_shard, assay_num_shards)
    if assay_num_shards > 1:
        print(f"shard {assay_shard}/{assay_num_shards} {task_key}: {len(assays)} assays")

    for _ai, g in enumerate(assays, 1):
        idx = np.asarray(g2idx[str(g)], dtype=int)
        if is_indel:
            print(f"[indel assay {_ai}/{len(assays)}] {g}: {idx.size} variants", flush=True)
        # Each long-sequence forward (per-mutant indel PLL, or per-site window) is
        # its own pass — cap variants so a multi-thousand-variant assay doesn't
        # dominate runtime. Short substitution assays share one WT table (cheap).
        if max_variants_per_assay and idx.size > max_variants_per_assay:
            idx = idx[:max_variants_per_assay]
        wt = wts[int(idx[0])]
        scores, ys, ys_bin = [], [], []

        if is_indel:
            # Indel: length-normalized PLL(mut) - PLL(WT). No mutation position to
            # window on (mutant == "N/A"), so long indels are skipped (default) or
            # truncated to the native window (opt-in).
            def _center_crop(seq, w):
                # Center crop keeps the edit region centered for most indels; the old
                # head crop (seq[:w]) truncated WT and mutant at different absolute
                # offsets, so PLL(mut)-PLL(WT) compared non-corresponding subsequences.
                if len(seq) <= w:
                    return seq
                st = (len(seq) - w) // 2
                return seq[st:st + w]
            def _mean_pll(seq):
                # Skip the wasted full-length table build for over-length seqs under
                # the default "skip" policy: pll_from_table returns None for them
                # anyway, but the old code first built+discarded a ~max_length table
                # (~21% of clinical_indels forwards, and the 44GB long-seq spikes).
                # Behavior-identical to the old code under "skip".
                if len(seq) > model_window:
                    if indel_long_policy != "truncate":
                        return None
                    s = _center_crop(seq, model_window)
                else:
                    s = seq
                tbl = masked_marginal_logprob_table(refs, tokenizer, s, device, max_length, batch_size)
                return pll_from_table(s, tbl, aa2id)
            if indel_score_mode.startswith("embedding_"):
                # ONE forward per sequence instead of `indel_pll_passes` (32) --
                # indels cannot amortize a masked forward across variants the way
                # substitutions can, so this is the whole benchmark at ~1/k the cost.
                from variant_embedding_scores import score_indel_variants

                variants = [muts[int(i)] for i in idx]
                vals = score_indel_variants(
                    wt, variants, residue_embedder,
                    arm="red" if indel_score_mode == "embedding_red" else "span",
                    model_window=model_window if indel_long_policy != "truncate" else None)
                for j, i in enumerate(idx):
                    sc = vals[j]
                    if sc is None:
                        n_skipped += 1
                        continue
                    scores.append(sc); ys.append(labels[int(i)])
                    ys_bin.append(bin_labels[int(i)] if bin_labels is not None else None)
            elif indel_score_mode in ("single_pass", "strided"):
                # Batched per-sequence scorer. seqs[0]=WT, then the variants.
                seqs = [wt] + [muts[int(i)] for i in idx]
                if indel_score_mode == "single_pass":
                    vals = single_pass_pll_table(
                        refs, tokenizer, seqs, aa2id, model_window, device,
                        max_length, batch_size, long_policy=indel_long_policy)
                else:
                    vals = strided_masked_pll_table(
                        refs, tokenizer, seqs, aa2id, model_window, device,
                        max_length, n_passes=indel_pll_passes,
                        batch_size=batch_size,
                        long_policy=indel_long_policy,
                        progress_label=f"{_ai}/{len(assays)} {str(g)[:24]}")
                wt_pll = vals[0]
                for j, i in enumerate(idx):
                    mp = vals[j + 1]
                    sc = None if (wt_pll is None or mp is None) else (mp - wt_pll)
                    if sc is None:
                        n_skipped += 1
                        continue
                    scores.append(sc); ys.append(labels[int(i)])
                    ys_bin.append(bin_labels[int(i)] if bin_labels is not None else None)
            else:  # masked_pll: original L-masked-forwards-per-variant path
                wt_pll = _mean_pll(wt)
                for i in idx:
                    mut = muts[int(i)]
                    sc = None
                    if wt_pll is not None:
                        mut_pll = _mean_pll(mut)
                        sc = None if mut_pll is None else (mut_pll - wt_pll)
                    if sc is None:
                        n_skipped += 1
                        continue
                    scores.append(sc); ys.append(labels[int(i)])
                    ys_bin.append(bin_labels[int(i)] if bin_labels is not None else None)
        else:
            # Substitution: mutation-centered optimal-window masked-marginals,
            # scored as sum logP(mut)-logP(wt) over the diffed positions. Compute
            # logP ONLY at positions some variant mutates (the union). For a dense
            # DMS that is every residue (== a full table, no waste); for SPARSE
            # clinical assays (~25 variants over a long WT) it is a handful, vs a
            # full L-forward table per assay — ~1.8M needless forwards across the
            # 2525 clinical assays otherwise. windowed_logp_table centers a window
            # on each position; for short WT the window is the whole sequence, so
            # the value equals the full-table row (verified) — one path, short+long.
            wt_bytes = np.frombuffer(wt.encode("latin-1"), dtype=np.uint8)
            need = set()
            for i in idx:
                mut = muts[int(i)]
                if len(mut) != len(wt):
                    continue
                mb = np.frombuffer(mut.encode("latin-1"), dtype=np.uint8)
                need.update(int(p) for p in np.nonzero(wt_bytes != mb)[0])
            pos_cache = windowed_logp_table(
                refs, tokenizer, wt, sorted(need), model_window,
                device, max_length, batch_size)

            def logp_at(p):
                return pos_cache.get(p)

            for i in idx:
                mut = muts[int(i)]
                if len(wt) != len(mut):
                    n_skipped += 1  # a true indel mis-filed in a substitution task
                    continue
                sc = _score_substitution_windowed(wt, mut, logp_at, aa2id, wt_bytes=wt_bytes)
                if sc is None:
                    n_skipped += 1
                    continue
                scores.append(sc); ys.append(labels[int(i)])
                ys_bin.append(bin_labels[int(i)] if bin_labels is not None else None)

        if len(scores) < 2:
            continue
        if cfg.problem_type == "regression":
            # ProteinGym PRIMARY for DMS is Spearman on continuous fitness.
            r, _ = spearmanr(ys, scores)
            primary = float(r) if not np.isnan(r) else 0.0
            # SECONDARY AUC via the OFFICIAL DMS_score_bin only. No median-split
            # fallback: ProteinGym drops single-class assays (emits NaN) rather
            # than inventing a label, so a fabricated label would drift the AUC
            # mean from the board. Assay with no usable bin label -> auc=None.
            sec_auc = None
            if bin_labels is not None:
                pairs = [(b, s) for b, s in zip(ys_bin, scores)
                         if b is not None and not np.isnan(float(b))]
                if len(pairs) == len(scores):
                    yb = [int(round(float(b))) for b, _ in pairs]
                    if len(set(yb)) == 2:
                        try:
                            sec_auc = float(roc_auc_score(yb, [s for _, s in pairs]))
                        except ValueError:
                            sec_auc = None
            recs.append({"assay": str(g), "primary": primary, "auc": sec_auc, "n": len(scores)})
        else:
            # Clinical pathogenicity: pathogenic = deleterious = lower log P(mut),
            # so negate (pathogenic ranks high). Feed EVERY gene into the pool (a
            # single-class gene still contributes to the pooled-both-class AUC used
            # for clinical INDELS); keep a per-gene AUC for clinical SUBS.
            neg = [-s for s in scores]
            pool_ys.extend(ys); pool_scores.extend(neg)
            try:
                a = float(roc_auc_score(ys, neg))
                recs.append({"assay": str(g), "primary": a, "auc": a, "n": len(scores)})
            except ValueError:
                pass  # single-class gene: no per-gene AUC, but it still feeds the pool

    return {"task": task_key, "recs": recs,
            "pool_ys": pool_ys, "pool_scores": pool_scores, "n_skipped": n_skipped}


DMS_REF_DEFAULT = str(Path(__file__).resolve().parent / "data" / "proteingym_ref" / "DMS_substitutions.csv")


def _hier_mean(recs, field, ref_path):
    """ProteinGym hierarchical mean (performance_DMS_benchmarks.py:296-309):
    per-assay -> mean within (UniProt_ID, functional category) -> mean within
    category -> mean across the 5 categories. Unweights proteins/functions with
    many assays. Needs DMS_substitutions.csv (DMS_id -> UniProt_ID,
    coarse_selection_type). Returns None if the ref is unavailable."""
    import pandas as pd
    ref = pd.read_csv(ref_path).set_index("DMS_id")[["UniProt_ID", "coarse_selection_type"]]
    rows = []
    for a in recs:
        v = a.get(field)
        if v is None or a["assay"] not in ref.index:
            continue
        rows.append((ref.loc[a["assay"], "UniProt_ID"],
                     ref.loc[a["assay"], "coarse_selection_type"], float(v)))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["up", "cat", "v"])
    per_cat = df.groupby(["up", "cat"])["v"].mean().reset_index().groupby("cat")["v"].mean()
    return float(per_cat.mean())


def aggregate_proteingym(result, dms_ref_path=DMS_REF_DEFAULT):
    """Leaderboard-faithful aggregation of per-assay records:
      DMS subs   -> hierarchical mean (UniProt_ID -> functional category) [board].
      DMS indels -> flat assay-mean (own-method MLM-PLL; not on the board).
      Clinical subs   -> per-gene mean AUC [board].
      Clinical indels -> ONE pooled AUC across all genes [board].
    Also reports the flat mean alongside the hierarchical for transparency."""
    task = result["task"]; cfg = TASKS[task]; recs = result["recs"]
    out = {"assays": len(recs), "variants_skipped": result["n_skipped"]}
    if task == "proteingym_clinical_indels_zeroshot":
        try:
            out["eval_auc"] = float(roc_auc_score(result["pool_ys"], result["pool_scores"]))
        except ValueError:
            out["eval_auc"] = None
        out["aggregation"] = "pooled"; out["n_pooled_variants"] = len(result["pool_ys"])
        return out
    if cfg.problem_type == "regression":
        prim = [a["primary"] for a in recs]
        aucs = [a["auc"] for a in recs if a["auc"] is not None]
        out["eval_spearman_flat"] = float(np.mean(prim)) if prim else None
        out["eval_auc_flat"] = float(np.mean(aucs)) if aucs else None
        if task == "proteingym_dms_substitutions_zeroshot":
            try:
                out["eval_spearman"] = _hier_mean(recs, "primary", dms_ref_path)
                out["eval_auc"] = _hier_mean(recs, "auc", dms_ref_path)
                out["aggregation"] = "hierarchical(UniProt->category)"
            except Exception as e:  # ref CSV missing/bad -> fall back to flat, say so
                out["eval_spearman"] = out["eval_spearman_flat"]
                out["eval_auc"] = out["eval_auc_flat"]
                out["aggregation"] = f"flat(hierarchical-unavailable: {type(e).__name__})"
        else:  # dms_indels: own-method, flat
            out["eval_spearman"] = out["eval_spearman_flat"]
            out["eval_auc"] = out["eval_auc_flat"]
            out["aggregation"] = "flat(own-method)"
        return out
    # clinical subs: per-gene mean AUC
    prim = [a["primary"] for a in recs if a["primary"] is not None]
    out["eval_auc"] = float(np.mean(prim)) if prim else None
    out["aggregation"] = "per_gene_mean"
    return out


# --- data-parallel sharding (score disjoint assay strides on N GPUs) ----------

def _shard_assays(assays, shard, num_shards):
    """Strided slice of the (sorted) assay array. Strided (not contiguous) so each
    shard gets a mix of big/small assays -> balanced wall-clock. nshards<=1 is the
    identity (unsharded), preserving the default single-GPU behaviour exactly."""
    if num_shards <= 1:
        return assays
    return assays[shard::num_shards]


def _merge_results(results):
    """Union disjoint-shard ``result`` dicts back into one. aggregate_proteingym is
    a pure function of (recs, pool_*, n_skipped) and shards score DISJOINT assays,
    so aggregating the union == aggregating the unsharded run, exactly."""
    if not results:
        raise ValueError("no shard results to merge")
    merged = {"task": results[0]["task"], "recs": [],
              "pool_ys": [], "pool_scores": [], "n_skipped": 0}
    for r in results:
        merged["recs"].extend(r["recs"])
        merged["pool_ys"].extend(r["pool_ys"])
        merged["pool_scores"].extend(r["pool_scores"])
        merged["n_skipped"] += int(r["n_skipped"])
    return merged


def _np_default(o):
    """JSON encoder fallback for numpy scalars/arrays in shard sidecars."""
    if hasattr(o, "tolist"):
        return o.tolist()
    return float(o)


def build_record(*, checkpoint, task, model_type, metric, n_eval, notes, ctx=None):
    """Assemble one JSONL record, stamped with the code that produced it.

    Two records can agree on model, task and metric and still be different
    measurements: the indel arm and its ``k`` change the score, and so does a code
    change (the RED arm's sign was corrected in one commit). ``timestamp_iso``
    cannot express either, so the scorer settings ride in ``metric`` and the
    git description in ``code_version``.
    """
    from benchmark_utils import code_version

    metric = dict(metric)
    ctx = ctx or {}
    if task in INDEL_ZS and "indel_score_mode" in ctx:
        metric.setdefault("indel_score_mode", ctx["indel_score_mode"])
        metric.setdefault(
            "indel_pll_passes",
            ctx.get("indel_pll_passes") if ctx["indel_score_mode"] == "strided" else None,
        )
    return {
        "checkpoint": checkpoint, "task": task, "mode": "mlm_zeroshot",
        "split": "zeroshot", "model_type": model_type,
        "metric": metric, "n_train": 0, "n_eval": n_eval,
        "notes": notes, "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code_version": code_version(),
    }


def _build_and_write(out, result, ctx, dms_ref):
    """Build the leaderboard metric_dict + JSONL record from a (possibly merged)
    ``result`` and a small ``ctx`` (model_window/strategy/indel-mode/etc). Shared by
    the normal single-process path and the --merge_only path so neither re-scores
    nor diverges. Returns (rec, metric_dict, jsonl_path)."""
    task_key = result["task"]
    metric_dict = aggregate_proteingym(result, dms_ref_path=dms_ref)
    metric_dict["window_strategy"] = ctx["window_strategy"]
    metric_dict["model_window"] = ctx["model_window"]
    if ctx.get("rope_extended_to"):
        metric_dict["rope_extended_to"] = ctx["rope_extended_to"]
    if task_key in INDEL_ZS:
        metric_dict["indel_score_mode"] = ctx["indel_score_mode"]
        metric_dict["indel_pll_passes"] = (ctx["indel_pll_passes"]
                                           if ctx["indel_score_mode"] == "strided" else None)
        metric_dict["leaderboard_comparable"] = False
        metric_dict["method"] = {
            "strided":     f"strided masked-PLL approx (own-method, k={ctx['indel_pll_passes']})",
            "masked_pll":  "exact masked-PLL (own-method)",
            "single_pass": "single-pass unmasked mean-logp (own-method; leakage, reference-only)",
            "embedding_span": "edit-span pooled embedding distance (own-method, 1 forward/variant)",
            "embedding_red": "residue-diversity delta (own-method, 1 forward/variant)",
        }[ctx["indel_score_mode"]]
    rec = build_record(
        checkpoint=ctx["checkpoint"], task=task_key, model_type=ctx.get("model_type"),
        metric=metric_dict, n_eval=len(result["recs"]), notes=ctx.get("notes", ""), ctx=ctx,
    )
    jsonl_path = write_jsonl_record(out, "mlm_zeroshot", ctx["checkpoint"], rec)
    (out / f"per_assay_{task_key}.json").write_text(json.dumps(
        {"task": task_key, "checkpoint": ctx["checkpoint"], "recs": result["recs"]},
        default=_np_default))
    return rec, metric_dict, jsonl_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--tasks", nargs="*", default=ALL_ZS,
                    help="default: all 4 ProteinGym benchmarks (subs via masked-"
                         "marginal, indels via pseudo-log-likelihood)")
    ap.add_argument("--output_dir", default="results/bench/discrim_mlmzs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=None,
                    help="token cap for a single forward (default: the model's "
                         "native context, auto-detected — 1024 Proteva / 2048 "
                         "AMPLIFY; substitutions are windowed to never exceed it).")
    ap.add_argument("--model_window", type=int, default=None,
                    help="residue window for long substitutions (default: "
                         "encoder.max_position-2, i.e. native context minus specials)")
    ap.add_argument("--dms_ref", default=DMS_REF_DEFAULT,
                    help="ProteinGym DMS_substitutions.csv for the hierarchical "
                         "DMS aggregation (DMS_id -> UniProt_ID, functional category)")
    ap.add_argument("--max_assays", type=int, default=None)
    ap.add_argument("--skip_huge_assays", type=int, default=None, metavar="N",
                    help="Skip indel assays with more than N variants. Off by default "
                         "(run all assays, leaderboard-faithful). In very-fast mode "
                         "pass --skip_huge_assays 10000 to skip CAPSD_AAV2S "
                         "(225k + 25k variants) which alone account for ~80%% of "
                         "DMS-indels compute. Has no effect on substitution tasks.")
    ap.add_argument("--max_variants_per_assay", type=int, default=None,
                    help="cap variants/assay (default None = ALL variants, "
                         "leaderboard-faithful). Substitutions share one WT table so "
                         "all-variants is ~free; only per-mutant indel PLL + per-site "
                         "windowed forwards scale with this. Set a value only to bound "
                         "a slow indel run.")
    ap.add_argument("--indel_long_policy", choices=["skip", "truncate"], default="skip",
                    help="indels longer than the window: skip (default) or truncate "
                         "to the first model_window residues (approximation)")
    ap.add_argument("--indel_score_mode", choices=list(INDEL_SCORE_MODES),
                    default="strided",
                    help="indel scorer. strided (default): leakage-free few-pass masked-PLL, "
                         "mask every k-th position over --indel_pll_passes forwards. Validated "
                         "2026-06-18 vs exact masked_pll: k=32 matches within 0.02 Spearman at "
                         "~50x speed (~7h vs ~weeks on CAPSD). masked_pll: L masked forwards/"
                         "variant, exact (correct, very slow). single_pass: 1 unmasked forward "
                         "-- FAST but naive leakage breaks ranking (Spearman 0.50->0.31); avoid. "
                         "embedding_span / embedding_red: ONE forward per sequence (~1/k the "
                         "cost of strided) reading the mutant's residue embeddings -- pooled "
                         "over the derived edit span, or residue-diversity delta. Cheap; "
                         "accuracy on proteins not yet benchmarked.")
    ap.add_argument("--indel_pll_passes", type=int, default=16,
                    help="strided mode: forward passes per sequence (k), and the whole "
                         "indel cost is linear in it. Each pass masks every k-th residue; "
                         "larger k -> closer to exact masked_pll (k>=L reproduces it). "
                         "Default 16, calibrated on ESM-C 300M over 2 DMS-indel assays: "
                         "k=16 is within 0.012 Spearman of k=32 for half the compute "
                         "(0.331/0.755 vs 0.343/0.756), while k=8 loses up to 0.07 "
                         "(0.271/0.729). See docs/ADVANCED.md.")
    ap.add_argument("--rope_extrapolate", action="store_true",
                    help="OPT-IN: grow the Proteva RoPE cache to --max_length and "
                         "score long sequences whole via extrapolation (OOD; the "
                         "default mutation-centered window stays in-distribution)")
    ap.add_argument("--bf16", action="store_true",
                    help="load model in bfloat16 (forwards bf16; logits upcast to fp32 "
                         "before log_softmax, so scores are safe). ~1.5-2x faster + half "
                         "memory on indels — own-method, fp32 weights not needed.")
    ap.add_argument("--notes", default="")
    ap.add_argument("--assay_shard", type=int, default=0,
                    help="this shard's index in [0, --assay_num_shards) for data-parallel scoring")
    ap.add_argument("--assay_num_shards", type=int, default=1,
                    help="GPU shard count; each process scores a strided 1/N slice of every task's "
                         "assays. 1 = unsharded (default, unchanged behaviour). Shards write "
                         "_shard_<task>__<i>of<N>.json sidecars; run --merge_only after to re-aggregate.")
    ap.add_argument("--merge_only", action="store_true",
                    help="don't score: read this run's _shard_*.json sidecars, union them, and write the "
                         "final leaderboard JSONL (re-aggregates on the union; no model/GPU needed).")
    args = ap.parse_args(argv)

    # Subdir so write_jsonl_record(.parent) resolves back to output_dir, matching the
    # finetune_sequence/residue pattern (picked up by collect_bench_results' glob).
    out = Path(args.output_dir) / f"mlm_zs_{safe_ckpt(args.model_name)}"
    out.mkdir(parents=True, exist_ok=True)

    # --merge_only: fold the per-shard sidecars into the final JSONL, no scoring/GPU.
    if args.merge_only:
        n = args.assay_num_shards
        # COMPLETENESS GATE: every shard drops a _shard_done__<i>of<n>.json marker
        # when it finishes (even if it scored zero assays). If fewer than n markers
        # exist, a shard CRASHED/was killed mid-run — merging the surviving sidecars
        # would silently write a leaderboard metric over a FRACTION of the assays.
        # Refuse, so the caller re-runs rather than trusting a partial number.
        done = sorted(out.glob(f"_shard_done__*of{n}.json"))
        if len(done) != n:
            raise SystemExit(
                f"--merge_only: found {len(done)}/{n} shard-done markers "
                f"({[d.name for d in done]}) — a shard crashed or did not finish; "
                "refusing to write a PARTIAL result. Re-run the shards."
            )
        for task_key in args.tasks:
            if task_key not in TASKS:
                continue
            shard_files = sorted(out.glob(f"_shard_{task_key}__*of{n}.json"))
            if not shard_files:
                # All n shards finished (gate passed) but none scored this task ->
                # genuinely no scorable assays. Correct to skip (not a partial).
                print(f"merge {task_key}: no scorable assays across all {n} shards")
                continue
            payloads = [json.loads(p.read_text()) for p in shard_files]
            merged = _merge_results([p["result"] for p in payloads])
            _rec, metric_dict, jsonl_path = _build_and_write(out, merged, payloads[0]["ctx"], args.dms_ref)
            prim_key = "eval_spearman" if TASKS[task_key].problem_type == "regression" else "eval_auc"
            print(f"merge {task_key}: {prim_key}={metric_dict.get(prim_key)} "
                  f"over {len(merged['recs'])} assays from {len(shard_files)} shards -> {jsonl_path}")
            for p in shard_files:
                p.unlink()
        for d in done:
            d.unlink()
        return 0

    # Load model using the same loader as protein_benchmark_suite / wt_tta_smoke.
    # Returns (model_obj, is_sbert, device). For AMPLIFY: model_obj=(tokenizer, model).
    from protein_benchmark_suite import load_model
    from wt_test_time_training import resolve_mlm_head

    import torch as _torch
    _dt = _torch.bfloat16 if args.bf16 else None
    model_obj, is_sbert, device = load_model(args.model_name, device=args.device,
                                             torch_dtype=_dt)
    if is_sbert:
        raise SystemExit(
            f"Model {args.model_name!r} loaded as SentenceTransformer — "
            "no MLM head available. Load a *ForMaskedLM checkpoint."
        )
    tokenizer, model = model_obj
    model.eval()
    if args.bf16:
        # Keep dense SDPA even in bf16 (avoid the fa2-varlen backend the bf16 runtime
        # would otherwise select); logits are upcast to fp32 before log_softmax.
        try:
            model.encoder.config.flash_attn_mode = "off"
        except Exception:
            pass

    # resolve_mlm_head(model, tokenizer) -> MLMHeadRefs with .forward_logits + .mask_token_id
    refs = resolve_mlm_head(model, tokenizer)
    if refs is None:
        raise SystemExit(f"No MLM head for {args.model_name}")

    # Native context (tokens) read from the model — NOT hardcoded: Proteva 1024,
    # vanilla AMPLIFY 2048. model_window = native residue capacity (native - 2
    # special tokens). max_length defaults to the native context so each model
    # uses its full trained window (AMPLIFY scores up to 2048 whole; only WT >
    # native-2 is windowed). --max_length / --model_window override this.
    native_max = _detect_native_context(model, tokenizer)
    if args.max_length is None:
        args.max_length = native_max
    model_window = args.model_window or (native_max - 2)

    # Default: stay IN-DISTRIBUTION. Long substitutions are handled by the
    # mutation-centered window (<= model_window), so the forward never exceeds the
    # trained length; cap max_length to the native context. RoPE extrapolation is
    # OPT-IN only (--rope_extrapolate) for ablation — it is OOD (trained at 1024).
    rope_extended_to = None
    if args.rope_extrapolate:
        try:
            from plm.hf.checkpoint_utils import extend_rope_cache
            prev = extend_rope_cache(model, args.max_length)
            if prev:
                rope_extended_to = int(args.max_length)
                print(f"Extended Proteva RoPE cache {prev} -> {args.max_length} (extrapolation)")
        except Exception as e:
            print(f"extend_rope_cache skipped ({e}); clamping instead")
            if native_max and args.max_length > native_max:
                args.max_length = int(native_max)
    elif native_max and args.max_length > native_max:
        args.max_length = int(native_max)
    print(f"window strategy: {'rope_extrapolation' if rope_extended_to else 'optimal-window'} "
          f"(model_window={model_window} residues, max_length={args.max_length} tokens)")

    # Static context shared by every task record (and stashed in shard sidecars so
    # --merge_only can rebuild the record without reloading the model).
    ctx = {
        "checkpoint": args.model_name,
        "model_type": getattr(getattr(model, "config", None), "model_type", None),
        "model_window": model_window,
        "window_strategy": "rope_extrapolation" if rope_extended_to else "optimal_window",
        "rope_extended_to": rope_extended_to,
        "indel_score_mode": args.indel_score_mode,
        "indel_pll_passes": args.indel_pll_passes,
        "notes": args.notes,
    }

    for task_key in args.tasks:
        if task_key not in TASKS:
            print(f"skip unknown task {task_key}")
            continue
        cfg = TASKS[task_key]
        result = _eval_task(
            task_key, refs, tokenizer, device,
            args.batch_size, args.max_length,
            max_assays=args.max_assays,
            max_variants_per_assay=args.max_variants_per_assay,
            model_window=model_window,
            indel_long_policy=args.indel_long_policy,
            skip_huge_assays=args.skip_huge_assays,
            indel_score_mode=args.indel_score_mode,
            indel_pll_passes=args.indel_pll_passes,
            assay_shard=args.assay_shard,
            assay_num_shards=args.assay_num_shards,
            residue_embedder=(
                _make_residue_embedder(model, tokenizer, device, args.batch_size, max_length)
                if args.indel_score_mode.startswith("embedding_") else None
            ),
        )
        if not result["recs"] and not result["pool_ys"]:
            print(f"{task_key}: no scorable assays (skipped={result['n_skipped']})")
            continue

        # Sharded: defer aggregation — write this shard's raw result, merge later.
        if args.assay_num_shards > 1:
            shard_path = out / f"_shard_{task_key}__{args.assay_shard}of{args.assay_num_shards}.json"
            shard_path.write_text(json.dumps({"result": result, "ctx": ctx}, default=_np_default))
            print(f"{task_key}: shard {args.assay_shard}/{args.assay_num_shards} wrote "
                  f"{len(result['recs'])} assays -> {shard_path.name}")
            continue

        # Leaderboard-faithful aggregation (hierarchical DMS / pooled clinical-indel
        # / per-gene clinical-subs); flat means kept alongside for transparency.
        _rec, metric_dict, jsonl_path = _build_and_write(out, result, ctx, args.dms_ref)
        prim_key = "eval_spearman" if cfg.problem_type == "regression" else "eval_auc"
        print(f"{task_key}: {prim_key}={metric_dict.get(prim_key)} auc={metric_dict.get('eval_auc')} "
              f"agg={metric_dict['aggregation']} over {len(result['recs'])} assays "
              f"(skipped={result['n_skipped']})")
        print(f"  -> JSONL: {jsonl_path}")

    # Sharded scoring: drop a DONE marker (even if this shard scored zero assays)
    # so --merge_only can verify ALL n shards finished. A crashed shard leaves no
    # marker -> merge refuses to write a partial result.
    if args.assay_num_shards > 1:
        (out / f"_shard_done__{args.assay_shard}of{args.assay_num_shards}.json").write_text(
            json.dumps({"shard": args.assay_shard, "num_shards": args.assay_num_shards,
                        "tasks": list(args.tasks)})
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
