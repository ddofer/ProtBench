"""Canonical MLM masked-marginal ProteinGym zero-shot (ESM-1v style).

Substitutions only (indels skipped). Output JSONL read by collect_bench_results.

Deviations from the reference template:
- load_model_and_tokenizer does not exist; uses load_model() from
  protein_benchmark_suite which returns (model_obj, is_sbert, device).
  For AMPLIFY model_obj = (tokenizer, model).
- resolve_mlm_head(model, tokenizer) takes two arguments (confirmed in
  wt_test_time_training.py line 68). Returns MLMHeadRefs with .forward_logits
  and .mask_token_id; there is no separate mlm module.
- masked_marginal_logprob_table calls refs.forward_logits(input_ids, mask)
  instead of mlm(input_ids=...).
- write_jsonl_record signature is (out_dir, prefix, ckpt, record) — 4 args.
  Note: writes to out_dir.parent / f"{prefix}_{safe_ckpt(ckpt)}.jsonl".
- Data loading mirrors protein_benchmark_suite.prepare_data for proteingym_zeroshot:
  load with data_dir kwarg, take train_split, columns from cfg.input_map
  ("mutant"->mutated_sequence, "wt"->target_seq), label_col, group_by.
"""
from __future__ import annotations
import argparse, time
import numpy as np, torch
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from plm.bench.benchmark_tasks import TASKS
from plm.bench._hf_finetune_common import write_jsonl_record, safe_ckpt

SUBSTITUTION_ZS = [
    "proteingym_dms_substitutions_zeroshot",
    "proteingym_clinical_substitutions_zeroshot",
]
INDEL_ZS = [
    "proteingym_dms_indels_zeroshot",
    "proteingym_clinical_indels_zeroshot",
]
ALL_ZS = SUBSTITUTION_ZS + INDEL_ZS


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
def single_position_masked_logp(refs, tokenizer, seq, local_res_idx, device, max_length):
    """log-softmax vocab row for ONE masked residue (``local_res_idx``-th residue
    of ``seq``). One forward pass; used by the windowed long-sequence path.

    ``seq`` is the already-cropped window, so ``local_res_idx`` is the mutated
    position re-based into the crop. Returns the [V] log-prob vector (cpu) or None
    if the position falls outside the (truncated) token range.
    """
    enc = tokenizer(seq, truncation=True, max_length=max_length, return_tensors="pt")
    ids = enc["input_ids"][0]
    am = enc.get("attention_mask", None)
    am = am[0] if am is not None else None
    special_set = set(refs.special_ids)
    res_tok = [t for t, x in enumerate(ids.tolist()) if x not in special_set]
    if local_res_idx >= len(res_tok):
        return None
    tpos = res_tok[local_res_idx]
    b = ids.clone()
    b[tpos] = refs.mask_token_id
    b = b.unsqueeze(0).to(device)
    bm = am.unsqueeze(0).to(device) if am is not None else None
    lg = refs.forward_logits(b, bm)
    return torch.log_softmax(lg.float(), dim=-1)[0, tpos].cpu()


@torch.no_grad()
def masked_marginal_logprob_table(refs, tokenizer, wt, device, max_length=2048, batch_size=64):
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
        for j, tpos in enumerate(chunk):
            logP[start + j] = lsm[j, tpos].cpu()

    return logP


def _score_substitution_windowed(wt, mut, logp_at, aa2id):
    """Substitution score = sum over mutated positions of logP(mut) - logP(wt).

    ``logp_at(p)`` returns the [V] masked log-prob row for residue position ``p``
    (a full-table slice for short WT, or a mutation-centered window for long WT);
    returns None if that position is unscorable. Mutated positions are found by
    diffing wt vs mut, so it works for single- and multi-mutants. Returns None
    if any required position is unscorable.
    """
    s = 0.0
    for p in range(len(wt)):
        w, m = wt[p], mut[p]
        if w == m or w not in aa2id or m not in aa2id:
            continue
        row = logp_at(p)
        if row is None:
            return None
        s += float(row[aa2id[m]] - row[aa2id[w]])
    return s


def _eval_task(task_key, refs, tokenizer, device, batch_size, max_length,
               max_assays=None, max_variants_per_assay=None,
               model_window=1022, indel_long_policy="skip"):
    cfg = TASKS[task_key]
    is_indel = task_key in INDEL_ZS
    from datasets import load_dataset

    # Mirror protein_benchmark_suite.prepare_data for proteingym_zeroshot:
    # pass data_dir kwarg and take the train split.
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
    muts    = data[mut_col]
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
    auc_label_source = "dms_score_bin" if bin_labels is not None else "median"

    # Build aa->token_id map for the 20 canonical amino acids
    aa2id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    per_assay, per_assay_auc, n_skipped = [], [], 0
    assays = np.unique(groups)
    if max_assays:
        assays = assays[:max_assays]

    for g in assays:
        idx = np.where(groups == g)[0]
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
            def _mean_pll(seq):
                s = seq[:model_window] if indel_long_policy == "truncate" else seq
                tbl = masked_marginal_logprob_table(refs, tokenizer, s, device, max_length, batch_size)
                return pll_from_table(s, tbl, aa2id)
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
            # Substitution: mutation-centered optimal-window masked-marginals.
            # Short WT -> one shared full table; long WT -> per-site windowed
            # forward (in-distribution, scores every variant — no skip).
            short = len(wt) <= model_window
            full_table = (masked_marginal_logprob_table(refs, tokenizer, wt, device, max_length, batch_size)
                          if short else None)
            pos_cache = {}

            def logp_at(p):
                if short:
                    return full_table[p] if p < full_table.shape[0] else None
                if p not in pos_cache:
                    st, en = get_optimal_window(p, len(wt), model_window)
                    pos_cache[p] = single_position_masked_logp(
                        refs, tokenizer, wt[st:en], p - st, device, max_length)
                return pos_cache[p]

            for i in idx:
                mut = muts[int(i)]
                if len(wt) != len(mut):
                    n_skipped += 1  # a true indel mis-filed in a substitution task
                    continue
                sc = _score_substitution_windowed(wt, mut, logp_at, aa2id)
                if sc is None:
                    n_skipped += 1
                    continue
                scores.append(sc); ys.append(labels[int(i)])
                ys_bin.append(bin_labels[int(i)] if bin_labels is not None else None)

        if len(scores) < 2:
            continue
        if cfg.problem_type == "regression":
            # ProteinGym leaderboard PRIMARY metric for DMS (subs AND indels) is
            # Spearman on continuous fitness — directly comparable to the board.
            r, _ = spearmanr(ys, scores)
            per_assay.append(float(r) if not np.isnan(r) else 0.0)
            # SECONDARY AUC (ProteinGym also reports this). Use the OFFICIAL
            # DMS_score_bin labels when available (leaderboard-faithful); fall back
            # to a per-assay median split only if the bin column is absent/degenerate.
            yb = [b for b in ys_bin if b is not None]
            if len(yb) == len(scores) and 0 < int(np.nansum(yb)) < len(yb):
                try:
                    per_assay_auc.append(float(roc_auc_score(yb, scores)))
                except ValueError:
                    pass
            else:
                arr_ys = np.array(ys)
                binary = (arr_ys > np.median(arr_ys)).astype(int)
                if 0 < int(binary.sum()) < len(binary):
                    try:
                        per_assay_auc.append(float(roc_auc_score(binary, scores)))
                    except ValueError:
                        pass
        else:
            # Clinical pathogenicity: pathogenic = deleterious = lower log P(mut).
            # Negate so pathogenic variants rank high (ProteinGym convention).
            try:
                per_assay.append(roc_auc_score(ys, [-s for s in scores]))
            except ValueError:
                pass

    return per_assay, per_assay_auc, n_skipped, auc_label_source


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--tasks", nargs="*", default=ALL_ZS,
                    help="default: all 4 ProteinGym benchmarks (subs via masked-"
                         "marginal, indels via pseudo-log-likelihood)")
    ap.add_argument("--output_dir", default="/data/proteva/plm/results/bench/discrim_mlmzs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=1024,
                    help="token cap for a single forward; substitutions never "
                         "exceed it (mutation-centered window), so 1024 = native.")
    ap.add_argument("--model_window", type=int, default=None,
                    help="residue window for long substitutions (default: "
                         "encoder.max_position-2, i.e. native context minus specials)")
    ap.add_argument("--max_assays", type=int, default=None)
    ap.add_argument("--max_variants_per_assay", type=int, default=200,
                    help="cap variants/assay (bounds per-mutant indel PLL + per-site "
                         "windowed forwards; short-WT substitutions share one table)")
    ap.add_argument("--indel_long_policy", choices=["skip", "truncate"], default="skip",
                    help="indels longer than the window: skip (default) or truncate "
                         "to the first model_window residues (approximation)")
    ap.add_argument("--rope_extrapolate", action="store_true",
                    help="OPT-IN: grow the Proteva RoPE cache to --max_length and "
                         "score long sequences whole via extrapolation (OOD; the "
                         "default mutation-centered window stays in-distribution)")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    # Load model using the same loader as protein_benchmark_suite / wt_tta_smoke.
    # Returns (model_obj, is_sbert, device). For AMPLIFY: model_obj=(tokenizer, model).
    from protein_benchmark_suite import load_model
    from wt_test_time_training import resolve_mlm_head

    model_obj, is_sbert, device = load_model(args.model_name, device=args.device)
    if is_sbert:
        raise SystemExit(
            f"Model {args.model_name!r} loaded as SentenceTransformer — "
            "no MLM head available. Load a *ForMaskedLM checkpoint."
        )
    tokenizer, model = model_obj
    model.eval()

    # resolve_mlm_head(model, tokenizer) -> MLMHeadRefs with .forward_logits + .mask_token_id
    refs = resolve_mlm_head(model, tokenizer)
    if refs is None:
        raise SystemExit(f"No MLM head for {args.model_name}")

    # Native residue capacity = encoder.max_position - 2 special tokens (Proteva
    # trained at 1024 -> 1022; AMPLIFY has no cap -> default to args.max_length-2).
    enc_cfg = getattr(getattr(model, "encoder", None), "config", None)
    native_max = getattr(enc_cfg, "max_position", None)
    model_window = args.model_window or ((int(native_max) - 2) if native_max else (args.max_length - 2))

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

    # Use a subdir so write_jsonl_record(.parent) resolves back to output_dir,
    # matching the pattern in finetune_sequence/residue (picked up by collect glob).
    out = Path(args.output_dir) / f"mlm_zs_{safe_ckpt(args.model_name)}"
    out.mkdir(parents=True, exist_ok=True)

    for task_key in args.tasks:
        if task_key not in TASKS:
            print(f"skip unknown task {task_key}")
            continue
        cfg = TASKS[task_key]
        per_assay, per_assay_auc, n_skipped, auc_label_source = _eval_task(
            task_key, refs, tokenizer, device,
            args.batch_size, args.max_length,
            max_assays=args.max_assays,
            max_variants_per_assay=args.max_variants_per_assay,
            model_window=model_window,
            indel_long_policy=args.indel_long_policy,
        )
        if not per_assay:
            print(f"{task_key}: no scorable assays (skipped={n_skipped})")
            continue

        # ProteinGym primary: Spearman for DMS (regression), AUC for clinical (binary).
        metric_key = "eval_spearman" if cfg.problem_type == "regression" else "eval_auc"
        metric_dict = {
            metric_key: float(np.mean(per_assay)),
            "assays": len(per_assay),
            "variants_skipped": n_skipped,
            "window_strategy": "rope_extrapolation" if rope_extended_to else "optimal_window",
            "model_window": model_window,
        }
        if per_assay_auc:
            metric_dict["eval_auc"] = float(np.mean(per_assay_auc))
            metric_dict["auc_label_source"] = auc_label_source if cfg.problem_type == "regression" else "annotation"
        if rope_extended_to:
            metric_dict["rope_extended_to"] = rope_extended_to  # >1024 = extrapolation
        rec = {
            "checkpoint": args.model_name,
            "task": task_key,
            "mode": "mlm_zeroshot",
            "split": "zeroshot",
            "model_type": getattr(getattr(model, "config", None), "model_type", None),
            "metric": metric_dict,
            "n_train": 0,
            "n_eval": len(per_assay),
            "notes": args.notes,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # write_jsonl_record(out_dir, prefix, ckpt, record) -> writes to
        # out_dir.parent / f"mlm_zeroshot_{safe_ckpt(model_name)}.jsonl"
        jsonl_path = write_jsonl_record(out, "mlm_zeroshot", args.model_name, rec)
        auc_str = f" auc={metric_dict['eval_auc']:.4f}" if per_assay_auc else ""
        print(
            f"{task_key}: {metric_key}={np.mean(per_assay):.4f}{auc_str} "
            f"over {len(per_assay)} assays (indel skipped={n_skipped})"
        )
        print(f"  -> JSONL written: {jsonl_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
