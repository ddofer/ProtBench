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
    """Pseudo-log-likelihood of ``seq`` from its own masked-marginal table.

    PLL = sum_i log P(seq[i] | seq with position i masked), over canonical-AA
    positions. This is the encoder/masked-LM analogue of a sequence likelihood
    (ESM-1v / ESM2 ProteinGym indel scoring): it needs no position alignment, so
    an indel score is simply ``PLL(mutant) - PLL(WT)`` even when lengths differ.
    Returns None if the sequence was truncated past the table.
    """
    if len(seq) > logP.shape[0]:
        return None  # truncated; skip variant
    s = 0.0
    for i, a in enumerate(seq):
        if a in aa2id:
            s += float(logP[i, aa2id[a]])
    return s


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


def _eval_task(task_key, refs, tokenizer, device, batch_size, max_length,
               max_assays=None, max_variants_per_assay=None):
    cfg = TASKS[task_key]
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

    # Build aa->token_id map for the 20 canonical amino acids
    aa2id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    per_assay, per_assay_auc, n_skipped = [], [], 0
    assays = np.unique(groups)
    if max_assays:
        assays = assays[:max_assays]

    for g in assays:
        idx = np.where(groups == g)[0]
        # Indel variants each need their OWN masked table (a per-mutant forward
        # pass over its length) — cap them so a 4500-mutant DMS-indel assay
        # doesn't dominate runtime. Substitutions share the WT table (cheap), so
        # the cap only bites the indel path.
        if max_variants_per_assay and idx.size > max_variants_per_assay:
            idx = idx[:max_variants_per_assay]
        wt = wts[int(idx[0])]
        logP = masked_marginal_logprob_table(refs, tokenizer, wt, device, max_length, batch_size)
        wt_pll = pll_from_table(wt, logP, aa2id)
        scores, ys = [], []
        for i in idx:
            mut = muts[int(i)]
            if len(wt) == len(mut):
                sc = score_substitution(wt, mut, logP, aa2id)
            elif wt_pll is None:
                sc = None  # WT truncated; can't form a PLL baseline
            else:
                # Indel: PLL(mutant) - PLL(WT) using the mutant's own masked
                # table. Length-agnostic, so deletions/insertions are scorable.
                mut_logP = masked_marginal_logprob_table(
                    refs, tokenizer, mut, device, max_length, batch_size)
                mut_pll = pll_from_table(mut, mut_logP, aa2id)
                sc = None if mut_pll is None else (mut_pll - wt_pll)
            if sc is None:
                n_skipped += 1
                continue
            scores.append(sc)
            ys.append(labels[int(i)])
        if len(scores) < 2:
            continue
        if cfg.problem_type == "regression":
            r, _ = spearmanr(ys, scores)
            per_assay.append(float(r) if not np.isnan(r) else 0.0)
            # AUC via median binarization — ProteinGym website convention
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

    return per_assay, per_assay_auc, n_skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--tasks", nargs="*", default=ALL_ZS,
                    help="default: all 4 ProteinGym benchmarks (subs via masked-"
                         "marginal, indels via pseudo-log-likelihood)")
    ap.add_argument("--output_dir", default="/data/proteva/plm/results/bench/discrim_mlmzs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_assays", type=int, default=None)
    ap.add_argument("--max_variants_per_assay", type=int, default=200,
                    help="cap variants/assay (bounds the per-mutant PLL forward "
                         "passes on large DMS-indel assays; subs are unaffected)")
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

    # Clamp max_length to the encoder's positional capacity. Proteva precomputes
    # a RoPE cache for encoder.config.max_position (=1024); a longer sequence
    # hits a cos/sin size mismatch in _apply_rope. AMPLIFY has no such cap.
    enc_cfg = getattr(getattr(model, "encoder", None), "config", None)
    max_pos = getattr(enc_cfg, "max_position", None)
    if max_pos and args.max_length > max_pos:
        print(f"Clamping max_length {args.max_length} -> {max_pos} (encoder.max_position)")
        args.max_length = int(max_pos)

    # Use a subdir so write_jsonl_record(.parent) resolves back to output_dir,
    # matching the pattern in finetune_sequence/residue (picked up by collect glob).
    out = Path(args.output_dir) / f"mlm_zs_{safe_ckpt(args.model_name)}"
    out.mkdir(parents=True, exist_ok=True)

    for task_key in args.tasks:
        if task_key not in TASKS:
            print(f"skip unknown task {task_key}")
            continue
        cfg = TASKS[task_key]
        per_assay, per_assay_auc, n_skipped = _eval_task(
            task_key, refs, tokenizer, device,
            args.batch_size, args.max_length,
            max_assays=args.max_assays,
            max_variants_per_assay=args.max_variants_per_assay,
        )
        if not per_assay:
            print(f"{task_key}: no scorable assays (skipped={n_skipped})")
            continue

        metric_key = "eval_spearman" if cfg.problem_type == "regression" else "eval_auc"
        metric_dict = {
            metric_key: float(np.mean(per_assay)),
            "assays": len(per_assay),
            "variants_skipped_indel": n_skipped,
        }
        if per_assay_auc:
            metric_dict["eval_auc"] = float(np.mean(per_assay_auc))
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
