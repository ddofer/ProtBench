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
from plm.bench._hf_finetune_common import write_jsonl_record

SUBSTITUTION_ZS = [
    "proteingym_dms_substitutions_zeroshot",
    "proteingym_clinical_substitutions_zeroshot",
]


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


def _eval_task(task_key, refs, tokenizer, device, batch_size, max_length, max_assays=None):
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

    per_assay, n_skipped = [], 0
    assays = np.unique(groups)
    if max_assays:
        assays = assays[:max_assays]

    for g in assays:
        idx = np.where(groups == g)[0]
        wt = wts[int(idx[0])]
        logP = masked_marginal_logprob_table(refs, tokenizer, wt, device, max_length, batch_size)
        scores, ys = [], []
        for i in idx:
            sc = score_substitution(wt, muts[int(i)], logP, aa2id)
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
        else:
            try:
                per_assay.append(roc_auc_score(ys, scores))
            except ValueError:
                pass

    return per_assay, n_skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--tasks", nargs="*", default=SUBSTITUTION_ZS)
    ap.add_argument("--output_dir", default="/data/proteva/plm/results/bench/discrim_mlmzs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_assays", type=int, default=None)
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

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for task_key in args.tasks:
        if task_key not in TASKS:
            print(f"skip unknown task {task_key}")
            continue
        cfg = TASKS[task_key]
        per_assay, n_skipped = _eval_task(
            task_key, refs, tokenizer, device,
            args.batch_size, args.max_length,
            max_assays=args.max_assays,
        )
        if not per_assay:
            print(f"{task_key}: no scorable assays (skipped={n_skipped})")
            continue

        metric_key = "eval_spearman" if cfg.problem_type == "regression" else "eval_auc"
        rec = {
            "checkpoint": args.model_name,
            "task": task_key,
            "mode": "mlm_zeroshot",
            "split": "zeroshot",
            "model_type": getattr(getattr(model, "config", None), "model_type", None),
            "metric": {
                metric_key: float(np.mean(per_assay)),
                "assays": len(per_assay),
                "variants_skipped_indel": n_skipped,
            },
            "n_train": 0,
            "n_eval": len(per_assay),
            "notes": args.notes,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # write_jsonl_record(out_dir, prefix, ckpt, record) -> writes to
        # out_dir.parent / f"mlm_zeroshot_{safe_ckpt(model_name)}.jsonl"
        jsonl_path = write_jsonl_record(out, "mlm_zeroshot", args.model_name, rec)
        print(
            f"{task_key}: {metric_key}={np.mean(per_assay):.4f} "
            f"over {len(per_assay)} assays (indel skipped={n_skipped})"
        )
        print(f"  -> JSONL written: {jsonl_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
