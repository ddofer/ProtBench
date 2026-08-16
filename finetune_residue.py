"""Residue-level (token-classification) fine-tuning for protein LMs.

Wires AMPLIFY / ESM-style encoders to HF ``AutoModelForTokenClassification``
+ ``Trainer`` for three residue tasks: SS3 (NetSurfP-SS3), intrinsic
disorder (same dataset, ``disorder`` column), and signal peptides
(SignalP6 via SaProtHub). Mirrors the CLI style of ``protein_benchmark_suite.py``.

See ``docs/DATASETS.md`` for citation details and
verified row counts.
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import transformers
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
)

# Allow ``python plm/bench/finetune_residue.py`` from anywhere; also be
# importable via ``import finetune_residue`` from inside plm/bench/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hf_finetune_common import (  # noqa: E402
    add_common_finetune_args,
    align_labels_with_tokens,
    apply_finetune_mode,
    build_training_args,
    decode_csv_label,
    decode_string_label,
    keep_logits_only,
    load_encoder_for_head,
    load_tokenizer,
    resolve_early_stopping,
    resolve_local_dataset_path,
    safe_ckpt,
    write_jsonl_record,
)
from benchmark_tasks import TASKS, TaskConfig  # noqa: E402

logger = logging.getLogger(__name__)

# Task-specific label alphabets / class names.
_SS3_ALPHABET = "HEC"
# Must match protein_benchmark_suite._SS8_ALPHABET so a fine-tuned SS8 model and
# a frozen SS8 probe report against the same class ids. `D` (unassigned) is not
# a class in either; it becomes -100, which HF Trainer already ignores.
_SS8_ALPHABET = "GHIBESTC"
_DISORDER_ALPHABET = "01"

_RESIDUE_TASKS = (
    "ss3",
    "ss8",
    "disorder",
    "signal_peptide",
    "conservation_flip",
    "disprot",
)


def _decode_disorder_label(label_str: Any) -> List[int]:
    """Decode per-residue disorder labels from the NetSurfP-SS3 dataset.

    The ``disorder`` column is stored as a stringified Python list of floats,
    e.g. ``"['0.0', '1.0', '1.0', ...]"``.  Each value is either 0.0 (ordered)
    or 1.0 (disordered) — round to int to get 0/1 class labels.

    Falls back to the simple ``_DISORDER_ALPHABET`` char-index decoder if the
    raw value is already a compact '0'/'1' string (backwards-compatibility).
    """
    if isinstance(label_str, (list, tuple)):
        # Already a real list (future-proof if HF fixes the stored type)
        return [int(round(float(x))) for x in label_str]
    s = str(label_str).strip()
    if s.startswith("["):
        # Stringified list: "['0.0', '1.0', ...]"
        try:
            items = ast.literal_eval(s)
            return [int(round(float(x))) for x in items]
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"_decode_disorder_label: failed to parse '['-prefixed string "
                f"(first 120 chars): {s[:120]!r}"
            ) from exc
    # Fallback: compact alphabet string '010110...'
    return [_DISORDER_ALPHABET.index(c) for c in s if c in _DISORDER_ALPHABET]


def _decode_labels(task: str, label_str: Any) -> List[int]:
    if task == "ss3":
        return decode_string_label(str(label_str), _SS3_ALPHABET)
    if task == "ss8":
        # Unassigned residues (GleghornLab/SS8's `D`) become -100, HF Trainer's
        # ignore index, rather than being dropped -- dropping would shorten the
        # list and shift every later label against its token.
        return [
            _SS8_ALPHABET.index(c) if c in _SS8_ALPHABET else -100
            for c in str(label_str)
        ]
    if task == "disorder":
        return _decode_disorder_label(label_str)
    if task == "disprot":
        # DisProt/CAID: disorder_labels is already a list[int] 0/1 (prep_disprot.py).
        if isinstance(label_str, (list, tuple)):
            return [int(round(float(x))) for x in label_str]
        return _decode_disorder_label(label_str)  # defensive (stringified fallback)
    if task == "signal_peptide":
        return decode_csv_label(str(label_str))
    if task == "conservation_flip":
        # Per-residue integer class labels: already-tokenized list, or a
        # comma-separated / digit string (mirrors the linear suite's decoder).
        if isinstance(label_str, (list, tuple)):
            return [int(x) for x in label_str]
        s = str(label_str)
        if "," in s:
            return [int(t) for t in s.split(",") if t.strip()]
        return [int(c) for c in s if c.isdigit()]
    raise ValueError(f"Unknown residue task: {task}")


def _build_label_meta(task: str, all_label_lists: List[List[int]]) -> Dict[str, Any]:
    if task == "ss3":
        names = list(_SS3_ALPHABET)
    elif task == "ss8":
        names = list(_SS8_ALPHABET)
    elif task in ("disorder", "disprot"):
        names = list(_DISORDER_ALPHABET)  # binary 0/1 (ordered/disordered)
    else:  # signal_peptide — infer from data
        max_id = 0
        for labs in all_label_lists:
            if labs:
                max_id = max(max_id, max(labs))
        names = [f"C{i}" for i in range(max_id + 1)]
    id2label = {i: n for i, n in enumerate(names)}
    label2id = {n: i for i, n in id2label.items()}
    return {"num_labels": len(names), "id2label": id2label, "label2id": label2id, "names": names}


def _prepare_dataset(cfg: TaskConfig, task: str, args: argparse.Namespace):
    from datasets import load_dataset, load_from_disk

    local_path = resolve_local_dataset_path(cfg.dataset)
    if local_path is not None:
        try:
            ds = load_from_disk(str(local_path))
        except (FileNotFoundError, OSError, ValueError):
            raise RuntimeError(f"Failed to load local dataset {cfg.dataset} from {local_path}")
    else:
        ds = load_dataset(cfg.dataset)
    seq_col = cfg.input_map["seq"]
    label_col = cfg.label_col

    splits: Dict[str, Any] = {}
    if cfg.split_column:
        source = ds[cfg.train_split]
        stage_to_split = {"train": "train", "valid": "validation", "test": "test"}
        if cfg.validation_column_values:
            for v in cfg.validation_column_values:
                stage_to_split[v] = "validation"
        for stage_val, split_name in stage_to_split.items():
            sub = source.filter(lambda r, s=stage_val: r[cfg.split_column] == s)
            if len(sub) > 0:
                splits.setdefault(split_name, sub)
    else:
        splits["train"] = ds[cfg.train_split]
        if cfg.validation_split and cfg.validation_split in ds:
            splits["validation"] = ds[cfg.validation_split]
        if cfg.test_split in ds:
            splits["test"] = ds[cfg.test_split]

    if args.max_train_samples is not None and "train" in splits:
        n = min(len(splits["train"]), args.max_train_samples)
        splits["train"] = splits["train"].select(range(n))

    return splits, seq_col, label_col


def _tokenize_and_align(
    splits: Dict[str, Any],
    seq_col: str,
    label_col: str,
    task: str,
    tokenizer,
    max_length: int,
):
    def _map_fn(batch: Dict[str, list]):
        encoded = tokenizer(
            batch[seq_col],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        labels_out = []
        for i, label_str in enumerate(batch[label_col]):
            residue_labels = _decode_labels(task, label_str)
            # Truncate residue labels to fit (max_length includes specials).
            input_ids = encoded["input_ids"][i]
            stm = encoded["special_tokens_mask"][i]
            # Number of non-special positions in the (possibly truncated) sequence:
            n_non_special = sum(1 for s in stm if s == 0)
            residue_labels = residue_labels[:n_non_special]
            labels_out.append(align_labels_with_tokens(input_ids, residue_labels, stm))
        encoded["labels"] = labels_out
        return encoded

    out: Dict[str, Any] = {}
    for split_name, sub in splits.items():
        cols_to_remove = list(sub.column_names)
        out[split_name] = sub.map(
            _map_fn,
            batched=True,
            remove_columns=cols_to_remove,
            desc=f"Tokenizing {split_name}",
        )
    return out


def _build_compute_metrics(task: str, label_names: List[str]):
    def _flatten(predictions, labels):
        preds = np.argmax(predictions, axis=-1)
        flat_preds, flat_labels = [], []
        for p_row, l_row in zip(preds, labels):
            for p, l in zip(p_row, l_row):
                if l != -100:
                    flat_preds.append(int(p))
                    flat_labels.append(int(l))
        return flat_preds, flat_labels

    if task == "disorder":
        from sklearn.metrics import f1_score, matthews_corrcoef

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            preds, labs = _flatten(predictions, labels)
            return {
                "MCC": float(matthews_corrcoef(labs, preds)) if labs else 0.0,
                "F1_Macro": float(f1_score(labs, preds, average="macro", zero_division=0)),
                "Accuracy": float(np.mean(np.array(preds) == np.array(labs))) if labs else 0.0,
            }

        return compute_metrics

    from sklearn.metrics import f1_score, matthews_corrcoef

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds, labs = _flatten(predictions, labels)
        if not labs:
            # MCC included so tasks whose main_metric is MCC (disprot) never
            # KeyError on the empty-eval edge case.
            return {"Accuracy": 0.0, "F1_Macro": 0.0, "MCC": 0.0}
        labs_a = np.array(labs)
        preds_a = np.array(preds)
        per_class = {}
        for cls_id, cls_name in enumerate(label_names):
            mask = labs_a == cls_id
            if mask.any():
                per_class[cls_name] = float(
                    f1_score(labs_a == cls_id, preds_a == cls_id, zero_division=0)
                )
        return {
            "Accuracy": float(np.mean(preds_a == labs_a)),
            "F1_Macro": float(f1_score(labs, preds, average="macro", zero_division=0)),
            # disprot's headline metric (main_metric=MCC) is a GENERIC-branch task
            # (task != "disorder"), so MCC must be emitted here too — otherwise
            # best-model selection / result collection KeyError on eval_MCC.
            "MCC": float(matthews_corrcoef(labs, preds)),
            **{f"F1_{k}": v for k, v in per_class.items()},
        }

    return compute_metrics


def _run_task(task: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = TASKS[task]
    logger.info("=== Task: %s (%s) ===", task, cfg.name)

    splits, seq_col, label_col = _prepare_dataset(cfg, task, args)
    if "train" not in splits:
        raise RuntimeError(f"No train split for task {task}")

    # Decode labels eagerly to compute num_labels (needed for the head).
    all_label_lists = [_decode_labels(task, x) for x in splits["train"][label_col]]
    label_meta = _build_label_meta(task, all_label_lists)

    tokenizer = load_tokenizer(args.model_name)
    tokenized = _tokenize_and_align(
        splits, seq_col, label_col, task, tokenizer, args.max_length
    )

    model = load_encoder_for_head(
        args.model_name,
        AutoModelForTokenClassification,
        num_labels=label_meta["num_labels"],
        id2label=label_meta["id2label"],
        label2id=label_meta["label2id"],
    )
    model_type = getattr(model.config, "model_type", None)
    # peft's TaskType is only needed by the lora branch; import lazily so
    # probe / full / last_n don't require peft installed.
    task_type = None
    if args.mode == "lora":
        from peft import TaskType

        task_type = TaskType.TOKEN_CLS
    model = apply_finetune_mode(
        model,
        mode=args.mode,
        model_type=model_type,
        task_type=task_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        last_n=args.last_n,
    )

    out_dir = Path(args.output_dir) / f"finetune_residue_{safe_ckpt(args.model_name)}_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping selects the best checkpoint on the VALIDATION split only —
    # never test (that would leak). No val + --early_stop -> warn + train to cap,
    # and load_best_model_at_end must be OFF (eval_available=False) so no "best"
    # is picked on test.
    has_validation = tokenized.get("validation") is not None
    training_args = build_training_args(
        args, out_dir, main_metric=cfg.main_metric, eval_available=has_validation
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = _build_compute_metrics(task, label_meta["names"])

    eval_during_train, callbacks = resolve_early_stopping(
        tokenized, early_stop=args.early_stop, task=task,
        patience=args.early_stop_patience,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_during_train,
        data_collator=collator,
        # transformers 5.x renamed Trainer's ``tokenizer`` -> ``processing_class``.
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        # Drop non-logits outputs BEFORE accumulation: ESM-C returns
        # (logits, hidden_states, ...) which otherwise reaches compute_metrics
        # as a ragged tuple (np.array ValueError) and pins every hidden state
        # in memory (OOM). No-op for single-tensor outputs (Proteva).
        preprocess_logits_for_metrics=keep_logits_only,
        callbacks=callbacks,
    )

    trainer.train()

    for split_name in ("validation", "test"):
        split_ds = tokenized.get(split_name)
        if split_ds is None:
            continue
        metric = trainer.evaluate(eval_dataset=split_ds, metric_key_prefix="eval")
        # Subset breakdown for SS3 / disorder (test split only, via the ``dataset``
        # column on the *raw* split before tokenization removed columns).
        eval_subsets: Dict[str, Any] = {}
        if split_name == "test" and task in ("ss3", "disorder") and "test" in splits:
            raw_test = splits["test"]
            if "dataset" in raw_test.column_names:
                for tag in ("cb513", "ts115", "casp12"):
                    idx = [i for i, v in enumerate(raw_test["dataset"]) if v == tag]
                    if not idx:
                        continue
                    sub = split_ds.select(idx)
                    eval_subsets[tag] = trainer.evaluate(
                        eval_dataset=sub, metric_key_prefix=f"eval_{tag}"
                    )
        record = {
            "checkpoint": args.model_name,
            "task": task,
            "mode": args.mode,
            "split": split_name,
            "model_type": model_type,
            "metric": metric,
            "eval_subsets": eval_subsets,
            "n_train": len(tokenized["train"]),
            "n_eval": len(split_ds),
            "notes": args.notes,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "transformers_version": transformers.__version__,
            "args": vars(args),
        }
        jsonl_path = write_jsonl_record(
            out_dir, "finetune_residue", f"{args.model_name}_{task}", record
        )
        logger.info("Wrote %s", jsonl_path)
    return record


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Residue-level fine-tuning for PLMs.")
    add_common_finetune_args(p)
    p.add_argument("--task", required=True, choices=list(_RESIDUE_TASKS) + ["all"])
    # ``lastn`` is the shell-driver alias for ``last_n`` (apply_finetune_mode maps it).
    p.add_argument("--mode", default="probe", choices=["probe", "full", "lora", "last_n", "lastn"])
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tasks_to_run = list(_RESIDUE_TASKS) if args.task == "all" else [args.task]
    for t in tasks_to_run:
        _run_task(t, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
