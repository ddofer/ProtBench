"""Residue-level (token-classification) fine-tuning for protein LMs.

Wires AMPLIFY / ESM-style encoders to HF ``AutoModelForTokenClassification``
+ ``Trainer`` for three residue tasks: SS3 (NetSurfP-SS3), intrinsic
disorder (same dataset, ``disorder`` column), and signal peptides
(SignalP6 via SaProtHub). Mirrors the CLI style of ``protein_benchmark_suite.py``.

Dataset provenance (verified 2026-05-21):

  SS3 + disorder — agemagician/NetSurfP-SS3.
    Per-residue 3-class secondary structure (C/H/E, DSSP-derived) and
    a parallel intrinsic-disorder mask from NetSurfP-2.0 (Klausen et
    al., Proteins 2019; doi:10.1002/prot.25674). Rehosted by Ahmed
    Elnaggar (agemagician), NOT the NetSurfP-2.0 authors. The HF
    rehost reshuffled the paper's 10,337/500 train/val into
    10,792/646; CB513 has 511 chains (vs 513 published), CASP12 has
    20 (vs 21 published) — minor filter loss. The "disorder" column
    is the PDB-missing-coordinate mask, NOT DisProt / CAID2 (do not
    cross-compare). License is unspecified on the HF card; original
    release is academic-use under DTU Health Tech.

  signal_peptide — SaProtHub/Dataset-Signal-Peptides.
    Per-residue 7-class signal-peptide tagging (S/T/L/P/I/M/O) over
    N-terminal regions (<=70 aa) from the SignalP 6.0 benchmark
    (Teufel et al., Nature Biotechnology 2022;
    doi:10.1038/s41587-021-01156-3). 25,693 sequences packed into a
    single HF "train" split; the actual partition lives in the
    "stage" column (20,490 train / 2,569 valid / 2,634 test).
    Rehosted by the SaProtHub community. License: MIT.

Datasets are loaded via ``datasets.load_dataset`` honoring the
``TaskConfig`` fields ``split_column``, ``validation_column_values``,
``validation_split``, ``test_split``. For SS3 / disorder we also surface
per-subset metrics across the ``dataset`` column (CB513 / TS115 / CASP12).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import transformers
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Allow ``python plm/bench/finetune_residue.py`` from anywhere; also be
# importable via ``import finetune_residue`` from inside plm/bench/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hf_finetune_common import (  # noqa: E402
    align_labels_with_tokens,
    decode_csv_label,
    decode_string_label,
    load_encoder_for_head,
)
from benchmark_tasks import TASKS, TaskConfig  # noqa: E402

logger = logging.getLogger(__name__)

# Task-specific label alphabets / class names.
_SS3_ALPHABET = "HEC"
_DISORDER_ALPHABET = "01"

# SignalP6 has up to ~7 classes; we infer ``num_labels`` from the data
# (max label id + 1) rather than hard-coding, to remain forward-compatible.

_RESIDUE_TASKS = ("ss3", "disorder", "signal_peptide")


def _safe_ckpt(ckpt: str) -> str:
    return ckpt.replace("/", "_").replace("\\", "_")


def _decode_labels(task: str, label_str: Any) -> List[int]:
    """Dispatch the right per-residue label decoder."""
    if task == "ss3":
        return decode_string_label(str(label_str), _SS3_ALPHABET)
    if task == "disorder":
        return decode_string_label(str(label_str), _DISORDER_ALPHABET)
    if task == "signal_peptide":
        return decode_csv_label(str(label_str))
    raise ValueError(f"Unknown residue task: {task}")


def _build_label_meta(task: str, all_label_lists: List[List[int]]) -> Dict[str, Any]:
    """Compute ``num_labels`` + ``id2label`` / ``label2id`` for a task."""
    if task == "ss3":
        names = list(_SS3_ALPHABET)
    elif task == "disorder":
        names = list(_DISORDER_ALPHABET)
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
    """Load and (lightly) preprocess the HF dataset for the task."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset)
    seq_col = cfg.input_map["seq"]
    label_col = cfg.label_col

    splits: Dict[str, Any] = {}
    if cfg.split_column:
        # SignalP-style: single source with a 'stage' column.
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

    # Smoke caps.
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
    """Tokenize sequences + align per-residue labels (1 token = 1 AA)."""

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

    cols_to_remove = None  # let datasets infer (keep input_ids/attention_mask/labels)
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
    """Return a ``compute_metrics`` fn compatible with HF Trainer."""

    def _flatten(predictions, labels):
        # predictions: (B, T, C) logits; labels: (B, T) with -100 ignored.
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

    # SS3 / signal_peptide: use seqeval-style per-class F1 over flat tokens.
    from sklearn.metrics import f1_score

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds, labs = _flatten(predictions, labels)
        if not labs:
            return {"Accuracy": 0.0, "F1_Macro": 0.0}
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
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
    if args.mode == "probe":
        # Freeze the entire encoder; only the classifier head trains.
        model.base_model.requires_grad_(False)

    out_dir = Path(args.output_dir) / f"finetune_residue_{_safe_ckpt(args.model_name)}_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=max(1, args.logging_steps),
        report_to=[],
        seed=args.seed,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = _build_compute_metrics(task, label_meta["names"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation") or tokenized.get("test"),
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Primary eval on the test split, plus optional per-subset breakdown.
    test_set = tokenized.get("test")
    metric: Dict[str, Any] = {}
    eval_subsets: Dict[str, Any] = {}
    if test_set is not None:
        metric = trainer.evaluate(eval_dataset=test_set, metric_key_prefix="eval")
        # Subset split for SS3 / disorder via the ``dataset`` column on
        # the *raw* split (before tokenization removed columns).
        if task in ("ss3", "disorder") and "test" in splits:
            raw_test = splits["test"]
            if "dataset" in raw_test.column_names:
                for tag in ("cb513", "ts115", "casp12"):
                    idx = [i for i, v in enumerate(raw_test["dataset"]) if v == tag]
                    if not idx:
                        continue
                    sub = test_set.select(idx)
                    eval_subsets[tag] = trainer.evaluate(
                        eval_dataset=sub, metric_key_prefix=f"eval_{tag}"
                    )

    record = {
        "checkpoint": args.model_name,
        "task": task,
        "mode": args.mode,
        "metric": metric,
        "eval_subsets": eval_subsets,
        "n_train": len(tokenized["train"]),
        "n_eval": len(test_set) if test_set is not None else 0,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "transformers_version": transformers.__version__,
        "args": vars(args),
    }
    jsonl_path = out_dir.parent / (
        f"finetune_residue_{_safe_ckpt(args.model_name)}_{task}.jsonl"
    )
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Wrote %s", jsonl_path)
    return record


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Residue-level fine-tuning for PLMs.")
    p.add_argument("--model_name", required=True, help="HF checkpoint (e.g. chandar-lab/AMPLIFY_120M)")
    p.add_argument("--task", required=True, choices=list(_RESIDUE_TASKS) + ["all"])
    p.add_argument("--mode", default="probe", choices=["probe", "full"])
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="plm/results/bench/")
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
