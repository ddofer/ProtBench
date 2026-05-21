"""Sequence-level fine-tuning for protein LMs: probe / full / LoRA modes.

Wraps any task in the vendored ``TASKS`` registry whose ``problem_type``
is one of ``{binary, multiclass, regression}``. Uses HF
``AutoModelForSequenceClassification`` + ``Trainer`` + ``DataCollatorWithPadding``;
LoRA mode wraps the encoder via ``peft.get_peft_model`` while leaving
the classification head fully trainable.

Multilabel and retrieval are intentionally out of scope here — those
tasks stay on the existing linear-probe path of
``protein_benchmark_suite.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hf_finetune_common import (  # noqa: E402
    add_common_finetune_args,
    build_training_args,
    load_encoder_for_head,
    safe_ckpt,
    write_jsonl_record,
)
from benchmark_tasks import TASKS, TaskConfig  # noqa: E402

logger = logging.getLogger(__name__)

_SUPPORTED_PROBLEM_TYPES = {"binary", "multiclass", "regression"}
_VALIDATION_ALIASES = ("validation", "valid", "val", "dev")


def _select_by_column(ds_split, column: str, values: Tuple[str, ...]):
    vset = set(values)
    return ds_split.filter(lambda r, c=column, vs=vset: r[c] in vs)


def _load_task_splits(cfg: TaskConfig, max_train_samples: int | None):
    from datasets import load_dataset

    load_kwargs: Dict[str, Any] = {}
    if cfg.dataset_config:
        load_kwargs["name"] = cfg.dataset_config
    if cfg.data_dir:
        load_kwargs["data_dir"] = cfg.data_dir
    ds = load_dataset(cfg.dataset, trust_remote_code=True, **load_kwargs)

    train = ds[cfg.train_split]
    eval_split = None
    test_split = None
    if cfg.split_column:
        train = _select_by_column(ds[cfg.train_split], cfg.split_column, (cfg.train_split,))
        if cfg.validation_column_values:
            eval_split = _select_by_column(
                ds[cfg.train_split], cfg.split_column, cfg.validation_column_values
            )
        test_split = _select_by_column(
            ds[cfg.train_split], cfg.split_column, (cfg.test_split,)
        )
    else:
        if cfg.validation_split and cfg.validation_split in ds:
            eval_split = ds[cfg.validation_split]
        else:
            eval_split = next((ds[n] for n in _VALIDATION_ALIASES if n in ds), None)
        if cfg.test_split in ds:
            test_split = ds[cfg.test_split]
        elif cfg.auto_split:
            n = len(train)
            n_train = int(n * 0.8)
            train_sh = train.shuffle(seed=42)
            train = train_sh.select(range(n_train))
            test_split = train_sh.select(range(n_train, n))

    if max_train_samples is not None and train is not None:
        train = train.select(range(min(len(train), max_train_samples)))

    return train, eval_split, test_split


def _parse_label(val: Any, problem_type: str):
    if problem_type == "regression":
        if isinstance(val, (list, tuple)):
            return float(val[0])
        return float(val)
    if isinstance(val, (list, tuple)):
        val = val[0]
    try:
        return int(val)
    except (ValueError, TypeError):
        return val


def _tokenize_splits(
    train, eval_split, test_split, cfg: TaskConfig, tokenizer, max_length: int
):
    seq_col = cfg.input_map.get("seq")
    if seq_col is None:
        raise ValueError(
            f"Task {cfg.name}: only single-sequence tasks are supported here "
            f"(input_map={cfg.input_map})."
        )
    label_col = cfg.label_col

    def _map_fn(batch):
        seqs = batch[seq_col]
        if cfg.remove_sequence_whitespace:
            seqs = ["".join(s.split()) for s in seqs]
        enc = tokenizer(seqs, truncation=True, max_length=max_length)
        enc["labels"] = [_parse_label(v, cfg.problem_type) for v in batch[label_col]]
        return enc

    out = {}
    for name, split in (("train", train), ("validation", eval_split), ("test", test_split)):
        if split is None:
            continue
        out[name] = split.map(
            _map_fn,
            batched=True,
            remove_columns=list(split.column_names),
            desc=f"Tokenizing {name}",
        )
    return out


def _label_meta(cfg: TaskConfig, train_split) -> Dict[str, Any]:
    if cfg.problem_type == "regression":
        return {"num_labels": 1, "id2label": None, "label2id": None, "problem_type_hf": "regression"}
    raw = train_split[cfg.label_col]
    parsed = [_parse_label(v, cfg.problem_type) for v in raw]
    classes = sorted(set(parsed))
    num_labels = max(classes) + 1 if all(isinstance(c, int) for c in classes) else len(classes)
    id2label = {i: f"L{i}" for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}
    hf_pt = "single_label_classification"
    return {"num_labels": num_labels, "id2label": id2label, "label2id": label2id, "problem_type_hf": hf_pt}


def _build_compute_metrics(cfg: TaskConfig):
    pt = cfg.problem_type
    if pt == "regression":
        from scipy.stats import spearmanr

        def cm(eval_pred):
            preds, labels = eval_pred
            preds = np.array(preds).reshape(-1)
            labels = np.array(labels).reshape(-1)
            corr, _ = spearmanr(labels, preds)
            mse = float(np.mean((preds - labels) ** 2))
            return {"Spearman": float(corr) if corr == corr else 0.0, "MSE": mse}

        return cm

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    def cm(eval_pred):
        preds_logits, labels = eval_pred
        preds = np.argmax(preds_logits, axis=-1)
        out: Dict[str, float] = {
            "Accuracy": float(accuracy_score(labels, preds)),
            "F1_Macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }
        if pt == "binary":
            num_labels = np.array(preds_logits).shape[-1]
            if num_labels == 2 and len(set(labels)) > 1:
                logits = np.array(preds_logits)
                exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probs = exp / exp.sum(axis=-1, keepdims=True)
                out["AUC"] = float(roc_auc_score(labels, probs[:, 1]))
        return out

    return cm


def _apply_mode(model, args: argparse.Namespace):
    if args.mode == "probe":
        model.base_model.requires_grad_(False)
    elif args.mode == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise RuntimeError(
                "peft is required for --mode lora. Install with "
                "`pip install peft>=0.13` into the sibling venv."
            ) from e
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        base = model.base_model
        wrapped = get_peft_model(base, lora_cfg)
        for name, child in model.named_children():
            if child is base:
                setattr(model, name, wrapped)
                break
    return model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sequence-level fine-tuning for PLMs.")
    add_common_finetune_args(p)
    p.add_argument("--task", required=True, help="Any TASKS key whose problem_type is binary / multiclass / regression.")
    p.add_argument("--mode", default="probe", choices=["probe", "full", "lora"])
    # LoRA-specific
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.task not in TASKS:
        raise SystemExit(f"Unknown task '{args.task}'. Available: {sorted(TASKS)}")
    cfg = TASKS[args.task]
    if cfg.problem_type not in _SUPPORTED_PROBLEM_TYPES:
        raise SystemExit(
            f"Task '{args.task}' has problem_type={cfg.problem_type!r}; this script "
            f"only supports {_SUPPORTED_PROBLEM_TYPES}."
        )

    train, eval_split, test_split = _load_task_splits(cfg, args.max_train_samples)
    if train is None:
        raise SystemExit(f"No train split for task {args.task}")

    label_meta = _label_meta(cfg, train)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenized = _tokenize_splits(train, eval_split, test_split, cfg, tokenizer, args.max_length)

    model = load_encoder_for_head(
        args.model_name,
        AutoModelForSequenceClassification,
        num_labels=label_meta["num_labels"],
        id2label=label_meta["id2label"],
        label2id=label_meta["label2id"],
        problem_type=label_meta["problem_type_hf"] if cfg.problem_type != "regression" else "regression",
    )
    model = _apply_mode(model, args)

    out_dir = Path(args.output_dir) / f"finetune_sequence_{safe_ckpt(args.model_name)}_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_args(args, out_dir)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = _build_compute_metrics(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_set = tokenized.get("test")
    metric: Dict[str, Any] = {}
    if test_set is not None:
        metric = trainer.evaluate(eval_dataset=test_set, metric_key_prefix="eval")

    # LoRA: save only the adapters (small, ~MB).
    if args.mode == "lora":
        adapter_dir = out_dir / "lora_adapter"
        model.base_model.save_pretrained(str(adapter_dir))

    record = {
        "checkpoint": args.model_name,
        "task": args.task,
        "mode": args.mode,
        "metric": metric,
        "n_train": len(tokenized["train"]),
        "n_eval": len(test_set) if test_set is not None else 0,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "transformers_version": transformers.__version__,
        "args": vars(args),
    }
    jsonl_path = write_jsonl_record(
        out_dir, "finetune_sequence", f"{args.model_name}_{args.task}", record
    )
    logger.info("Wrote %s", jsonl_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
