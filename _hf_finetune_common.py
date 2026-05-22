"""Shared helpers for HF Trainer-based fine-tuning (residue + sequence scripts).

Pure functions used by ``finetune_residue.py`` and ``finetune_sequence.py``:
encoder loading with AMPLIFY ``model_type`` quirk, label decoders for
string and CSV residue labels, and special-token-aware label alignment
for token-classification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoConfig, TrainingArguments


def load_encoder_for_head(
    ckpt: str,
    head_cls,
    *,
    num_labels: int,
    id2label: Dict[int, str] | None = None,
    label2id: Dict[str, int] | None = None,
    **kwargs,
):
    """Load ``ckpt`` into the HF auto-model ``head_cls``.

    Ensures ``config.model_type == "AMPLIFY"`` is set (some AMPLIFY repos
    omit this; the sibling suite's AMPLIFY detection depends on it).
    Passes ``trust_remote_code=True`` to support AMPLIFY's remote modules.
    """
    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    if getattr(config, "model_type", None) in (None, "", "amplify") and (
        "amplify" in ckpt.lower() or "AMPLIFY" in ckpt
    ):
        config.model_type = "AMPLIFY"
    config.num_labels = num_labels
    if id2label is not None:
        config.id2label = id2label
    if label2id is not None:
        config.label2id = label2id
    return head_cls.from_pretrained(
        ckpt,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        **kwargs,
    )


def decode_string_label(label_str: str, alphabet: str) -> List[int]:
    return [alphabet.index(c) for c in label_str]


def decode_csv_label(label_str: str) -> List[int]:
    return [int(tok) for tok in label_str.split(",") if tok.strip() != ""]


def align_labels_with_tokens(
    input_ids: Sequence[int],
    residue_labels: Sequence[int],
    special_tokens_mask: Sequence[int],
) -> List[int]:
    """Align per-residue labels to a tokenized sequence (1 token = 1 AA).

    Inserts ``-100`` at every position where ``special_tokens_mask == 1``
    (CLS / SEP / PAD). Non-special positions consume the next residue
    label in order. Output length equals ``len(input_ids)``.
    """
    aligned: List[int] = []
    res_iter = iter(residue_labels)
    for is_special in special_tokens_mask:
        if is_special:
            aligned.append(-100)
        else:
            # short residue_labels (malformed CSV row, short signal_peptide
            # sequence) defaults to ignore-index rather than raising mid-batch.
            aligned.append(int(next(res_iter, -100)))
    return aligned


def safe_ckpt(ckpt: str) -> str:
    return ckpt.replace("/", "_").replace("\\", "_")


def add_common_finetune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="plm/results/bench/")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)


def build_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir / "trainer"),
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
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=args.dataloader_num_workers,
    )


def write_jsonl_record(output_dir: Path, prefix: str, ckpt: str, record: Dict[str, Any]) -> Path:
    jsonl_path = output_dir.parent / f"{prefix}_{safe_ckpt(ckpt)}.jsonl"
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return jsonl_path
