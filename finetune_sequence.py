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
    DataCollatorWithPadding,
    Trainer,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hf_finetune_common import (  # noqa: E402
    add_common_finetune_args,
    apply_finetune_mode,
    build_training_args,
    load_encoder_for_head,
    load_tokenizer,
    resolve_local_dataset_path,
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
    from datasets import load_dataset, load_from_disk

    local_path = resolve_local_dataset_path(cfg.dataset)
    if local_path is not None:
        # Local Arrow/DatasetDict on disk — use load_from_disk with CheZOD fallback
        try:
            ds = load_from_disk(str(local_path))
        except (FileNotFoundError, OSError, ValueError):
            if local_path.name == "chezod":
                # Import the same raw-JSON fallback the probe uses
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from protein_benchmark_suite import _load_chezod_from_raw
                ds = _load_chezod_from_raw(local_path)
            else:
                raise
    else:
        load_kwargs: Dict[str, Any] = {}
        if cfg.dataset_config:
            load_kwargs["name"] = cfg.dataset_config
        if cfg.data_dir:
            load_kwargs["data_dir"] = cfg.data_dir
        ds = load_dataset(cfg.dataset, **load_kwargs)

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

    # ``full_train`` is kept unsampled so _label_meta can build the complete label
    # vocabulary (val/test may contain classes absent from the sampled train rows).
    full_train = train
    if max_train_samples is not None and train is not None:
        train = train.select(range(min(len(train), max_train_samples)))

    return train, eval_split, test_split, full_train


def _parse_label(val: Any, problem_type: str, force_str: bool = False):
    """Parse a raw label value.

    Regression → float.  Classification:
    * ``force_str=True`` (for string-typed label columns like EC / subcellular_loc):
        always return str(val) — caller maps to int via label2id.
    * ``force_str=False`` (default, for int-typed columns like remote_homology):
        unwrap list/tuple then return int if possible, else str.

    The caller is responsible for mapping str labels to contiguous indices via
    the ``label2id`` dict built in ``_label_meta``.
    """
    if problem_type == "regression":
        if isinstance(val, (list, tuple)):
            return float(val[0])
        return float(val)
    if isinstance(val, (list, tuple)):
        val = val[0]
    if force_str:
        return str(val)
    try:
        return int(val)
    except (ValueError, TypeError):
        return str(val)


def _tokenize_splits(
    train, eval_split, test_split, cfg: TaskConfig, tokenizer, max_length: int,
    normalize_targets: bool = False,
    label2id: Dict[str, Any] | None = None,
    force_str: bool = False,
):
    seq_col = cfg.input_map.get("seq")
    if seq_col is None:
        raise ValueError(
            f"Task {cfg.name}: only single-sequence tasks are supported here "
            f"(input_map={cfg.input_map})."
        )
    label_col = cfg.label_col

    # Optional (off by default, like PEER's `normalization:False`): z-score the
    # regression targets with TRAIN stats. Spearman is rank-invariant so the
    # metric is unchanged in principle, BUT normalizing rescales the MSE gradient
    # by 1/std^2 — for beta_lactamase (std~0.32) that ~10x's the effective LR, so
    # it is NOT free. Only help when target std is far from 1; pair with a lower
    # LR. (Lit: van Hasselt 2016.)
    label_mean = label_std = None
    if cfg.problem_type == "regression" and normalize_targets:
        import numpy as np

        _raw = np.asarray(
            [_parse_label(v, "regression") for v in train[label_col]], dtype="float64"
        )
        label_mean = float(_raw.mean())
        label_std = float(_raw.std()) or 1.0

    def _map_fn(batch):
        seqs = batch[seq_col]
        if cfg.remove_sequence_whitespace:
            seqs = ["".join(s.split()) for s in seqs]
        enc = tokenizer(seqs, truncation=True, max_length=max_length)
        labels = [_parse_label(v, cfg.problem_type, force_str=force_str) for v in batch[label_col]]
        if label_mean is not None:
            labels = [(x - label_mean) / label_std for x in labels]
        # Map parsed labels to contiguous int indices via label2id.
        # label2id keys are str (HF compat): convert parsed value to str before lookup.
        # The vocab was built from ALL splits so every label is guaranteed in label2id.
        if label2id is not None and cfg.problem_type != "regression":
            labels = [label2id[str(lbl)] for lbl in labels]
        enc["labels"] = labels
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


def _label_meta(cfg: TaskConfig, train_split, *extra_splits) -> Dict[str, Any]:
    """Build label metadata for the classification head.

    ``extra_splits`` (val / test) are included when building the label
    vocabulary so that classes exclusive to eval splits don't cause KeyError
    during tokenization — they are rare but possible for very small train
    sample caps (``--max_train_samples``) on large multiclass tasks.

    Detects whether the label column stores strings (EC / subcellular_loc) or
    native ints (remote_homology), and builds the mapping accordingly:

    * Native-int column: preserves original indices (0..N-1 contiguous or dense
      enough), label2id is identity int→int.
    * String column: sorts unique string labels alphabetically and builds a
      contiguous 0..N-1 str→int mapping.  label2id keys are str so that HF
      model-config validation accepts them.
    """
    if cfg.problem_type == "regression":
        return {"num_labels": 1, "id2label": None, "label2id": None, "problem_type_hf": "regression"}

    # Detect whether the raw label column contains strings or native ints
    # by peeking at the dataset features (avoids depending on sample content).
    import datasets as _ds_mod
    raw_feature = train_split.features.get(cfg.label_col)
    _raw_is_string = (
        isinstance(raw_feature, _ds_mod.Value) and raw_feature.dtype == "string"
    ) or (
        isinstance(raw_feature, _ds_mod.ClassLabel)  # ClassLabel → int already handled
        and False  # ClassLabel is always int, let it fall through to int path
    )
    # Also treat it as string if the first few raw values are non-int Python objects
    if not _raw_is_string:
        sample_vals = train_split[cfg.label_col][:10]
        _raw_is_string = any(isinstance(v, str) for v in sample_vals)

    force_str = _raw_is_string

    all_splits = [train_split] + [s for s in extra_splits if s is not None]
    parsed_all: list = []
    for sp in all_splits:
        parsed_all.extend([_parse_label(v, cfg.problem_type, force_str=force_str) for v in sp[cfg.label_col]])

    classes = sorted(set(parsed_all), key=lambda x: (isinstance(x, str), str(x) if isinstance(x, str) else x))
    if all(isinstance(c, int) for c in classes):
        # Native-int labels (e.g. remote_homology 0..N-1): use max+1 so sparse
        # ranges work; id2label/label2id keys are int for internal use but HF
        # config wants str keys — store int keys, convert in load_encoder_for_head
        # call where they are set on the config object (HF accepts int keys on the
        # config dataclass; the validation error was for label2id with int keys).
        num_labels = max(classes) + 1
        id2label = {i: str(i) for i in range(num_labels)}
        label2id = {str(i): i for i in range(num_labels)}  # str keys for HF compat
    else:
        # String labels: build stable sorted contiguous str→int mapping
        num_labels = len(classes)
        id2label = {i: str(c) for i, c in enumerate(classes)}
        label2id = {str(c): i for i, c in enumerate(classes)}  # str keys
    hf_pt = "single_label_classification"
    return {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
        "problem_type_hf": hf_pt,
        "_force_str": force_str,
    }


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


def _apply_mode(model, args: argparse.Namespace, model_type: str | None):
    """Apply probe / full / lora / last_n via the shared helper.

    ``model_type`` (from the loaded config) selects the LoRA target modules:
    Proteva -> wq/wk/wv/wo + w12/w3 (NOT "all-linear").
    """
    # peft's ``TaskType`` is only consumed by the lora branch; import it lazily so
    # probe / full / last_n runs don't require peft to be installed.
    task_type = None
    if args.mode == "lora":
        from peft import TaskType

        task_type = TaskType.SEQ_CLS
    return apply_finetune_mode(
        model,
        mode=args.mode,
        model_type=model_type,
        task_type=task_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        last_n=args.last_n,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sequence-level fine-tuning for PLMs.")
    add_common_finetune_args(p)
    p.add_argument("--task", required=True, help="Any TASKS key whose problem_type is binary / multiclass / regression.")
    p.add_argument("--normalize_targets", action="store_true", help="z-score regression targets with train stats (off by default; see _tokenize_splits).")
    # ``lastn`` is the shell-driver alias for ``last_n`` (apply_finetune_mode maps it).
    p.add_argument("--mode", default="probe", choices=["probe", "full", "lora", "last_n", "lastn"])
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

    train, eval_split, test_split, full_train = _load_task_splits(cfg, args.max_train_samples)
    if train is None:
        raise SystemExit(f"No train split for task {args.task}")

    # Build label vocabulary from all available splits combined so that labels
    # exclusive to val/test (absent from --max_train_samples train subset, or
    # even from the full train) are still in label2id and don't cause KeyError
    # during tokenization of eval splits.
    label_meta = _label_meta(cfg, full_train, eval_split, test_split)
    tokenizer = load_tokenizer(args.model_name)
    tokenized = _tokenize_splits(
        train, eval_split, test_split, cfg, tokenizer, args.max_length,
        normalize_targets=args.normalize_targets,
        label2id=label_meta["label2id"],
        force_str=label_meta.get("_force_str", False),
    )

    model = load_encoder_for_head(
        args.model_name,
        AutoModelForSequenceClassification,
        num_labels=label_meta["num_labels"],
        id2label=label_meta["id2label"],
        label2id=label_meta["label2id"],
        problem_type=label_meta["problem_type_hf"] if cfg.problem_type != "regression" else "regression",
    )
    model_type = getattr(model.config, "model_type", None)
    model = _apply_mode(model, args, model_type)

    out_dir = Path(args.output_dir) / f"finetune_sequence_{safe_ckpt(args.model_name)}_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_args(args, out_dir, main_metric=cfg.main_metric)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = _build_compute_metrics(cfg)

    # Early stopping needs an in-loop eval set (the validation split) + the
    # callback; without --early_stop both stay absent (old fixed-epoch behavior).
    callbacks = []
    eval_during_train = None
    if args.early_stop:
        from transformers import EarlyStoppingCallback

        eval_during_train = tokenized.get("validation") or tokenized.get("test")
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_during_train,
        data_collator=collator,
        # transformers 5.x renamed Trainer's ``tokenizer`` -> ``processing_class``.
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    # LoRA: save only the adapters (small, ~MB). ``model`` is the PeftModel
    # (apply_finetune_mode wraps the whole head model), so save_pretrained writes
    # the adapter + the modules_to_save'd classifier, not the full encoder.
    if args.mode == "lora":
        adapter_dir = out_dir / "lora_adapter"
        model.save_pretrained(str(adapter_dir))

    for split_name in ("validation", "test"):
        split_ds = tokenized.get(split_name)
        if split_ds is None:
            continue
        metric = trainer.evaluate(eval_dataset=split_ds, metric_key_prefix="eval")
        record = {
            "checkpoint": args.model_name,
            "task": args.task,
            "mode": args.mode,
            "split": split_name,
            "model_type": model_type,
            "metric": metric,
            "n_train": len(tokenized["train"]),
            "n_eval": len(split_ds),
            "notes": args.notes,
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
