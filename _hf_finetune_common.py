"""Shared helpers for HF Trainer-based fine-tuning (residue + sequence scripts).

Pure functions used by ``finetune_residue.py`` and ``finetune_sequence.py``:
encoder loading with AMPLIFY ``model_type`` quirk, label decoders for
string and CSV residue labels, and special-token-aware label alignment
for token-classification.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

logger = logging.getLogger(__name__)

# Proteva stage-2 checkpoints ship no tokenizer files (they reuse the AMPLIFY
# vocab); fall back to AMPLIFY's tokenizer when AutoTokenizer can't build one.
AMPLIFY_TOKENIZER = "chandar-lab/AMPLIFY_120M"


def load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except (ValueError, OSError):
        return AutoTokenizer.from_pretrained(AMPLIFY_TOKENIZER, trust_remote_code=True)


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

    Proteva (``model_type == "proteva"``): ``plm.hf`` registers the Proteva heads
    with the Auto factories; the head's ``from_pretrained`` forces dense SDPA and
    repairs the RoPE cache. AMPLIFY: force ``config.model_type == "AMPLIFY"`` (some
    repos omit it). Everything else: standard HF dispatch.
    """
    # Must import plm.hf BEFORE AutoConfig: a model_type=="proteva" checkpoint is
    # unknown until it registers. Best-effort so non-Proteva envs still work.
    try:
        import plm.hf  # noqa: F401
    except Exception:
        pass

    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)

    if model_type == "proteva":
        config.num_labels = num_labels
        if id2label is not None:
            config.id2label = id2label
        if label2id is not None:
            config.label2id = label2id
        # ProtevaConfig has no problem_type kwarg; the head reads it via getattr.
        problem_type = kwargs.pop("problem_type", None)
        if problem_type is not None:
            config.problem_type = problem_type
        # Load fp32: a RoPE cache computed in bf16 goes NaN; Trainer casts later.
        return head_cls.from_pretrained(
            ckpt,
            config=config,
            ignore_mismatched_sizes=True,
            **kwargs,
        )

    if model_type in (None, "", "amplify") and (
        "amplify" in ckpt.lower() or "AMPLIFY" in ckpt
    ):
        config.model_type = "AMPLIFY"
    config.num_labels = num_labels
    if id2label is not None:
        config.id2label = id2label
    if label2id is not None:
        config.label2id = label2id
    # ``problem_type`` is a config attribute, not a model __init__ kwarg.
    problem_type = kwargs.pop("problem_type", None)
    if problem_type is not None:
        config.problem_type = problem_type
    return head_cls.from_pretrained(
        ckpt,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        **kwargs,
    )


# Per-family LoRA targets (body attention + FFN Linears). NOT "all-linear": that
# would also wrap the embedding, MLM decoder and aux heads, which must stay frozen.
# Proteva: every body Linear + the ve_first/ve_last value embeddings.
PROTEVA_LORA_TARGETS = ["wq", "wk", "wv", "wo", "attn_gate", "w12", "w3",
                        "ve_first", "ve_last"]
# AMPLIFY: separate q/k/v (NOT fused ``Wqkv`` -- another rev; it would silently
# match nothing and leave q/k/v unadapted) + SwiGLU w12/w3.
AMPLIFY_LORA_TARGETS = ["q", "k", "v", "wo", "w12", "w3"]
ESM_LORA_TARGETS = ["query", "key", "value", "dense"]


def lora_target_modules(model_type: str | None) -> list[str] | str:
    """Return the LoRA ``target_modules`` for a model family.

    Unknown families fall back to PEFT's ``"all-linear"`` default (never used for Proteva).
    """
    mt = (model_type or "").lower()
    if mt == "proteva":
        return list(PROTEVA_LORA_TARGETS)
    if mt == "amplify":
        return list(AMPLIFY_LORA_TARGETS)
    if mt in ("esm", "esm2"):
        return list(ESM_LORA_TARGETS)
    return "all-linear"


def _encoder_blocks(base_model):
    """Best-effort handle on the per-layer block ``ModuleList`` for last-N.

    Proteva: ``encoder.blocks``; HF ESM: ``encoder.layer``; AMPLIFY:
    ``transformer_encoder``. Returns ``None`` if not found."""
    for attr in ("blocks", "layer", "layers", "transformer_encoder"):
        mod = getattr(base_model, attr, None)
        if mod is not None and hasattr(mod, "__len__"):
            return mod
        # one level down (e.g. encoder.encoder.layer)
        inner = getattr(base_model, "encoder", None)
        if inner is not None:
            sub = getattr(inner, attr, None)
            if sub is not None and hasattr(sub, "__len__"):
                return sub
    return None


def apply_finetune_mode(
    model,
    *,
    mode: str,
    model_type: str | None,
    task_type,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    last_n: int = 4,
    classifier_module: str = "classifier",
):
    """Apply ``probe`` / ``full`` / ``lora`` / ``last_n`` to a head model.

    ``probe`` freezes ``model.base_model``; ``full`` is a no-op; ``lora`` wraps the
    body with PEFT LoRA (head kept trainable via ``modules_to_save``); ``last_n``
    unfreezes the top N blocks + final norm. Accepts ``"lastn"`` as an alias.
    Returns the (possibly PEFT-wrapped) model.
    """
    mode = "last_n" if mode == "lastn" else mode

    if mode == "full":
        return model

    if mode == "probe":
        model.base_model.requires_grad_(False)
        return model

    if mode == "last_n":
        base = model.base_model
        base.requires_grad_(False)
        blocks = _encoder_blocks(base)
        if blocks is None:
            import warnings

            warnings.warn(
                "apply_finetune_mode(last_n): could not find the encoder block "
                "ModuleList; the body stays fully frozen (acts like probe).",
                stacklevel=2,
            )
        else:
            n = min(int(last_n), len(blocks))
            for blk in list(blocks)[len(blocks) - n:]:
                blk.requires_grad_(True)
            final_norm = getattr(base, "final_norm", None)
            if final_norm is not None:
                final_norm.requires_grad_(True)
        return model

    if mode == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise RuntimeError(
                "peft is required for --mode lora. Install with "
                "`pip install 'peft>=0.13'` into the bench venv."
            ) from e
        lora_cfg = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            # No intercept is lost: the head's own bias trains via modules_to_save.
            bias="none",
            # rsLoRA scales by alpha/sqrt(r) so high r (we run r=64) stays stable.
            use_rslora=True,
            task_type=task_type,
            target_modules=lora_target_modules(model_type),
            # The head is NOT in target_modules; PEFT would otherwise freeze it.
            modules_to_save=[classifier_module],
        )
        # Wrap the WHOLE head model: wrapping the bare encoder routes through PEFT's
        # task forward, which reads config.use_return_dict (absent on EncoderConfig) and crashes.
        return get_peft_model(model, lora_cfg)

    raise ValueError(f"Unknown finetune mode: {mode!r}")


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
            # short residue_labels default to ignore-index rather than raising mid-batch.
            aligned.append(int(next(res_iter, -100)))
    return aligned


def safe_ckpt(ckpt: str) -> str:
    return ckpt.replace("/", "_").replace("\\", "_")


def add_common_finetune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="plm/results/bench/")
    parser.add_argument("--notes", type=str, default="",
                        help="Free-text note copied into each result record (e.g. 'amplify-120M-HEAD + epoch1').")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--last_n", type=int, default=4,
                        help="Top-N encoder blocks (+ final norm) to unfreeze for --mode last_n.")
    parser.add_argument("--early_stop", action="store_true",
                        help="Eval each epoch on validation, keep the best (load_best_model_at_end) "
                             "with EarlyStoppingCallback; --num_train_epochs becomes the max cap.")
    parser.add_argument("--early_stop_patience", type=int, default=1,
                        help="Epochs of no val-metric improvement before stopping (with --early_stop).")
    parser.add_argument("--fp32", action="store_true",
                        help="Train in fp32 (disable bf16). Needed for regression: a bf16 forward "
                             "is too coarse for the regression head -> constant preds / Spearman~0.")


# Metric direction must follow the metric: ``meltome``'s main metric is MSE, and a
# hardcoded greater_is_better=True would make early stopping keep the WORST epoch.
from benchmark_tasks import (  # noqa: E402  -- canonical metric direction
    metric_greater_is_better,
)


def resolve_early_stopping(
    tokenized: Dict[str, Any],
    *,
    early_stop: bool,
    task: str,
    patience: int,
):
    """Select the in-loop eval dataset + early-stopping callbacks for the FT scripts.

    Only ``"validation"`` may drive best-model selection; ``"test"`` is NEVER used
    (selecting on test then reporting test is a leak). With ``early_stop`` but no
    validation split, warn and return no callbacks (caller must also pass
    ``eval_available=False`` to :func:`build_training_args`).

    Returns ``(eval_during_train, callbacks)``; both empty without ``early_stop``."""
    eval_during_train = None
    callbacks: List[Any] = []
    if not early_stop:
        return eval_during_train, callbacks
    eval_during_train = tokenized.get("validation")
    if eval_during_train is not None:
        from transformers import EarlyStoppingCallback

        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    else:
        logger.warning(
            "no validation split for %s; not early-stopping/selecting on the "
            "test set (leak); training to max epochs",
            task,
        )
    return eval_during_train, callbacks


def keep_logits_only(logits, labels):
    """``preprocess_logits_for_metrics`` hook: keep the logits, drop the rest.

    ESM-C heads return ``(logits, hidden_states, ...)``; HF accumulates every tensor
    over the eval set, so the ragged tuple breaks ``np.array`` in ``compute_metrics``
    and the retained hidden states OOM. Passes a single tensor through unchanged.
    """
    return logits[0] if isinstance(logits, (tuple, list)) else logits


def build_training_args(
    args: argparse.Namespace, output_dir: Path, *, main_metric: str | None = None,
    eval_available: bool = True,
) -> TrainingArguments:
    """TrainingArguments for the FT scripts.

    With ``args.early_stop`` + ``main_metric`` + ``eval_available``: per-epoch eval and
    ``load_best_model_at_end``; ``num_train_epochs`` becomes the MAX cap. Otherwise
    fixed epochs, no eval, keep last. ``eval_available=False`` (no validation split)
    forces the fixed-epoch path: selecting on test then reporting test is a leak."""
    kw = dict(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=max(1, args.logging_steps),
        report_to=[],
        seed=args.seed,
        fp16=False,
        # bf16 logits (~3 sig digits) cannot rank fine regression targets -> constant
        # preds, Spearman~0; --fp32 (set for regression tasks) keeps the forward fp32.
        bf16=torch.cuda.is_bf16_supported() and not getattr(args, "fp32", False),
        dataloader_num_workers=args.dataloader_num_workers,
    )
    if getattr(args, "early_stop", False) and main_metric is not None and eval_available:
        kw.update(
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=main_metric,
            greater_is_better=metric_greater_is_better(main_metric),
            save_total_limit=1,
        )
    else:
        kw.update(eval_strategy="no", save_strategy="no")
    return TrainingArguments(**kw)


def resolve_local_dataset_path(dataset_name: str):
    """Resolve a dataset specifier to a local path if it exists (mirrors probe logic)."""
    dataset_path = Path(dataset_name).expanduser()
    candidates = [dataset_path]
    if not dataset_path.is_absolute():
        candidates.append(Path(__file__).resolve().parent / dataset_path)
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    return None


def write_jsonl_record(output_dir: Path, prefix: str, ckpt: str, record: Dict[str, Any]) -> Path:
    jsonl_path = output_dir.parent / f"{prefix}_{safe_ckpt(ckpt)}.jsonl"
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return jsonl_path
