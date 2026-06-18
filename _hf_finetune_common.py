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
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

# Proteva stage-2 checkpoints are saved WITHOUT tokenizer files (they reuse the
# AMPLIFY vocab). When AutoTokenizer can't build one from the checkpoint, fall
# back to AMPLIFY's tokenizer — what the probe suite already does for them.
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

    Handles three model families:

    * **Proteva** (``model_type == "proteva"``): import ``plm.hf`` to register
      ``ProtevaForSequenceClassification`` / ``ProtevaForTokenClassification``
      with the Auto factories, then dispatch through them. The head's own
      ``from_pretrained`` forces ``flash_attn_mode="off"`` (dense SDPA, honors
      the padded ``attention_mask``) and repairs the non-persistent RoPE cache —
      the validated equivalent of the bench loader. ``trust_remote_code`` is
      irrelevant (types are locally registered), but we keep it harmless.
    * **AMPLIFY**: set ``config.model_type == "AMPLIFY"`` (some repos omit it;
      the sibling suite's AMPLIFY detection depends on it) + remote code.
    * Everything else (ESM, …): standard HF dispatch.
    """
    # Register ProtevaConfig + the two discriminative heads on the Auto factories
    # BEFORE the AutoConfig lookup — otherwise ``AutoConfig.from_pretrained`` on a
    # ``model_type=="proteva"`` checkpoint raises (the type is unknown until
    # ``plm.hf`` is imported). Idempotent: the registration swallows duplicate
    # errors. Best-effort so non-Proteva environments (no plm package) still work.
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
        # ProtevaConfig has no problem_type in its __init__ signature; the head
        # reads it via getattr, so set it as a plain attribute when supplied.
        problem_type = kwargs.pop("problem_type", None)
        if problem_type is not None:
            config.problem_type = problem_type
        # Load fp32 (the RoPE cache must be computed in fp32, NOT bf16 -> NaN);
        # HF Trainer casts to bf16 for the train/eval forward when bf16=True.
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
    # ``problem_type`` is a config attribute, not a model __init__ kwarg; set it
    # on the config object directly so it isn't forwarded to the model constructor.
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


# Per-model-family LoRA target module names (the body's attention + FFN
# Linears). Resolved by ``config.model_type``. NOT "all-linear" — that would
# also wrap the embedding, the MLM decoder, and the aux heads, which we want
# left frozen (we measure the BODY's adaptability; the classifier head trains
# separately via ``modules_to_save``).
#
# Proteva — ALL body Linears (QLoRA best practice: adapt every linear projection,
# not just attention) PLUS the value embeddings. Verified by named_modules() on
# the checkpoint: 24× each of wq/wk/wv/wo + attn_gate (attention, incl. the
# --head-gate Linear) and w12 (packed gate+value) / w3 (down) (SwiGLU FFN), plus
# ve_first/ve_last (the value-embedding nn.Embeddings — PEFT wraps these with
# LoRA Embedding adapters). The MLM ``decoder`` and pretraining aux heads
# (di3_head/cons_head/…) are deliberately NOT listed — the downstream classifier
# trains separately via ``modules_to_save``.
PROTEVA_LORA_TARGETS = ["wq", "wk", "wv", "wo", "attn_gate", "w12", "w3",
                        "ve_first", "ve_last"]
# AMPLIFY (chandar-lab/AMPLIFY_120M remote code): SEPARATE attention projections
# ``q``/``k``/``v``/``wo`` (NOT fused ``Wqkv``) + SwiGLU FFN ``w12``/``w3``.
# Verified by named_modules() on the 120M checkpoint: 24× each of q/k/v/wo/w12/w3.
# (``Wqkv``/``fc1``/``fc2`` are other AMPLIFY revs and don't exist here → would
# silently match nothing, leaving q/k/v unadapted.)
AMPLIFY_LORA_TARGETS = ["q", "k", "v", "wo", "w12", "w3"]
# ESM-2 (HF): self-attention q/k/v/out + FFN intermediate/output dense.
ESM_LORA_TARGETS = ["query", "key", "value", "dense"]


def lora_target_modules(model_type: str | None) -> list[str] | str:
    """Return the LoRA ``target_modules`` for a model family.

    Falls back to ``"all-linear"`` for unknown families (PEFT's own default),
    which is safe for vanilla encoders but is explicitly NOT used for Proteva
    (see ``PROTEVA_LORA_TARGETS``).
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

    Proteva: ``encoder.blocks`` (plm/model.py). HF ESM: ``encoder.layer``.
    AMPLIFY: ``transformer_encoder``. Returns ``None`` if not found (caller then
    treats last-N as a no-op and warns)."""
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

    * ``probe``: freeze the encoder body (``model.base_model``); train only the head.
    * ``full``: train everything (no-op).
    * ``lora``: wrap the body with PEFT LoRA using model-type-aware target
      modules (``r``, ``alpha`` configurable); the classifier head is kept
      trainable + saved via ``modules_to_save``.
    * ``last_n``: freeze the body, then unfreeze the top ``N`` encoder blocks +
      the encoder's final norm; the head is always trainable.

    Returns the (possibly PEFT-wrapped) model. The accepted ``mode`` aliases
    ``"lastn"`` -> ``"last_n"`` so the shell driver's token matches the script.
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
            # Final norm sits after the last block; unfreeze it too.
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
            # bias="none" (QLoRA standard): LoRA adapts no bias terms; base-model
            # biases stay frozen. Regression's intercept is NOT lost — the task
            # head (Linear with bias) is fully trained via modules_to_save.
            bias="none",
            # rank-stabilized LoRA: scale by alpha/sqrt(r) not alpha/r, so higher
            # r (we run r=64 for ~16% trainable) actually pays off + stays stable.
            use_rslora=True,
            task_type=task_type,
            target_modules=lora_target_modules(model_type),
            # Keep the classification head fully trainable + saved with the
            # adapter (it is NOT in target_modules, so PEFT would otherwise
            # freeze it). We measure the body's adaptability; the head learns
            # alongside in fp32.
            modules_to_save=[classifier_module],
        )
        # Wrap the WHOLE head model (not just ``model.base_model``): the head's
        # ``forward`` (pooling + classifier + loss) must remain the entry point,
        # and ``PeftModel.forward`` delegates to it. ``target_modules`` match by
        # name suffix anywhere in the tree, so they still hit ONLY the encoder
        # body Linears (wq/wk/wv/wo, w12/w3). Wrapping the bare encoder instead
        # routes through PEFT's task-specific forward, which reads
        # ``self.config.use_return_dict`` — absent on the encoder's plain
        # ``EncoderConfig`` dataclass — and crashes. The PeftModel exposes
        # ``.config`` (the ProtevaConfig) and ``save_pretrained`` (adapter only),
        # so the Trainer + JSONL/adapter-save paths are unaffected.
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
    # Discriminative-tier (LoRA / last-N) hyper-parameters. Shared by the
    # sequence + residue scripts. Spec defaults: r=16, alpha=32 (=2r), last_n=4.
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


# Metrics where LOWER is better, for early-stopping / best-model selection.
# Everything else our tasks use (AUC / F1* / Spearman / Pearson / Accuracy / MCC
# / Recall@k / AP) is higher-is-better. NOTE: e.g. ``meltome`` is a regression
# task whose main_metric is MSE — hardcoding greater_is_better=True would make
# early stopping keep the WORST epoch, so the direction must follow the metric.
_LOWER_IS_BETTER_METRICS = {"mse", "mae", "rmse", "loss", "perplexity"}


def metric_greater_is_better(metric_name: str | None) -> bool:
    return (metric_name or "").lower() not in _LOWER_IS_BETTER_METRICS


def build_training_args(
    args: argparse.Namespace, output_dir: Path, *, main_metric: str | None = None
) -> TrainingArguments:
    """TrainingArguments for the FT scripts.

    When ``args.early_stop`` is set AND ``main_metric`` is given, switch on
    per-epoch eval + ``load_best_model_at_end`` (paired with an
    ``EarlyStoppingCallback`` in the driver). ``num_train_epochs`` then acts as
    the MAX-epoch cap, not a fixed count: easy/regression tasks stop after the
    metric plateaus (avoiding overfit), hard tasks (e.g. 1195-class fold) get
    the epochs they need. All our main metrics (F1_Macro/Spearman/AUC/Accuracy)
    are higher-is-better. Without early-stop it keeps the old fixed-epoch /
    no-eval / keep-last behavior."""
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
        # bf16 forward is too coarse for regression: the MSE loss is fp32 but the
        # LOGITS come from a bf16 forward (~3 sig digits) -> the head cannot rank
        # fine fitness differences -> constant predictions, Spearman~0. --fp32
        # (set by run_full_bench for regression tasks) keeps the whole forward fp32.
        bf16=torch.cuda.is_bf16_supported() and not getattr(args, "fp32", False),
        dataloader_num_workers=args.dataloader_num_workers,
    )
    if getattr(args, "early_stop", False) and main_metric is not None:
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


def write_jsonl_record(output_dir: Path, prefix: str, ckpt: str, record: Dict[str, Any]) -> Path:
    jsonl_path = output_dir.parent / f"{prefix}_{safe_ckpt(ckpt)}.jsonl"
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return jsonl_path
