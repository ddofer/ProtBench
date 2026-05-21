"""Shared helpers for HF Trainer-based fine-tuning (residue + sequence scripts).

Pure functions used by ``finetune_residue.py`` and ``finetune_sequence.py``:
encoder loading with AMPLIFY ``model_type`` quirk, label decoders for
string and CSV residue labels, and special-token-aware label alignment
for token-classification.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from transformers import AutoConfig


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
    """Convert a per-residue string label to ints via ``alphabet.index``.

    e.g. ``decode_string_label("HHEEC", "HEC") == [0, 0, 1, 1, 2]``.
    """
    return [alphabet.index(c) for c in label_str]


def decode_csv_label(label_str: str) -> List[int]:
    """Parse a comma-separated string of ints, e.g. ``"0, 4, 6, 1"`` → ``[0,4,6,1]``."""
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
            try:
                aligned.append(int(next(res_iter)))
            except StopIteration:
                # Defensive: tokenizer added more non-special tokens than
                # residue labels (shouldn't happen for AA-per-token
                # tokenizers without truncation mismatch). Pad with -100.
                aligned.append(-100)
    return aligned
