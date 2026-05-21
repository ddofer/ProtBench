"""Tests for align_labels_with_tokens — special-token-aware label alignment."""

from _hf_finetune_common import align_labels_with_tokens


def test_align_basic_cls_sep():
    # Simulate ``[CLS] A C D [SEP]`` for residues with labels [0, 1, 2].
    input_ids = [101, 5, 7, 9, 102]
    residue_labels = [0, 1, 2]
    special_tokens_mask = [1, 0, 0, 0, 1]
    out = align_labels_with_tokens(input_ids, residue_labels, special_tokens_mask)
    assert out == [-100, 0, 1, 2, -100]
    assert len(out) == len(input_ids)


def test_align_with_padding():
    # ``[CLS] A B [SEP] [PAD] [PAD]`` — padding positions also masked.
    input_ids = [101, 5, 6, 102, 0, 0]
    residue_labels = [3, 4]
    special_tokens_mask = [1, 0, 0, 1, 1, 1]
    out = align_labels_with_tokens(input_ids, residue_labels, special_tokens_mask)
    assert out == [-100, 3, 4, -100, -100, -100]
    assert len(out) == 6


def test_align_passes_through_non_special_positions():
    input_ids = [101, 1, 2, 3, 4, 102]
    residue_labels = [10, 11, 12, 13]
    special_tokens_mask = [1, 0, 0, 0, 0, 1]
    out = align_labels_with_tokens(input_ids, residue_labels, special_tokens_mask)
    assert out == [-100, 10, 11, 12, 13, -100]


def test_align_handles_residue_label_shortage():
    # Defensive: more non-special tokens than residue labels → trailing -100.
    input_ids = [101, 1, 2, 3, 102]
    residue_labels = [9]  # only 1 label for 3 non-special tokens
    special_tokens_mask = [1, 0, 0, 0, 1]
    out = align_labels_with_tokens(input_ids, residue_labels, special_tokens_mask)
    assert out == [-100, 9, -100, -100, -100]
