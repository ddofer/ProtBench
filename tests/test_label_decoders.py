"""Unit tests for label-decoder helpers in _hf_finetune_common."""

import pytest

from _hf_finetune_common import decode_csv_label, decode_string_label


def test_decode_string_label_ss3():
    assert decode_string_label("HHEECCC", "HEC") == [0, 0, 1, 1, 2, 2, 2]


def test_decode_string_label_disorder():
    assert decode_string_label("01100", "01") == [0, 1, 1, 0, 0]


def test_decode_csv_label_basic():
    assert decode_csv_label("0, 4, 6, 1") == [0, 4, 6, 1]


def test_decode_csv_label_handles_whitespace_and_empty():
    assert decode_csv_label("1,2,3") == [1, 2, 3]
    assert decode_csv_label(" 7 ,  8 ") == [7, 8]
    assert decode_csv_label("") == []


def test_decode_string_label_bad_char_raises():
    with pytest.raises(ValueError):
        decode_string_label("HHX", "HEC")


# ---------------------------------------------------------------------------
# SS8 in the fine-tuning path
# ---------------------------------------------------------------------------


def test_finetune_ss8_uses_the_same_eight_classes_as_the_frozen_probe():
    """A fine-tuned SS8 model and an SS8 linear probe must report against the
    same class ids, or their numbers cannot be put in the same table."""
    from finetune_residue import _SS8_ALPHABET as ft_alphabet
    from protein_benchmark_suite import _SS8_ALPHABET as probe_alphabet

    assert ft_alphabet == probe_alphabet


def test_finetune_ss8_marks_unassigned_residues_as_ignored():
    """GleghornLab/SS8 uses `D` for unassigned termini. `decode_string_label`
    would raise on it, so the ss8 branch needs its own handling -- and must emit
    HF's -100 rather than dropping the position and shifting every later label."""
    from finetune_residue import _decode_labels

    assert _decode_labels("ss8", "DDCHH") == [-100, -100, 7, 1, 1]


def test_finetune_ss8_decode_preserves_length():
    from finetune_residue import _decode_labels

    label = "DDDCHHHEEGGITTSSBBCDD"
    assert len(_decode_labels("ss8", label)) == len(label)


def test_finetune_ss8_label_names_exclude_the_unassigned_marker():
    from finetune_residue import _build_label_meta

    meta = _build_label_meta("ss8", [[0, 1, 2]])
    assert meta["num_labels"] == 8
    assert "D" not in meta["names"]
