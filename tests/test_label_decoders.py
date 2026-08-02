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
