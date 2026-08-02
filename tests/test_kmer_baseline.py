"""The k-mer baseline must be usable as a model, with a split-independent vocab."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protein_benchmark_suite as pbs  # noqa: E402
from kmer_baseline import kmer_features, parse_kmer_model_name  # noqa: E402


@pytest.mark.parametrize(
    "name,expected",
    [("kmer", 3), ("KMER", 3), ("kmer2", 2), ("kmer5", 5), ("esm2", None), ("", None)],
)
def test_parse_kmer_model_name(name, expected):
    assert parse_kmer_model_name(name) == expected


def test_feature_width_is_fixed_by_k_not_by_the_data():
    """Train and test must land in the same space even with disjoint content."""
    train = kmer_features(["AAAA", "CCCC"], k=2)
    test = kmer_features(["WWWW"], k=2)
    assert train.shape[1] == test.shape[1] == 400


def test_frequencies_sum_to_one_and_ignore_length():
    """A repeat and a longer repeat of the same residue must be identical."""
    short, long = kmer_features(["AAAA", "AAAAAAAAAA"], k=2)
    assert np.isclose(short.sum(), 1.0)
    assert np.allclose(short, long)


def test_nonstandard_residues_are_dropped_not_substituted():
    """AXA must not contribute an AA 2-mer -- that k-mer is not in the sequence."""
    (with_x,) = kmer_features(["AXA"], k=2)
    assert with_x.sum() == 0.0


def test_empty_sequence_yields_zeros_without_dividing_by_zero():
    (row,) = kmer_features([""], k=3)
    assert row.shape == (8000,)
    assert not np.isnan(row).any()


def test_suite_embeds_sequences_via_the_kmer_path():
    """load_model('kmer') must produce something embed_sequences can consume."""
    model_obj, is_sbert = pbs.load_model("kmer2", device="cpu")[:2]
    embs = pbs.embed_sequences(model_obj, is_sbert, ["ACDEF", "WWWW"], device="cpu")
    assert embs.shape == (2, 400)
    assert embs.dtype == np.float32


def test_large_k_refuses_with_arithmetic_not_oom():
    """k=6 on any real workload is terabytes; fail fast and say why."""
    with pytest.raises(ValueError, match="GiB"):
        kmer_features(["ACDEF"] * 10_000, k=6)


def test_vocab_is_cached_across_calls():
    from kmer_baseline import _vocab

    assert _vocab(2) is _vocab(2)


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be"):
        kmer_features(["ACDEF"], k=0)
