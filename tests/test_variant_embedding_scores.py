"""Embedding-based variant scorers (indels + multi-mutants): RED and span pooling.

Ported from diploid_glm's covariance_pooling branch (auxfeat.py). Indels cannot
amortize a masked forward across variants the way substitutions can, so a
readout that needs ONE forward per variant is ~k times cheaper than strided
masked PLL (k = --indel_pll_passes, default 32).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from variant_embedding_scores import (  # noqa: E402
    edit_span,
    red,
    score_indel_variants,
    span_pooled_score,
)


def _brute_force_red(X):
    """Reference O(T^2 d) definition: 1 - mean pairwise cosine over the off-diagonal."""
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = U @ U.T
    T = len(X)
    off = (G.sum() - np.trace(G)) / (T * (T - 1))
    return 1.0 - off


def test_identical_residues_have_zero_diversity():
    X = np.tile(np.array([[3.0, -1.0, 2.0]], dtype=np.float32), (8, 1))
    assert red(X) == 0.0


def test_orthogonal_residues_have_unit_diversity():
    X = np.eye(5, dtype=np.float32) * 2.0  # pairwise cosine 0 -> RED 1
    assert abs(red(X) - 1.0) < 1e-6


def test_antipodal_pair_has_diversity_two():
    X = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)  # cosine -1 -> RED 2
    assert abs(red(X) - 2.0) < 1e-6


def test_matches_brute_force_gram_on_random_block():
    rng = np.random.RandomState(0)
    X = rng.randn(64, 32).astype(np.float32)
    assert abs(red(X) - _brute_force_red(X)) < 1e-6


def test_edit_span_locates_a_substitution():
    assert edit_span("ACDEFG", "ACDXFG") == (3, 4)


def test_edit_span_locates_an_insertion():
    start, end = edit_span("ACDEFG", "ACDWWEFG")
    assert (start, end) == (3, 5)  # span in MUTANT coordinates covers the inserted residues


def test_edit_span_locates_a_deletion():
    # WT ACDEFG -> ACFG: 'DE' deleted; mutant span is the empty join point, widened to 1 residue
    start, end = edit_span("ACDEFG", "ACFG")
    assert start == 2 and end >= start + 1


def test_edit_span_of_identical_sequences_is_empty():
    assert edit_span("ACDEFG", "ACDEFG") == (0, 0)


def test_edit_span_spans_multiple_edit_regions():
    # edits at both ends -> enclosing span, so one pooled window covers them all
    start, end = edit_span("AACDEFGG", "AXCDEFGY")
    assert start <= 1 and end >= 8


def _toy_embeddings(seq, rng_seed=0):
    """Deterministic per-residue embedding: each amino acid maps to a fixed vector."""
    rng = np.random.RandomState(rng_seed)
    table = {aa: rng.randn(16).astype(np.float32) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    return np.stack([table[c] for c in seq])


def test_span_pooling_is_dominated_by_the_edit_not_the_rest_of_the_protein():
    """The point of span pooling: a change inside the span must move the score more
    than an equally sized change hundreds of residues away."""
    wt = "ACDEFGHIKL" * 30  # 300 residues
    in_span = wt[:150] + "WWWW" + wt[154:]
    out_span = wt[:10] + "WWWW" + wt[14:]
    span = edit_span(wt, in_span)

    wt_X, in_X, out_X = (_toy_embeddings(s) for s in (wt, in_span, out_span))
    d_in = span_pooled_score(wt_X, in_X, span)
    d_out = span_pooled_score(wt_X, out_X, span)
    assert d_in > d_out
    # whole-sequence pooling cannot tell them apart nearly as well
    whole = (0, len(wt))
    assert d_in - d_out > span_pooled_score(wt_X, in_X, whole) - span_pooled_score(wt_X, out_X, whole)


def test_span_pooled_score_of_an_unchanged_sequence_is_zero():
    X = _toy_embeddings("ACDEFGHIKL" * 5)
    assert span_pooled_score(X, X, (10, 20)) < 1e-9


def test_span_pooled_score_handles_length_change_and_empty_span():
    wt, mut = "ACDEFGHIKL" * 5, "ACDEFGHIKL" * 5
    wt_X, mut_X = _toy_embeddings(wt), _toy_embeddings(mut + "WWW")
    assert span_pooled_score(wt_X, mut_X, (48, 53)) >= 0.0
    # empty span means "no located edit" -> pool the whole sequence instead
    assert span_pooled_score(wt_X, mut_X, (0, 0)) == span_pooled_score(
        wt_X, mut_X, (0, max(len(wt_X), len(mut_X)))
    )


def _fake_embedder(seqs):
    """Stand-in for iter_residue_embeddings: deterministic per-residue vectors."""
    return [_toy_embeddings(s) for s in seqs]


def test_score_indel_variants_returns_one_aligned_score_per_variant():
    wt = "ACDEFGHIKL" * 10
    variants = [wt[:50] + "WWW" + wt[50:], wt[:20] + wt[24:], wt]
    scores = score_indel_variants(wt, variants, _fake_embedder)
    assert len(scores) == len(variants)
    assert scores[2] < 1e-9  # the unchanged "variant" scores as no disruption
    assert scores[0] > 0.0 and scores[1] > 0.0


def test_score_indel_variants_supports_the_red_arm():
    wt = "ACDEFGHIKL" * 10
    variants = [wt[:50] + "WWWWWWWWWW" + wt[50:], wt]
    scores = score_indel_variants(wt, variants, _fake_embedder, arm="red")
    assert len(scores) == 2
    assert scores[1] == 0.0  # RED(mut) - RED(wt) == 0 for an unchanged sequence
    assert scores[0] != 0.0


def test_score_indel_variants_skips_over_length_sequences_like_the_pll_path():
    wt = "ACDEFGHIKL" * 5
    variants = [wt[:10] + "WWW" + wt[10:], "A" * 500]
    scores = score_indel_variants(wt, variants, _fake_embedder, model_window=100)
    assert scores[0] is not None
    assert scores[1] is None  # matches strided_masked_pll_table's skip contract


def test_score_indel_variants_embeds_each_sequence_exactly_once():
    wt = "ACDEFGHIKL" * 5
    variants = [wt[:10] + "W" + wt[10:], wt[:20] + "WW" + wt[20:]]
    calls = []

    def counting_embedder(seqs):
        calls.append(list(seqs))
        return _fake_embedder(seqs)

    score_indel_variants(wt, variants, counting_embedder)
    # one batched call, WT + variants, no per-variant re-embedding of the WT
    assert len(calls) == 1
    assert len(calls[0]) == len(variants) + 1


def test_indel_score_modes_include_the_embedding_arms():
    """The embedding arms are reachable from the CLI, alongside the PLL ones."""
    import proteingym_mlm_zeroshot as pz

    parser = pz._build_parser() if hasattr(pz, "_build_parser") else None
    if parser is None:  # parser built inline in main(); inspect the choices constant
        assert set(pz.INDEL_SCORE_MODES) >= {"strided", "embedding_span", "embedding_red"}
        return
    action = next(a for a in parser._actions if a.dest == "indel_score_mode")
    assert {"embedding_span", "embedding_red"} <= set(action.choices)


def test_embedder_wiring_yields_one_matrix_per_sequence():
    """_make_residue_embedder adapts a MaskedLM to the (T, d) contract the arms need."""
    import sys as _sys

    _sys.path.insert(0, str(_BENCH / "tests"))
    from test_token_classification_probe import _TinyEncoder, _TinyTokenizer

    from proteingym_mlm_zeroshot import _make_residue_embedder

    embedder = _make_residue_embedder(_TinyEncoder(hidden=8), _TinyTokenizer(), "cpu", 4, 64)
    out = embedder(["ACDEF", "ACD"])
    assert [x.shape[0] for x in out] == [5, 3]
    assert out[0].shape[1] == 8


def test_red_arm_is_negated_to_match_the_shared_convention():
    """Measured on ProteinGym indels (ESM-C 300M): residue diversity RISES with
    fitness, the opposite of the DNA-domain assumption the port came with. The
    raw delta is therefore flipped so every arm reads "higher = more disrupted".
    A scorer wired with the wrong sign is worse than no scorer."""
    wt = "ACDEFGHIKLMNPQRSTVWY" * 5
    mut = wt[:40] + "WWWW" + wt[44:]
    wt_X, mut_X = _toy_embeddings(wt), _toy_embeddings(mut)
    raw_delta = red(mut_X) - red(wt_X)
    (score,) = score_indel_variants(wt, [mut], _fake_embedder, arm="red")
    assert score == -raw_delta != 0.0
