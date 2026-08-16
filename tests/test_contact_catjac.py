"""Categorical Jacobian: coupling extraction and APC background.

Offline: a stub model stands in for the MLM, so no weights are downloaded.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from contact_catjac import CANONICAL_AA, categorical_jacobian
from contact_metrics import MIN_SEPARATION, separation_matrix

LENGTH = 24
COUPLED = (3, 19)


class StubTokenizer:
    """One token per residue, ids 0..19, plus a CLS/SEP pair."""

    vocab = {aa: i for i, aa in enumerate(CANONICAL_AA)}
    unk_token_id = None

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __call__(self, seq, **kwargs):
        ids = [20] + [self.vocab[c] for c in seq] + [21]
        special = [1] + [0] * len(seq) + [1]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "special_tokens_mask": torch.tensor([special]),
        }


class StubRefs:
    """Each position predicts itself with strength ``self_strength`` (the
    diagonal), plus one planted coupling COUPLED[0] -> COUPLED[1]."""

    def __init__(self, self_strength=1.0, coupling=5.0):
        self.self_strength = self_strength
        self.coupling = coupling

    def forward_logits(self, input_ids, attention_mask=None):
        batch, tokens = input_ids.shape
        out = torch.zeros(batch, tokens, 22)
        residues = input_ids[:, 1:-1]
        for pos in range(LENGTH):
            out[:, pos + 1, :20] = (
                torch.nn.functional.one_hot(residues[:, pos], 22)[:, :20].float()
                * self.self_strength
            )
        src = residues[:, COUPLED[0]]
        out[:, COUPLED[1] + 1, :20] += (
            torch.nn.functional.one_hot(src, 22)[:, :20].float() * self.coupling
        )
        return out


@pytest.fixture
def sequence():
    rng = np.random.RandomState(0)
    return "".join(CANONICAL_AA[i] for i in rng.randint(0, 20, size=LENGTH))


def _scores(sequence, **kwargs):
    return categorical_jacobian(
        StubRefs(**kwargs), StubTokenizer(), sequence, batch_size=40
    )


def test_recovers_the_planted_coupling(sequence):
    scores = _scores(sequence)
    upper = np.triu(np.ones((LENGTH, LENGTH), dtype=bool), k=MIN_SEPARATION)
    best = np.unravel_index(np.argmax(np.where(upper, scores, -np.inf)), scores.shape)
    assert tuple(sorted(best)) == COUPLED


def test_output_is_symmetric_with_a_zeroed_near_diagonal(sequence):
    scores = _scores(sequence)
    assert scores.shape == (LENGTH, LENGTH)
    assert np.allclose(scores, scores.T)
    assert not scores[separation_matrix(LENGTH) < MIN_SEPARATION].any()


def test_self_coupling_does_not_influence_contact_scores(sequence):
    """A residue's coupling to ITSELF is not evidence about any pair, so it must
    be removed before APC builds its rank-1 background -- otherwise the (large)
    diagonal inflates the row/column sums and shifts every off-diagonal score.
    """
    weak = _scores(sequence, self_strength=1.0)
    strong = _scores(sequence, self_strength=40.0)
    assert np.allclose(weak, strong, atol=1e-8), (
        "changing only the self-coupling changed the pair scores, so the "
        "diagonal is still feeding the APC background"
    )


def test_stronger_coupling_widens_that_pair_margin(sequence):
    """APC leaves the background near zero and can push it negative, so compare
    the MARGIN over the runner-up, not a ratio."""
    faint = _scores(sequence, coupling=1.0)
    loud = _scores(sequence, coupling=20.0)
    i, j = COUPLED
    others = np.triu(np.ones((LENGTH, LENGTH), dtype=bool), k=MIN_SEPARATION).copy()
    others[i, j] = False
    assert loud[i, j] - loud[others].max() > faint[i, j] - faint[others].max()


def test_rejects_a_tokenizer_that_does_not_map_one_token_per_residue(sequence):
    class Doubling(StubTokenizer):
        def __call__(self, seq, **kwargs):
            return super().__call__(seq * 2, **kwargs)

    with pytest.raises(ValueError, match="one token per residue"):
        categorical_jacobian(StubRefs(), Doubling(), sequence)


def test_rejects_a_tokenizer_missing_an_amino_acid(sequence):
    class Missing(StubTokenizer):
        def convert_tokens_to_ids(self, token):
            return None if token == "W" else super().convert_tokens_to_ids(token)

    with pytest.raises(ValueError, match="no single-token entry"):
        categorical_jacobian(StubRefs(), Missing(), sequence)


def test_result_tracker_creates_a_missing_output_directory(tmp_path):
    """`contact_catjac` scores for ~50 minutes before it writes anything. If
    saving is the first thing to touch the output directory, a typo in `-o`
    throws all of that away at the very last step, which is exactly what
    happened. Creating it on save makes the failure impossible for every caller.
    """
    from protein_benchmark_suite import ResultTracker

    tracker = ResultTracker("some/model")
    tracker.add("Contact Prediction (Categorical Jacobian)", {"P@L/5_long": 0.45}, 40)
    target = tmp_path / "does" / "not" / "exist"
    path = tracker.save(str(target))
    assert target.is_dir()
    assert path is not None and Path(path).exists()
