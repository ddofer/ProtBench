"""Pairwise contact probe: feature construction, sampling, label alignment.

Offline: a stub encoder stands in for the model, so nothing downloads.
"""

import numpy as np
import pytest

from contact_metrics import MIN_SEPARATION, contacts_from_tertiary
from contact_probe import _labels_for, _sample_training_pairs, pair_features


def _coords(length, seed=0):
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.randn(length, 3) * 2.0, axis=0)


def test_pair_features_are_symmetric():
    """A contact map is undirected; the probe must not see pair order."""
    rng = np.random.RandomState(0)
    residues = rng.randn(30, 16).astype("float32")
    i = np.array([1, 5, 9, 20])
    j = np.array([14, 2, 25, 3])
    assert np.allclose(pair_features(residues, i, j), pair_features(residues, j, i))


def test_pair_feature_width():
    residues = np.zeros((10, 16), dtype="float32")
    feats = pair_features(residues, np.array([0]), np.array([7]))
    assert feats.shape == (1, 16 * 2 + 1)


def test_pair_features_encode_separation():
    residues = np.ones((40, 4), dtype="float32")
    near = pair_features(residues, np.array([0]), np.array([6]))[0, -1]
    far = pair_features(residues, np.array([0]), np.array([39]))[0, -1]
    assert far > near


def test_training_sample_is_balanced_and_upper_triangular():
    length = 80
    contacts, valid = contacts_from_tertiary(_coords(length))
    idx_i, idx_j, y = _sample_training_pairs(
        contacts, valid, length, np.random.RandomState(0)
    )
    assert y.size > 0
    assert y.sum() * 2 == y.size, "positives and negatives must be equal"
    assert (idx_i < idx_j).all(), "pairs must come from the upper triangle only"
    assert (np.abs(idx_i - idx_j) >= MIN_SEPARATION).all()
    assert contacts[idx_i[y == 1], idx_j[y == 1]].all()
    assert not contacts[idx_i[y == 0], idx_j[y == 0]].any()


def test_training_sample_respects_valid_mask():
    length = 80
    mask = np.ones(length, dtype=bool)
    mask[10:20] = False
    contacts, valid = contacts_from_tertiary(_coords(length), mask)
    idx_i, idx_j, _ = _sample_training_pairs(
        contacts, valid, length, np.random.RandomState(0)
    )
    assert not ((idx_i >= 10) & (idx_i < 20)).any()
    assert not ((idx_j >= 10) & (idx_j < 20)).any()


def test_training_sample_empty_when_no_contacts():
    """A fully extended chain has no long-range contacts; return nothing rather
    than fabricating an unbalanced sample."""
    coords = np.zeros((40, 3))
    coords[:, 0] = np.arange(40) * 20.0
    contacts, valid = contacts_from_tertiary(coords)
    _, _, y = _sample_training_pairs(contacts, valid, 40, np.random.RandomState(0))
    assert y.size == 0


def test_labels_crop_to_encoder_length():
    """max_length truncation shortens the embedding; labels must follow, or the
    contact map and the residue rows silently misalign."""
    length = 60
    coords = _coords(length)
    record = {
        "seq": "A" * length,
        "tertiary": coords,
        "valid_mask": np.ones(length, dtype=bool),
    }
    contacts, valid = _labels_for(record, 25)
    assert contacts.shape == (25, 25)
    assert valid.shape == (25, 25)
    full, _ = _labels_for(record, length)
    assert (contacts == full[:25, :25]).all()


def test_labels_tolerate_missing_valid_mask():
    record = {"seq": "A" * 30, "tertiary": _coords(30)}
    _, valid = _labels_for(record, 30)
    assert valid[0, 10]


HIDDEN = 4


class _StubTokenizer:
    """One token per residue plus a CLS/SEP pair, so residue bookkeeping is
    exercised without downloading a real tokenizer."""

    def __call__(self, seqs, **kwargs):
        import torch

        longest = max(len(s) for s in seqs) + 2
        ids = torch.zeros(len(seqs), longest, dtype=torch.long)
        attn = torch.zeros(len(seqs), longest, dtype=torch.long)
        special = torch.ones(len(seqs), longest, dtype=torch.long)
        for row, seq in enumerate(seqs):
            attn[row, : len(seq) + 2] = 1
            special[row, 1 : len(seq) + 1] = 0
            ids[row, 1 : len(seq) + 1] = torch.arange(len(seq)) + 1
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "special_tokens_mask": special,
        }


class _StubEncoder:
    config = None

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        states = input_ids.unsqueeze(-1).float().repeat(1, 1, HIDDEN)

        class _Out:
            last_hidden_state = states

        return _Out()


def test_iter_residue_embeddings_preserves_protein_boundaries():
    """The contact path needs one array per protein, not one stacked block."""
    from token_classification_probe import (
        extract_residue_embeddings,
        iter_residue_embeddings,
    )

    hidden = HIDDEN
    sequences = ["ACDEF", "GHIKLMN", "PQ"]

    arrays = list(
        iter_residue_embeddings(
            encoder=_StubEncoder(),
            tokenizer=_StubTokenizer(),
            sequences=sequences,
            batch_size=2,
        )
    )
    assert len(arrays) == len(sequences)
    assert [a.shape for a in arrays] == [(len(s), hidden) for s in sequences]

    # The stacked helper must still return exactly the concatenation.
    labels = [list(range(len(s))) for s in sequences]
    X, y, _ = extract_residue_embeddings(
        encoder=_StubEncoder(),
        tokenizer=_StubTokenizer(),
        sequences=sequences,
        labels=labels,
        batch_size=2,
    )
    assert X.shape == (sum(len(s) for s in sequences), hidden)
    assert np.allclose(X, np.concatenate(arrays, axis=0))
    assert y.tolist() == [i for s in sequences for i in range(len(s))]


def test_contact_probe_selfcheck_runs():
    from contact_probe import _selfcheck

    _selfcheck()


def test_categorical_jacobian_recovers_a_planted_coupling():
    """The catjac selfcheck runs the real algorithm against a stub model whose
    logits at one position depend on another, and asserts that exact pair comes
    out on top. Offline -- no weights, no download."""
    from contact_catjac import _selfcheck

    _selfcheck()


def _stub_records(count, length=50, seed=0):
    rng = np.random.RandomState(seed)
    records = []
    for i in range(count):
        coords = _coords(length, seed=seed + i)
        records.append(
            {
                "seq": "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=length)),
                "tertiary": coords,
                "valid_mask": np.ones(length, dtype=bool),
            }
        )
    return records


def test_evaluate_contact_prediction_end_to_end_with_explicit_eval_split():
    from contact_probe import evaluate_contact_prediction

    metrics = evaluate_contact_prediction(
        encoder=_StubEncoder(),
        tokenizer=_StubTokenizer(),
        train_records=_stub_records(8),
        test_records=_stub_records(3, seed=100),
        batch_size=4,
        train_proteins=8,
    )
    assert metrics["Proteins_Scored"] == 3.0
    for key in ("P@L/5_long", "P@L_short", "P@L/2_medium"):
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_contact_prediction_holds_out_proteins_when_no_eval_split():
    """The fallback splits by PROTEIN, not by residue pair -- a pair-level split
    would put (i, j) in train and (j, i) in eval and leak the answer."""
    from contact_probe import evaluate_contact_prediction

    metrics = evaluate_contact_prediction(
        encoder=_StubEncoder(),
        tokenizer=_StubTokenizer(),
        train_records=_stub_records(10),
        test_records=None,
        batch_size=4,
        train_proteins=10,
    )
    assert metrics["Proteins_Scored"] == 2.0  # 20% of 10


def test_evaluate_contact_prediction_refuses_a_too_small_holdout():
    from contact_probe import evaluate_contact_prediction

    with pytest.raises(ValueError, match="at least 5 training proteins"):
        evaluate_contact_prediction(
            encoder=_StubEncoder(),
            tokenizer=_StubTokenizer(),
            train_records=_stub_records(3),
            test_records=None,
        )


@pytest.mark.parametrize("length", [3, MIN_SEPARATION])
def test_short_proteins_yield_no_training_pairs(length):
    coords = _coords(length)
    contacts, valid = contacts_from_tertiary(coords)
    _, _, y = _sample_training_pairs(
        contacts, valid, length, np.random.RandomState(0)
    )
    assert y.size == 0
