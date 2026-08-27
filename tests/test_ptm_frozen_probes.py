import numpy as np
import pytest

from ptm_benchmark import score_ptm_sites
from ptm_frozen_probes import (
    FrozenSequenceExample,
    FrozenSiteExample,
    fit_residue_identity_baseline,
    fit_sequence_probe,
    fit_site_probe,
    predict_residue_identity_baseline,
    predict_sequence_probe,
    predict_site_probe,
    sampled_fit_positions,
)


def test_site_sampling_is_deterministic_and_retains_negative_only_proteins():
    mixed = FrozenSiteExample("mixed", "AAAA", (0, 1, 0, 0))
    negative = FrozenSiteExample("negative", "AAA", (0, 0, 0))
    assert sampled_fit_positions(mixed, seed=3, negatives_per_positive=2) == (
        sampled_fit_positions(mixed, seed=3, negatives_per_positive=2)
    )
    assert len(sampled_fit_positions(negative, seed=3, negatives_per_positive=2)) == 1


def test_site_probe_and_residue_baseline_emit_aligned_predictions():
    examples = [
        FrozenSiteExample("p1", "STAA", (1, 0, 0, 0)),
        FrozenSiteExample("p2", "ASTY", (0, 1, 0, 1)),
    ]
    embedded = [
        np.asarray([[2, 0], [0, 2], [0, 1], [0, 1]], dtype=np.float32),
        np.asarray([[0, 1], [2, 0], [0, 1], [2, 0]], dtype=np.float32),
    ]
    probe, fit = fit_site_probe(examples, iter(embedded), negatives_per_positive=2)
    predictions = predict_site_probe(examples, iter(embedded), probe)
    assert fit["fit_positive"] == 3
    assert len(predictions) == 8
    assert score_ptm_sites(predictions)["n_positive"] == 3

    baseline = fit_residue_identity_baseline(examples)
    baseline_predictions = predict_residue_identity_baseline(examples, baseline)
    assert len(baseline_predictions) == len(predictions)


def test_alignment_mismatch_fails_instead_of_truncating():
    examples = [FrozenSiteExample("p1", "ST", (1, 0))]
    with pytest.raises(ValueError, match="alignment mismatch"):
        fit_site_probe(examples, [np.ones((1, 2), dtype=np.float32)])


def test_sequence_probe_scores_one_record_per_example():
    examples = [
        FrozenSequenceExample("n1", "AK", 0),
        FrozenSequenceExample("p1", "KK", 1),
        FrozenSequenceExample("n2", "AA", 0),
        FrozenSequenceExample("p2", "KA", 1),
    ]
    x = np.asarray([[0, 1], [1, 0], [0, 2], [2, 0]], dtype=np.float32)
    probe = fit_sequence_probe(examples, x)
    predictions = predict_sequence_probe(examples, x, probe, position=30)
    assert [prediction.position for prediction in predictions] == [30] * 4
    assert score_ptm_sites(predictions)["n_positive"] == 2
