import numpy as np
import pytest

from ptm_benchmark import score_ptm_sites
from ptm_frozen_probes import (
    PROBE_MAX_ITER,
    FrozenSequenceExample,
    FrozenSiteExample,
    fit_center_residue_baseline,
    fit_residue_identity_baseline,
    fit_sequence_probe,
    fit_site_probe,
    predict_center_residue_baseline,
    predict_residue_identity_baseline,
    predict_sequence_probe,
    predict_site_probe,
    fit_scaled_probe,
    probe_n_iter,
    sampled_fit_positions,
)


def test_site_sampling_is_deterministic_and_retains_negative_only_proteins():
    mixed = FrozenSiteExample("mixed", "AAAA", (0, 1, 0, 0))
    negative = FrozenSiteExample("negative", "AAA", (0, 0, 0))
    assert sampled_fit_positions(mixed, seed=3, negatives_per_positive=2) == (
        sampled_fit_positions(mixed, seed=3, negatives_per_positive=2)
    )
    assert len(sampled_fit_positions(negative, seed=3, negatives_per_positive=2)) == 1


def test_site_sampling_can_restrict_a_phosphosite_task_to_sty_candidates():
    example = FrozenSiteExample("phospho", "ASGTKY", (0, 1, 0, 1, 0, 0))
    positions = sampled_fit_positions(
        example,
        seed=3,
        negatives_per_positive=5,
        candidate_residues=frozenset("STY"),
    )
    assert positions == [1, 3, 5]

    invalid = FrozenSiteExample("invalid", "AS", (1, 0))
    with pytest.raises(ValueError, match="outside candidate residues"):
        sampled_fit_positions(
            invalid,
            seed=3,
            negatives_per_positive=5,
            candidate_residues=frozenset("STY"),
        )
    assert sampled_fit_positions(
        invalid,
        seed=3,
        negatives_per_positive=5,
        candidate_residues=frozenset("STY"),
        outside_candidate_positives="ignore",
    ) == [1]


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


def test_site_probe_can_fit_and_predict_only_candidate_residues():
    examples = [
        FrozenSiteExample("p1", "STAA", (1, 0, 0, 0)),
        FrozenSiteExample("p2", "ASTY", (0, 1, 0, 1)),
    ]
    embedded = [
        np.asarray([[2, 0], [0, 2], [0, 1], [0, 1]], dtype=np.float32),
        np.asarray([[0, 1], [2, 0], [0, 1], [2, 0]], dtype=np.float32),
    ]
    candidate_residues = frozenset("STY")
    probe, fit = fit_site_probe(
        examples,
        iter(embedded),
        negatives_per_positive=2,
        candidate_residues=candidate_residues,
    )
    predictions = predict_site_probe(
        examples,
        iter(embedded),
        probe,
        candidate_residues=candidate_residues,
    )
    assert fit == {
        "fit_proteins": 2,
        "fit_residues": 5,
        "fit_positive": 3,
        "fit_negative": 2,
        "fit_max_iter": PROBE_MAX_ITER,
        "fit_n_iter": fit["fit_n_iter"],
    }
    assert 0 < fit["fit_n_iter"] <= PROBE_MAX_ITER
    assert [(item.row_id, item.position) for item in predictions] == [
        ("p1", 0),
        ("p1", 1),
        ("p2", 1),
        ("p2", 2),
        ("p2", 3),
    ]


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


def test_center_residue_baseline_uses_only_the_candidate_site():
    train = [
        FrozenSequenceExample("p1", "AAKAA", 1),
        FrozenSequenceExample("p2", "AAKAA", 1),
        FrozenSequenceExample("n1", "AAKAA", 0),
        FrozenSequenceExample("n2", "AASAA", 0),
    ]
    probabilities = fit_center_residue_baseline(train)
    predictions = predict_center_residue_baseline(train, probabilities)

    assert probabilities["K"] > probabilities["S"]
    assert [prediction.position for prediction in predictions] == [2, 2, 2, 2]
    assert score_ptm_sites(predictions)["n_positive"] == 2


def test_scaled_probe_beats_the_unscaled_low_iteration_fit():
    """Raw lbfgs underfits badly conditioned embeddings; the scaled pipeline does not."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    labels = np.tile([0, 1], 200)
    signal = (labels * 2.0 - 1.0) + rng.normal(scale=0.5, size=labels.size)
    features = np.stack(
        [signal * 1e-4, rng.normal(scale=1e4, size=labels.size)], axis=1
    )

    probe = fit_scaled_probe(features, labels, seed=1337)
    assert isinstance(probe[0], StandardScaler)
    assert probe[-1].max_iter == PROBE_MAX_ITER
    assert probe[-1].class_weight == "balanced"
    assert probe[-1].random_state == 1337
    assert probe_n_iter(probe) <= PROBE_MAX_ITER

    unscaled = LogisticRegression(
        max_iter=300, class_weight="balanced", random_state=1337, solver="lbfgs"
    ).fit(features, labels)
    assert (probe.predict(features) == labels).mean() > 0.9
    assert (unscaled.predict(features) == labels).mean() < 0.6
