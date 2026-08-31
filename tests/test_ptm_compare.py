from __future__ import annotations

import pytest

from ptm_benchmark import PTMSitePrediction, score_ptm_sites
from ptm_compare import (
    build_comparison,
    build_external_comparison,
    paired_site_auprc_bootstrap,
)


def _report(tag: str, auprc: float, *, n_sites: int = 100) -> dict[str, object]:
    return {
        "model_tag": tag,
        "protocol": "fixture",
        "seed": 7,
        "fit_rows": 8,
        "validation_rows": 2,
        "modes": {
            "canonical": {
                "frozen_probe": {
                    "site": {"n_sites": n_sites, "n_positive": 5, "auprc": auprc},
                    "type": {"n_total": 4, "top1": 0.5},
                }
            }
        },
    }


def test_comparison_validates_panel_and_computes_baseline_delta() -> None:
    comparison = build_comparison(
        [_report("vanilla", 0.2), _report("trained", 0.3)],
        baselines={"trained": "vanilla"},
    )
    rows = comparison["rows"]
    trained_site = next(
        row for row in rows if row["model_tag"] == "trained" and row["task"] == "site"
    )
    assert trained_site["delta_auprc"] == pytest.approx(0.1)


def test_comparison_rejects_different_residue_panels() -> None:
    with pytest.raises(ValueError, match="panels differ"):
        build_comparison([_report("one", 0.2), _report("two", 0.3, n_sites=99)])


def test_external_comparison_checks_panels_and_adds_baseline_deltas() -> None:
    vanilla = {
        "model_tag": "vanilla",
        "task": "phosphosite",
        "protocol": "frozen",
        "seed": 3,
        "exact_dedup_test": {"n_sites": 20, "n_positive": 2, "auprc": 0.2},
    }
    improved = {
        **vanilla,
        "model_tag": "improved",
        "exact_dedup_test": {"n_sites": 20, "n_positive": 2, "auprc": 0.3},
    }
    comparison = build_external_comparison(
        [vanilla, improved], baselines={"improved": "vanilla"}
    )
    row = next(row for row in comparison["rows"] if row["model_tag"] == "improved")
    assert row["delta_auprc"] == pytest.approx(0.1)


def _site(row_id: str, position: int, label: int, score: float) -> PTMSitePrediction:
    return PTMSitePrediction(row_id, position, label, score)


def test_paired_site_bootstrap_is_grouped_deterministic_and_zero_for_same_scores() -> (
    None
):
    records = [
        _site("protein-a", 0, 1, 0.9),
        _site("protein-a", 1, 0, 0.1),
        _site("protein-b", 0, 1, 0.7),
        _site("protein-b", 1, 0, 0.4),
        _site("protein-c", 0, 0, 0.6),
        _site("protein-c", 1, 1, 0.5),
    ]
    first = paired_site_auprc_bootstrap(records, records, n_boot=50, seed=17)
    second = paired_site_auprc_bootstrap(records, records, n_boot=50, seed=17)

    assert first == second
    assert first["n_groups"] == 3
    assert first["n_sites"] == 6
    assert first["delta_auprc"] == pytest.approx(0.0)
    assert first["delta_ci_low"] == pytest.approx(0.0)
    assert first["delta_ci_high"] == pytest.approx(0.0)


def test_paired_site_bootstrap_rejects_misaligned_panels() -> None:
    baseline = [_site("protein-a", 0, 1, 0.8)]
    candidate = [_site("protein-b", 0, 1, 0.9)]
    with pytest.raises(ValueError, match="keys differ"):
        paired_site_auprc_bootstrap(candidate, baseline, n_boot=10)


def test_paired_site_bootstrap_central_scores_match_canonical_scorer_with_ties() -> (
    None
):
    baseline = [
        _site("protein-a", 0, 1, 0.7),
        _site("protein-a", 1, 0, 0.7),
        _site("protein-b", 0, 1, 0.2),
        _site("protein-b", 1, 0, 0.2),
    ]
    candidate = [
        _site("protein-a", 0, 1, 0.8),
        _site("protein-a", 1, 0, 0.5),
        _site("protein-b", 0, 1, 0.5),
        _site("protein-b", 1, 0, 0.1),
    ]
    result = paired_site_auprc_bootstrap(candidate, baseline, n_boot=20)

    assert result["candidate_auprc"] == pytest.approx(
        score_ptm_sites(candidate)["auprc"]
    )
    assert result["baseline_auprc"] == pytest.approx(score_ptm_sites(baseline)["auprc"])


def test_paired_site_bootstrap_ranks_a_better_candidate_above_zero() -> None:
    labels = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    candidate = [
        _site(f"protein-{index // 3}", index % 3, label, 0.9 if label else 0.1)
        for index, label in enumerate(labels)
    ]
    baseline = [
        _site(f"protein-{index // 3}", index % 3, label, 0.1 if label else 0.9)
        for index, label in enumerate(labels)
    ]

    report = paired_site_auprc_bootstrap(candidate, baseline, n_boot=200, seed=7)

    assert report["candidate_auprc"] == pytest.approx(1.0)
    assert report["delta_auprc"] > 0.0
    assert report["delta_ci_low"] > 0.0
    assert report["bootstrap_fraction_delta_gt_zero"] == pytest.approx(1.0)


def test_paired_site_bootstrap_can_resample_dbptm_record_groups() -> None:
    baseline = [
        _site("dbptm:demo:1:P1_HUMAN:0", 10, 1, 0.7),
        _site("dbptm:demo:0:P1_HUMAN:1", 10, 0, 0.4),
        _site("dbptm:demo:1:P2_HUMAN:2", 10, 1, 0.6),
        _site("dbptm:demo:0:P3_HUMAN:3", 10, 0, 0.3),
    ]
    report = paired_site_auprc_bootstrap(
        baseline,
        baseline,
        n_boot=20,
        group_key=lambda row_id: row_id.split(":")[3],
        resampling_unit="dbptm_record_id",
    )

    assert report["n_groups"] == 3
    assert report["resampling_unit"] == "dbptm_record_id"


def test_paired_site_bootstrap_rejects_a_panel_with_no_positive_labels() -> None:
    records = [
        _site("protein-a", 0, 0, 0.9),
        _site("protein-a", 1, 0, 0.1),
        _site("protein-b", 0, 0, 0.7),
    ]
    with pytest.raises(ValueError, match="no bootstrap replicate"):
        paired_site_auprc_bootstrap(records, records, n_boot=10)
