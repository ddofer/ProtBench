from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ptm_benchmark import (
    PTMSitePrediction,
    PTMTypePrediction,
    assert_same_metrics,
    read_ptm_predictions,
    score_ptm_predictions,
    score_ptm_sites,
    score_ptm_types,
    structured_ptm_input,
    write_ptm_predictions,
)


def test_structured_input_decodes_little_endian_tracks() -> None:
    record = structured_ptm_input(
        {
            "row_id": "P1",
            "sequence": " ASK ",
            "ptm_any_u8": bytes([0, 1, 0]),
            "ptm_token_id": np.asarray([0, 27, 0], dtype="<u2").tobytes(),
            "ptm_class_mask": np.asarray([0, 4, 0], dtype="<u2").tobytes(),
        }
    )
    assert record.sequence == "ASK"
    assert record.ptm_any == (0, 1, 0)
    assert record.ptm_token_ids == (0, 27, 0)
    assert record.ptm_class_masks == (0, 4, 0)


def test_structured_input_rejects_misaligned_tracks() -> None:
    with pytest.raises(ValueError, match="ptm_any length"):
        structured_ptm_input(
            {"row_id": "P1", "sequence": "ASK", "ptm_any_u8": bytes([0, 1])}
        )


def test_site_metrics_cover_auprc_auroc_mcc_and_f1() -> None:
    records = [
        PTMSitePrediction("P1", 0, 0, 0.1),
        PTMSitePrediction("P1", 1, 1, 0.9),
        PTMSitePrediction("P2", 0, 0, 0.2),
        PTMSitePrediction("P2", 1, 1, 0.8),
    ]
    metrics = score_ptm_sites(records)
    assert metrics["n_sites"] == 4
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_type_metrics_report_supported_denominator_and_unsupported_types() -> None:
    records = [
        PTMTypePrediction("P1", 1, "phospho_S", ("phospho_S", "acetyl_K")),
        PTMTypePrediction("P2", 2, "acetyl_K", ("phospho_S", "methyl_K", "acetyl_K")),
        PTMTypePrediction("P3", 3, "sulfotyrosine", None, "not_in_model_vocab"),
    ]
    metrics = score_ptm_types(records)
    assert metrics["n_total"] == 3
    assert metrics["n_supported"] == 2
    assert metrics["mapping_coverage"] == pytest.approx(2 / 3)
    assert metrics["top1"] == pytest.approx(0.5)
    assert metrics["top3"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3) / 2)
    assert metrics["unsupported_types"] == {"sulfotyrosine": 1}


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_prediction_round_trip_reproduces_metrics(tmp_path: Path, suffix: str) -> None:
    sites = [PTMSitePrediction("P1", 1, 1, 0.75), PTMSitePrediction("P1", 0, 0, 0.1)]
    types = [PTMTypePrediction("P1", 1, "phospho_S", ("phospho_S", "acetyl_K"))]
    before = score_ptm_predictions(sites, types)
    path = tmp_path / f"predictions{suffix}"
    write_ptm_predictions(
        path,
        site_predictions=sites,
        type_predictions=types,
        metadata={"model": "fixture"},
    )
    metadata, loaded_sites, loaded_types = read_ptm_predictions(path)
    after = score_ptm_predictions(loaded_sites, loaded_types)
    assert metadata == {"model": "fixture"}
    assert_same_metrics(before, after)
