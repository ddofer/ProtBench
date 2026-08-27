from __future__ import annotations

import pytest

from ptm_compare import build_comparison


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
        row
        for row in rows
        if row["model_tag"] == "trained" and row["task"] == "site"
    )
    assert trained_site["delta_auprc"] == pytest.approx(0.1)


def test_comparison_rejects_different_residue_panels() -> None:
    with pytest.raises(ValueError, match="panels differ"):
        build_comparison([_report("one", 0.2), _report("two", 0.3, n_sites=99)])
