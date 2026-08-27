from __future__ import annotations

import csv
from pathlib import Path

import pytest

from panel_compare import compare_panel, write_comparison

FIELDS = [
    "Model",
    "Task",
    "Samples",
    "Probe",
    "EvalMode",
    "EvalSplit",
    "EvalStrategy",
    "EmbeddingNorm",
    "CodeVersion",
    "AUC",
    "Spearman",
    "MSE",
]


def _write(path: Path, model: str, solubility: float, stability: float) -> None:
    rows = [
        {
            "Model": model,
            "Task": "Solubility (DeepSol)",
            "Samples": "Full",
            "Probe": "linear",
            "EvalMode": "standard",
            "EvalSplit": "test",
            "EvalStrategy": "test_split",
            "EmbeddingNorm": "none",
            "CodeVersion": "abc",
            "AUC": solubility,
        },
        {
            "Model": model,
            "Task": "Stability (Biomap)",
            "Samples": "Full",
            "Probe": "linear",
            "EvalMode": "standard",
            "EvalSplit": "test",
            "EvalStrategy": "test_split",
            "EmbeddingNorm": "none",
            "CodeVersion": "abc",
            "Spearman": stability,
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_panel_comparison_uses_task_headlines_and_is_deterministic(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    baseline = tmp_path / "baseline.csv"
    _write(candidate, "candidate", solubility=0.8, stability=0.4)
    _write(baseline, "baseline", solubility=0.7, stability=0.5)

    table, summary = compare_panel(candidate, baseline, n_boot=500, seed=7)
    _, repeated = compare_panel(candidate, baseline, n_boot=500, seed=7)
    assert summary == repeated
    assert len(table) == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["median_signed_delta"] == pytest.approx(0.0)
    assert set(table["metric"]) == {"AUC", "Spearman"}

    out = tmp_path / "report"
    write_comparison(table, summary, out)
    assert (out / "per_task.csv").exists()
    assert "task-bootstrap" in (out / "REPORT.md").read_text()


def test_panel_comparison_rejects_protocol_mismatch(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    baseline = tmp_path / "baseline.csv"
    _write(candidate, "candidate", solubility=0.8, stability=0.4)
    _write(baseline, "baseline", solubility=0.7, stability=0.5)
    text = baseline.read_text().replace("Full,linear", "100,linear", 1)
    baseline.write_text(text)
    with pytest.raises(ValueError, match="panel protocol mismatch"):
        compare_panel(candidate, baseline, n_boot=10)


def test_panel_comparison_can_match_complete_candidate_to_baseline_superset(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.csv"
    baseline = tmp_path / "baseline.csv"
    _write(candidate, "candidate", solubility=0.8, stability=0.4)
    _write(baseline, "baseline", solubility=0.7, stability=0.5)
    lines = candidate.read_text().splitlines()
    candidate.write_text("\n".join(lines[:2]) + "\n")

    table, summary = compare_panel(
        candidate,
        baseline,
        n_boot=10,
        allow_baseline_superset=True,
    )

    assert len(table) == 1
    assert summary["panel_scope"] == {
        "candidate_rows": 1,
        "baseline_rows": 2,
        "baseline_superset_allowed": True,
        "interpretation": (
            "complete candidate panel matched against the same baseline protocols; "
            "not a claim about tasks absent from the candidate"
        ),
    }
