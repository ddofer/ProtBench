"""run_bench: probe choice per task + skip-what-is-already-done."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH / "scripts"))

from run_bench import choose_probe, pending_tasks, summarize  # noqa: E402


def test_choose_probe_picks_torch_linear_where_sklearn_is_slow():
    # multilabel (OvR = one sklearn fit per label) and many-class tasks
    assert choose_probe("ec_classification") == "torch_linear"
    assert choose_probe("go_mf") == "torch_linear"
    assert choose_probe("remote_homology") == "torch_linear"
    assert choose_probe("cath_eat") == "torch_linear"
    # everything else: sklearn linear is faster at these sizes
    for task in ("solubility", "stability", "ss3", "subcellular_loc"):
        assert choose_probe(task) == "linear"


def test_pending_tasks_skips_rows_already_in_the_csv(tmp_path):
    csv = tmp_path / "bench_m.csv"
    pd.DataFrame(
        [
            {"Task": "Solubility (DeepSol)", "Probe": "linear", "EvalSplit": "test"},
            {"Task": "EC Classification", "Probe": "linear", "EvalSplit": "test"},
            {"Task": "Stability (Biomap)", "Probe": "linear", "EvalSplit": "validation"},
        ]
    ).to_csv(csv, index=False)
    tasks = ["solubility", "ec_classification", "stability"]
    # solubility done; EC done only with the wrong probe; stability done on the wrong split
    assert pending_tasks(tasks, csv, "test") == {
        "torch_linear": ["ec_classification"],
        "linear": ["stability"],
    }
    assert pending_tasks(tasks, tmp_path / "missing.csv", "test") == {
        "linear": ["solubility", "stability"],
        "torch_linear": ["ec_classification"],
    }


def test_summarize_writes_readable_markdown(tmp_path):
    csv = tmp_path / "bench_m.csv"
    pd.DataFrame(
        [
            {"Model": "m", "Task": "Solubility (DeepSol)", "Probe": "linear", "EvalSplit": "test",
             "EvalStrategy": "test_split", "Samples": 100, "AUC": 0.71, "Accuracy": 0.65, "ProbeFitSec": 1.2},
            {"Model": "m", "Task": "Stability (Biomap)", "Probe": "linear", "EvalSplit": "test",
             "EvalStrategy": "test_split", "Samples": 100, "Spearman": 0.69, "ProbeFitSec": 0.3},
        ]
    ).to_csv(csv, index=False)
    out = summarize(csv, tmp_path / "SUMMARY.md")
    text = out.read_text()
    assert "| solubility |" in text and "AUC" in text and "0.71" in text
    assert "| stability |" in text and "Spearman" in text and "0.69" in text
    assert "1.2" in text  # fit seconds carried through
