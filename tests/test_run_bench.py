"""run_bench: probe choice per task + skip-what-is-already-done."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

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
    # residue tasks too: full-data ss3 is ~2.8M rows, 615s single-core in lbfgs
    for task in ("ss3", "conservation_flip", "disprot"):
        assert choose_probe(task) == "torch_linear"
    # everything else: sklearn linear is faster at these sizes
    for task in ("solubility", "stability", "subcellular_loc"):
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


def test_failed_tasks_are_retried_not_treated_as_done(tmp_path):
    """The suite writes a row for a FAILED task too (EvalStrategy='task_exception').
    Treating that as done makes the sweep unresumable: the failure is permanent
    until --force."""
    csv = tmp_path / "bench_m.csv"
    pd.DataFrame(
        [
            {"Task": "Solubility (DeepSol)", "Probe": "linear", "EvalSplit": "test",
             "EvalStrategy": "test_split", "Samples": "Full"},
            {"Task": "Stability (Biomap)", "Probe": "linear", "EvalSplit": "test",
             "EvalStrategy": "task_exception", "Samples": "Full"},
        ]
    ).to_csv(csv, index=False)
    assert pending_tasks(["solubility", "stability"], csv, "test") == {"linear": ["stability"]}


def test_capped_scout_rows_do_not_satisfy_a_full_run(tmp_path):
    """A --very-fast row is capped (Samples=100000); it must not make the full
    sweep look complete, or the CSV silently mixes subsampled and full results."""
    csv = tmp_path / "bench_m.csv"
    pd.DataFrame(
        [{"Task": "Solubility (DeepSol)", "Probe": "linear", "EvalSplit": "test",
          "EvalStrategy": "test_split", "Samples": 100000}]
    ).to_csv(csv, index=False)
    assert pending_tasks(["solubility"], csv, "test") == {"linear": ["solubility"]}
    assert pending_tasks(["solubility"], csv, "test", max_samples=100000) == {}


def test_auto_probe_is_resolved_by_the_suite_not_only_the_wrapper():
    """`python protein_benchmark_suite.py -p auto` must get the same fast routing
    as the wrapper, or the documented CLI keeps taking the slow path."""
    from benchmark_tasks import TASKS
    from protein_benchmark_suite import PROBE_LABELS, effective_probe_type

    assert "auto" in PROBE_LABELS
    for task in ("remote_homology", "cath_eat", "ec_classification", "ss3", "conservation_flip"):
        assert effective_probe_type(TASKS[task], "auto") == "torch_linear", task
    for task in ("solubility", "stability", "subcellular_loc"):
        assert effective_probe_type(TASKS[task], "auto") == "linear", task


def test_auto_never_leaks_into_a_result_row():
    """'auto' is a request, not an identity: the CSV must record what actually ran."""
    from benchmark_tasks import TASKS
    from protein_benchmark_suite import effective_probe_type

    for task in TASKS:
        assert effective_probe_type(TASKS[task], "auto") != "auto", task


def test_choose_probe_delegates_to_the_suite():
    """The wrapper must not carry a second copy of the routing rule."""
    from benchmark_tasks import TASKS
    from protein_benchmark_suite import effective_probe_type

    for task in ("remote_homology", "ec_classification", "solubility", "ss3"):
        assert choose_probe(task) == effective_probe_type(TASKS[task], "auto")


def test_presets_match_the_suite_exactly():
    """Re-deriving preset task lists in the wrapper drifts from the suite: --fast
    there is FAST_TASKS + RETRIEVAL_TASKS, and --very-fast carries a sample cap
    that only applies when the suite resolves the preset itself."""
    from benchmark_tasks import DEFAULT_TASKS, FAST_TASKS, RETRIEVAL_TASKS, VERY_FAST_TASKS
    from run_bench import PRESETS, preset_flag

    assert PRESETS["fast"] == list(FAST_TASKS) + list(RETRIEVAL_TASKS)
    assert PRESETS["very-fast"] == list(VERY_FAST_TASKS)
    assert PRESETS["no-fast"] == list(DEFAULT_TASKS)
    assert preset_flag("fast") == "--fast"
    assert preset_flag("very-fast") == "--very-fast"
    assert preset_flag("no-fast") == "--no-fast"


def test_only_very_fast_subsamples():
    """Subsampling must be intentional: a capped run is a scout run, and mixing
    capped with full rows invalidates cross-model comparison."""
    from benchmark_tasks import FAST_MAX_SAMPLES
    from run_bench import preset_max_samples

    assert preset_max_samples("very-fast") == FAST_MAX_SAMPLES
    assert preset_max_samples("fast") is None
    assert preset_max_samples("no-fast") is None


def test_summary_headline_metric_matches_the_rest_of_the_repo(tmp_path):
    """A fourth hand-rolled metric-priority list means SUMMARY.md, bench_results_all.csv
    and COMPARISON.md can each headline a different metric for the same row."""
    from benchmark_utils import get_best_metric_for_task

    csv = tmp_path / "bench_m.csv"
    rows = [
        {"Model": "m", "Task": "Solubility (DeepSol)", "Probe": "linear", "EvalSplit": "test",
         "EvalStrategy": "test_split", "Samples": "Full", "AUC": 0.71, "Accuracy": 0.65,
         "MCC": 0.3, "ProbeFitSec": 1.2},
        {"Model": "m", "Task": "Metal Ion Binding", "Probe": "linear", "EvalSplit": "test",
         "EvalStrategy": "test_split", "Samples": "Full", "Accuracy": 0.68, "MCC": 0.36,
         "ProbeFitSec": 0.1},
    ]
    pd.DataFrame(rows).to_csv(csv, index=False)
    text = summarize(csv, tmp_path / "SUMMARY.md").read_text()
    for row in rows:
        expected, _ = get_best_metric_for_task(pd.Series(row))
        assert f"| {expected} |" in text, f"{row['Task']} should headline {expected}"


def test_summary_escapes_pipes_in_cell_values(tmp_path):
    csv = tmp_path / "bench_m.csv"
    pd.DataFrame([
        {"Model": "m", "Task": "Solubility (DeepSol)", "Probe": "linear|weird", "EvalSplit": "test",
         "EvalStrategy": "test_split", "Samples": "Full", "AUC": 0.71, "ProbeFitSec": 1.0}
    ]).to_csv(csv, index=False)
    text = summarize(csv, tmp_path / "SUMMARY.md").read_text()
    body = [ln for ln in text.splitlines() if ln.startswith("| solubility")][0]
    assert "linear\\|weird" in body, "a raw | would break the table"
    # escaped, so the row still has exactly as many real column separators as the header
    unescaped = lambda ln: len(re.findall(r"(?<!\\)\|", ln))
    assert unescaped(body) == unescaped(text.splitlines()[2])


def test_auto_is_resolved_before_any_probe_is_built():
    """'auto' is a request, not a probe: it must never reach make_probe_model."""
    import inspect

    from benchmark_tasks import TASKS
    from protein_benchmark_suite import evaluate_task, make_probe_model

    with pytest.raises(ValueError, match="Unsupported probe"):
        make_probe_model("auto", "multiclass")  # the raw registry rejects it...
    # ...so evaluate_task must resolve it first, for every caller, not just main()
    src = inspect.getsource(evaluate_task)
    assert "effective_probe_type(cfg, probe_type)" in src
    assert TASKS  # registry loaded
