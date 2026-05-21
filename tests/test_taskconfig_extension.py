"""Verify the new token-classification tasks and validator extension."""

import pytest

from benchmark_tasks import TASKS, TaskConfig


def test_new_tasks_present_with_token_classification_type():
    for key in ("ss3", "disorder", "signal_peptide"):
        assert key in TASKS, f"{key} missing from TASKS"
        assert TASKS[key].problem_type == "token_classification"


def test_existing_35_tasks_still_load():
    # We expect at least the originally vendored 35 tasks plus our 3
    # additions, i.e. 38+.
    assert len(TASKS) >= 38
    # Spot-check a few well-known keys.
    for key in ("stability", "solubility", "chezod_disorder", "ec_classification"):
        assert key in TASKS


def test_validator_accepts_token_classification():
    cfg = TaskConfig(
        name="tmp",
        dataset="x/y",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="token_classification",
        main_metric="Accuracy",
    )
    assert cfg.problem_type == "token_classification"


def test_validator_rejects_bad_type():
    with pytest.raises(ValueError):
        TaskConfig(
            name="tmp",
            dataset="x/y",
            input_map={"seq": "seq"},
            label_col="label",
            problem_type="foo",
            main_metric="Accuracy",
        )
