"""Verify the new token-classification tasks and validator extension."""

import pytest

from benchmark_tasks import TASKS, TaskConfig


def test_new_tasks_present_with_token_classification_type():
    for key in ("ss3", "disorder", "signal_peptide", "conservation_flip"):
        assert key in TASKS, f"{key} missing from TASKS"
        assert TASKS[key].problem_type == "token_classification"


def test_conservation_flip_config():
    cfg = TASKS["conservation_flip"]
    assert cfg.label_col == "conservation_labels"
    # Ordinal grades 1-9 -> Spearman headline (was nominal F1_Macro; audit fix).
    assert cfg.main_metric == "Spearman"
    assert cfg.dataset.startswith("data/conservation_flip")


def test_meltome_config():
    cfg = TASKS["meltome"]
    assert cfg.problem_type == "regression"
    assert cfg.main_metric == "MSE"
    assert cfg.split_column == "split"


def test_flip2_configs():
    for key in ("flip2_amylase", "flip2_rhomax"):
        cfg = TASKS[key]
        assert cfg.problem_type == "regression"
        assert cfg.main_metric == "Spearman"
        assert cfg.label_col == "score"


def test_existing_35_tasks_still_load():
    # 4 new tasks added: conservation_flip, meltome, flip2_amylase, flip2_rhomax
    assert len(TASKS) >= 42
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
