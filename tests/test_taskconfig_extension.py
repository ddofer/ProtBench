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
    for key in ("stability", "solubility", "disprot", "ec_classification"):
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


@pytest.mark.parametrize(
    "key,problem_type,metric",
    [
        ("deepet_topt", "regression", "Spearman"),
        ("ppi_affinity", "regression", "Spearman"),
        ("tcr_pmhc_affinity", "binary", "AUC"),
    ],
)
def test_growth_temperature_and_affinity_tasks(key, problem_type, metric):
    cfg = TASKS[key]
    assert cfg.problem_type == problem_type
    assert cfg.main_metric == metric
    assert cfg.validation_split is not None, "needs a real validation split"


def test_ppi_affinity_is_the_pairwise_regression_task():
    """`ppi_bernett` is pairwise BINARY; this is the only pairwise regression, so
    it exercises the pair path (two pooled embeddings concatenated) against a
    continuous target."""
    cfg = TASKS["ppi_affinity"]
    assert set(cfg.input_map) == {"seq1", "seq2"}
    assert cfg.problem_type == "regression"
    assert TASKS["ppi_bernett"].problem_type == "binary"


def test_tcr_task_is_single_sequence_despite_packing_three_chains():
    """The three chains arrive pre-joined as "CDR3a|CDR3b|peptide" in one
    column, so this is a single-sequence task, not a pair task. Declaring it
    seq1/seq2 would silently look for columns that do not exist."""
    cfg = TASKS["tcr_pmhc_affinity"]
    assert set(cfg.input_map) == {"seq"}


def test_growth_temperature_is_distinct_from_melting_temperature():
    """Three temperature tasks with different meanings; keeping their datasets
    distinct is what stops them being read as replicates of each other."""
    sources = {k: TASKS[k].dataset for k in ("deepet_topt", "thermostability", "meltome")}
    assert len(set(sources.values())) == 3, sources
