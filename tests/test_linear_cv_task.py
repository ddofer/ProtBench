"""``evaluate_task(..., probe_type="linear_cv")``: select C on validation, score on test.

Dataset loading is replaced at its boundary (``prepare_data``) so the test can see
exactly which split each stage consumed.
"""

import numpy as np
import pytest

import protein_benchmark_suite as pbs
from benchmark_tasks import TaskConfig
from probe_cv import DEFAULT_GRID
from test_embed_pooling import PositionModel, PositionTokenizer

# The stand-in embeds a sequence by its length, so "long vs short" is learnable.
TRAIN = (["M" * n for n in range(1, 41)], [int(n > 20) for n in range(1, 41)])
VALID = (["K" * n for n in range(1, 21)], [int(n > 20) for n in range(1, 21)] )
TEST = (["V" * n for n in range(5, 36)], [int(n > 20) for n in range(5, 36)])


@pytest.fixture
def splits_seen(monkeypatch):
    seen = []

    def fake_prepare_data(cfg, max_samples=None, eval_split="validation", top_k_labels_override=None):
        seen.append(eval_split)
        held_out = TEST if eval_split == "test" else VALID
        meta = {"resolved_eval_split": eval_split, "eval_strategy": f"{eval_split}_split", "cv_fallback": False}
        return TRAIN[0], TRAIN[1], held_out[0], held_out[1], None, meta

    monkeypatch.setattr(pbs, "prepare_data", fake_prepare_data)
    return seen


def test_linear_cv_selects_on_validation_scores_on_test_and_reports_the_chosen_c(splits_seen):
    cfg = TaskConfig(name="toy", dataset="toy", input_map={"seq": "seq"}, label_col="y",
                     problem_type="binary", main_metric="AP")
    metrics, split, _ = pbs.evaluate_task(
        cfg, (PositionTokenizer(), PositionModel()), False, "cpu",
        probe_type="linear_cv", eval_split="test", pooling="last",
    )
    assert sorted(splits_seen) == ["test", "validation"]
    assert split == "test"
    assert metrics["ProbeC"] in DEFAULT_GRID
    assert metrics["AP"] == pytest.approx(1.0)
