"""SCOPe hierarchy levels + MAP/eligible metrics for the retrieval evaluator."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_tasks import TASKS  # noqa: E402
from protein_benchmark_suite import (  # noqa: E402
    evaluate_retrieval,
    ranking_from_similarity,
    retrieval_metrics_from_ranking,
    truncate_label_fields,
)


def test_truncate_label_fields_sccs_levels():
    labels = ["a.5.6.1", "b.1.2.3"]
    assert truncate_label_fields(labels, None) == labels
    assert truncate_label_fields(labels, 4) == ["a.5.6.1", "b.1.2.3"]  # family
    assert truncate_label_fields(labels, 3) == ["a.5.6", "b.1.2"]  # superfamily
    assert truncate_label_fields(labels, 2) == ["a.5", "b.1"]  # fold


def test_scope_task_keys_map_to_levels():
    assert TASKS["scope40_retrieval"].label_prefix_fields is None  # legacy = family
    assert TASKS["scope40_retrieval_superfamily"].label_prefix_fields == 3
    assert TASKS["scope40_retrieval_fold"].label_prefix_fields == 2
    for key in ("scope40_retrieval", "scope40_retrieval_superfamily", "scope40_retrieval_fold"):
        assert TASKS[key].dataset == "tattabio/scope40_test"
        assert TASKS[key].problem_type == "retrieval"


def _toy():
    # 4 gallery items: q0,q1 share family "a.1.1.1"; q2 is "a.1.1.2" (same
    # superfamily a.1.1 as q0/q1); q3 is a singleton fold "b.1.1.1".
    emb = np.array(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    fam = np.array(["a.1.1.1", "a.1.1.1", "a.1.1.2", "b.1.1.1"])
    return emb, fam


def test_evaluate_retrieval_family_level_eligible_and_map():
    emb, fam = _toy()
    res = evaluate_retrieval(emb, fam, k_list=(1, 2))
    # Only q0 and q1 have a same-family partner; both find it at rank 1.
    assert res["n_queries"] == 4
    assert res["n_eligible_queries"] == 2
    assert res["Recall@1"] == pytest.approx(2 / 4)
    assert res["eligible_Recall@1"] == pytest.approx(1.0)
    assert res["MAP"] == pytest.approx(2 / 4)
    assert res["eligible_MAP"] == pytest.approx(1.0)


def test_evaluate_retrieval_superfamily_level_changes_eligibility():
    emb, fam = _toy()
    sf = np.array(truncate_label_fields(fam.tolist(), 3))
    res = evaluate_retrieval(emb, sf, k_list=(1, 2))
    # q0,q1,q2 now share superfamily a.1.1 -> 3 eligible; q3 still singleton.
    assert res["n_eligible_queries"] == 3
    # q2's nearest is q1 (cos .1) then q0 -> both relevant -> AP 1.0; q0/q1 have
    # ranks 1 and 2 relevant -> AP 1.0. MAP over all = 3/4.
    assert res["eligible_Recall@1"] == pytest.approx(1.0)
    assert res["MAP"] == pytest.approx(3 / 4)


def test_ranking_from_similarity_excludes_self_and_matches_cosine_path():
    emb, fam = _toy()
    normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    ranking = ranking_from_similarity(normed @ normed.T)
    assert ranking.shape == (4, 3)
    assert all(q not in ranking[q] for q in range(4))
    direct = retrieval_metrics_from_ranking(ranking, fam, (1, 2))
    assert direct == evaluate_retrieval(emb, fam, k_list=(1, 2))


def test_evaluate_retrieval_returns_zero_without_matches():
    emb = np.eye(3, dtype=np.float32)
    res = evaluate_retrieval(emb, np.array(["x", "y", "z"]), k_list=(1,))
    assert res["Recall@1"] == 0.0 and res["MAP"] == 0.0
    assert res["n_eligible_queries"] == 0 and "eligible_MAP" not in res
