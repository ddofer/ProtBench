"""Strided assay sharding + shard-merge for proteingym_mlm_zeroshot must reproduce
the unsharded leaderboard metric EXACTLY. aggregate_proteingym is a pure function
of (recs, pool_*, n_skipped) and shards score DISJOINT assays, so aggregating the
union == aggregating the unsharded run. These tests pin that invariant."""
from __future__ import annotations

import numpy as np

from plm.bench.proteingym_mlm_zeroshot import (
    _shard_assays,
    _merge_results,
    aggregate_proteingym,
)


def test_shard_assays_strided_disjoint_cover():
    assays = np.array([f"a{i}" for i in range(23)])
    n = 4
    shards = [_shard_assays(assays, i, n) for i in range(n)]
    seen: set[str] = set()
    for s in shards:
        sl = set(s.tolist())
        assert not (sl & seen), "shards must be disjoint"
        seen |= sl
    assert seen == set(assays.tolist()), "shards must cover every assay"
    assert sum(len(s) for s in shards) == len(assays), "no assay duplicated"
    assert list(_shard_assays(assays, 0, 1)) == list(assays), "nshards<=1 is identity"


def test_merge_flat_regression_equals_unsharded():
    # dms_indels: flat mean over per-assay recs ({"primary","auc"}).
    full_recs = [{"primary": 0.1 * i, "auc": 0.5 + 0.01 * i} for i in range(12)]
    full = {"task": "proteingym_dms_indels_zeroshot", "recs": full_recs,
            "pool_ys": [], "pool_scores": [], "n_skipped": 3}
    shards = [{"task": full["task"], "recs": full_recs[k::3],
               "pool_ys": [], "pool_scores": [], "n_skipped": 1} for k in range(3)]
    merged = _merge_results(shards)
    a_full = aggregate_proteingym(full)
    a_merged = aggregate_proteingym(merged)
    assert merged["n_skipped"] == 3
    assert abs(a_full["eval_spearman"] - a_merged["eval_spearman"]) < 1e-9
    assert abs(a_full["eval_auc"] - a_merged["eval_auc"]) < 1e-9


def test_merge_pooled_clinical_indels_equals_unsharded():
    # clinical_indels: ONE pooled AUC across all genes. roc_auc is pairing-preserving
    # and order-independent, so strided split + merge == unsharded.
    rng = np.random.default_rng(0)
    ys = (rng.random(40) > 0.5).astype(float).tolist()
    scores = rng.random(40).tolist()
    full = {"task": "proteingym_clinical_indels_zeroshot", "recs": [],
            "pool_ys": ys, "pool_scores": scores, "n_skipped": 0}
    shards = [{"task": full["task"], "recs": [],
               "pool_ys": ys[k::4], "pool_scores": scores[k::4], "n_skipped": 0}
              for k in range(4)]
    a_full = aggregate_proteingym(full)
    a_merged = aggregate_proteingym(_merge_results(shards))
    assert abs(a_full["eval_auc"] - a_merged["eval_auc"]) < 1e-9
