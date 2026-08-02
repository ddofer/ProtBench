"""Strided assay sharding + shard-merge for proteingym_mlm_zeroshot must reproduce
the unsharded leaderboard metric EXACTLY. aggregate_proteingym is a pure function
of (recs, pool_*, n_skipped) and shards score DISJOINT assays, so aggregating the
union == aggregating the unsharded run. These tests pin that invariant."""
from __future__ import annotations

import numpy as np

from proteingym_mlm_zeroshot import (
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


# --- merge completeness gate: a crashed shard must NOT yield a partial result ---
import json as _json
import os as _os
import subprocess as _sp
import sys as _sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]

from _hf_finetune_common import safe_ckpt as _safe_ckpt

_CTX = {
    "checkpoint": "fakemodel", "model_type": "amplify", "model_window": 1022,
    "window_strategy": "optimal_window", "rope_extended_to": None,
    "indel_score_mode": "strided", "indel_pll_passes": 16, "notes": "t",
}


def _write_shard(out, task, shard, n, recs):
    (out / f"_shard_{task}__{shard}of{n}.json").write_text(_json.dumps(
        {"result": {"task": task, "recs": recs, "pool_ys": [], "pool_scores": [],
                    "n_skipped": 0}, "ctx": _CTX}))


def _run_merge(tmpdir, n):
    # sys.executable, not a hardcoded venv: the interpreter running the tests is
    # the one that should run the subprocess. The old "plm/.venv/bin/python" was
    # relative to a directory layout this repo no longer has -- and did not exist
    # there either, so this test never actually ran its subprocess.
    return _sp.run(
        [_sys.executable, str(ROOT / "proteingym_mlm_zeroshot.py"),
         "--model_name", "fakemodel", "--merge_only", "--assay_num_shards", str(n),
         "--tasks", "proteingym_dms_indels_zeroshot", "--output_dir", str(tmpdir)],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**_os.environ, "PYTHONPATH": str(ROOT)})


def test_merge_refuses_partial_when_a_shard_marker_is_missing(tmp_path):
    """A shard that crashes leaves no _shard_done marker; --merge_only must REFUSE
    (nonzero exit, no JSONL) rather than aggregate a partial leaderboard metric."""
    out = tmp_path / f"mlm_zs_{_safe_ckpt('fakemodel')}"
    out.mkdir(parents=True)
    task = "proteingym_dms_indels_zeroshot"
    recs = [{"primary": 0.1 * i, "auc": 0.5 + 0.01 * i} for i in range(4)]
    _write_shard(out, task, 0, 2, recs[0::2])
    _write_shard(out, task, 1, 2, recs[1::2])
    (out / "_shard_done__0of2.json").write_text("{}")  # only shard 0 finished

    r = _run_merge(tmp_path, 2)
    assert r.returncode != 0, "merge must refuse a partial result\n" + r.stdout + r.stderr
    assert not (tmp_path / f"mlm_zeroshot_{_safe_ckpt('fakemodel')}.jsonl").exists()

    (out / "_shard_done__1of2.json").write_text("{}")  # shard 1 now finished
    r = _run_merge(tmp_path, 2)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / f"mlm_zeroshot_{_safe_ckpt('fakemodel')}.jsonl").exists()
