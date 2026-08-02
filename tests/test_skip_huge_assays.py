"""Unit test for the --skip_huge_assays filter (pure helper).

Pins the 2026-06-18 bug: the inline skip branch called an undefined ``logger``,
so ``--skip_huge_assays`` crashed with NameError the moment it skipped an assay.
The logic now lives in ``_filter_huge_assays`` and is unit-tested here; the
caller prints (the file's idiom) instead of touching a logger.

Pure-function test — no GPU, no model, no dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from proteingym_mlm_zeroshot import _filter_huge_assays


def test_drops_only_over_threshold():
    g2idx = {"small": list(range(5)), "huge": list(range(100)), "edge": list(range(10))}
    kept, n_dropped = _filter_huge_assays(np.array(["small", "huge", "edge"]), g2idx, 10)
    assert sorted(kept.tolist()) == ["edge", "small"]   # "huge" (100>10) dropped; "edge" (==10) kept
    assert n_dropped == 1


def test_keeps_all_when_under_threshold():
    g2idx = {"a": [0, 1], "b": [0]}
    kept, n_dropped = _filter_huge_assays(np.array(["a", "b"]), g2idx, 10_000)
    assert n_dropped == 0
    assert sorted(kept.tolist()) == ["a", "b"]


def test_all_dropped_returns_empty():
    # everything over the cap -> empty kept, downstream `for g in assays` is a no-op
    kept, n_dropped = _filter_huge_assays(np.array(["huge"]), {"huge": list(range(100))}, 10)
    assert len(kept) == 0 and n_dropped == 1
