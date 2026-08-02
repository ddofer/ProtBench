"""Embedding caching must be opt-in.

Cached embeddings are large, and the cache key for a hub model id is just the
model name -- it does not invalidate when the upstream weights change. Silently
reusing them is the wrong default for a benchmark: a stale cache reports a
number for a model you are no longer running. Sweeps that genuinely want reuse
already pass the flag explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protein_benchmark_suite as pbs  # noqa: E402


def _args(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["protein_benchmark_suite.py", *argv])
    return pbs.parse_args()


def test_cache_embeddings_is_off_by_default(monkeypatch):
    assert _args(monkeypatch, "-m", "dummy").cache_embeddings is False


def test_cache_embeddings_can_still_be_turned_on(monkeypatch):
    assert _args(monkeypatch, "-m", "dummy", "--cache_embeddings").cache_embeddings
