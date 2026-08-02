"""Content-addressed disk cache for whole-sequence probe embeddings.

The Proteva packed sequence-embedding path did NOT persist embeddings (only the
residue path + the Synthyra embed_dataset path did), so the linear probe
re-extracted the TRAIN embeddings on every eval-split invocation (validation
then test = 2 separate processes) — the heaviest tasks (Solubility/Peptide-HLA/
Stability) paid their full train extraction twice.

``cached_embed_sequences`` wraps an extraction callable with a CONTENT-ADDRESSED
cache (key = hash of the exact seqs + an embed-config string), so an identical
seq-set is extracted once and reused. Content addressing makes a wrong-embedding
reuse impossible: any change to the seqs/config changes the key -> a miss -> a
fresh extract. Any cache error falls back to a fresh extract (perf-only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from seq_embed_cache import cached_embed_sequences  # noqa: E402


def _fn(seqs, counter):
    """A zero-arg extraction stub whose output depends on the seq CONTENT, so a
    wrong reuse would produce a visibly wrong array."""

    def go():
        counter["n"] += 1
        return np.array([[float(len(s)), float(ord(s[0]))] for s in seqs], dtype="float32")

    return go


def test_identical_seqs_extract_once_and_return_identical(tmp_path):
    counter = {"n": 0}
    root = str(tmp_path / "seq_cache")
    seqs = ["ACDEF", "GHIKL", "MNPQR"]

    x1 = cached_embed_sequences(_fn(seqs, counter), seqs, cache_root=root, cfg_key="c")
    x2 = cached_embed_sequences(_fn(seqs, counter), seqs, cache_root=root, cfg_key="c")

    assert counter["n"] == 1, "second identical call must HIT the cache"
    np.testing.assert_array_equal(x1, x2)
    assert x1.shape == (3, 2)


def test_different_seqs_miss(tmp_path):
    counter = {"n": 0}
    root = str(tmp_path / "seq_cache")
    cached_embed_sequences(_fn(["ACDEF"], counter), ["ACDEF"], cache_root=root, cfg_key="c")
    cached_embed_sequences(_fn(["WXYZ"], counter), ["WXYZ"], cache_root=root, cfg_key="c")
    assert counter["n"] == 2, "a different seq-set must MISS and re-extract"


def test_cfg_change_misses(tmp_path):
    counter = {"n": 0}
    root = str(tmp_path / "seq_cache")
    seqs = ["ACDEF"]
    cached_embed_sequences(_fn(seqs, counter), seqs, cache_root=root, cfg_key="trunk|l2=0")
    cached_embed_sequences(_fn(seqs, counter), seqs, cache_root=root, cfg_key="trunk|l2=1")
    assert counter["n"] == 2, "an embed-config change must MISS (no stale reuse)"


def test_cache_root_none_never_caches(tmp_path):
    counter = {"n": 0}
    seqs = ["ACDEF"]
    cached_embed_sequences(_fn(seqs, counter), seqs, cache_root=None, cfg_key="c")
    cached_embed_sequences(_fn(seqs, counter), seqs, cache_root=None, cfg_key="c")
    assert counter["n"] == 2, "cache_root=None must fall through every call"


def test_shape_mismatch_guard_falls_back(tmp_path):
    """If a cached array's row count != len(seqs) (corrupt/stale), ignore it and
    re-extract rather than return a mismatched matrix to the probe."""
    counter = {"n": 0}
    root = str(tmp_path / "seq_cache")
    seqs = ["ACDEF", "GHIKL"]
    # Prime the cache, then poison the stored array to a wrong row count.
    cached_embed_sequences(_fn(seqs, counter), seqs, cache_root=root, cfg_key="c")
    from seq_embed_cache import _seq_cache_key
    from token_classification_probe import EmbeddingCache
    cache = EmbeddingCache(Path(root))
    key = _seq_cache_key(seqs, "c")
    cache.put(key, np.zeros((5, 2), dtype="float32"), np.zeros(5, dtype="int64"))  # wrong N
    out = cached_embed_sequences(_fn(seqs, counter), seqs, cache_root=root, cfg_key="c")
    assert out.shape[0] == 2, "shape-mismatched cache entry must be ignored + re-extracted"
    assert counter["n"] == 2
