"""Content-addressed disk cache for whole-sequence probe embeddings.

The linear-probe extraction (``embed_sequences``) is the single-process long pole
of the bench. Its Proteva packed path did not persist embeddings, so the TRAIN
embeddings were re-extracted on every eval-split invocation (validation then test
= 2 separate processes). This wraps an extraction callable with a
content-addressed :class:`EmbeddingCache` so an identical seq-set is extracted
once and reused across those invocations.

Safety: the key is a SHA-256 of the exact seqs + an embed-config string, so any
change to the seqs or the embedding config produces a different key -> a miss ->
a fresh extract. A wrong-embedding reuse is therefore impossible. On any cache
error (or a shape-mismatched entry) we fall back to a fresh extract — the cache
is a pure performance optimisation and never changes results or hard-fails.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


def _seq_cache_key(seqs: Sequence, cfg_key: str) -> str:
    h = hashlib.sha256()
    h.update(cfg_key.encode("utf-8"))
    h.update(f"|n={len(seqs)}|".encode("utf-8"))
    for s in seqs:
        h.update(str(s).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:40]


def cached_embed_sequences(
    embed_fn: Callable[[], np.ndarray],
    seqs: Sequence,
    *,
    cache_root: Optional[str],
    cfg_key: str,
) -> np.ndarray:
    """Return embeddings for ``seqs``, reusing a content-addressed disk cache.

    Args:
        embed_fn: zero-arg callable that extracts + returns the ``[N, H]`` array
            (the caller binds the real ``embed_sequences(...)`` in a lambda).
        seqs: the exact sequences being embedded (hashed into the cache key).
        cache_root: directory for the cache; ``None`` disables caching entirely.
        cfg_key: a string capturing every embed-config knob that changes the
            output (probe_embed_mode, l2-normalize, max_length, dtype, ...).
    """
    if not cache_root:
        return embed_fn()

    cache = None
    key = None
    try:
        from token_classification_probe import EmbeddingCache

        cache = EmbeddingCache(Path(cache_root))
        key = _seq_cache_key(seqs, cfg_key)
        if cache.has(key):
            X, _, _ = cache.get(key)
            if X.shape[0] == len(seqs):
                return X
            # Row count disagrees with this seq-set — ignore the stale/corrupt
            # entry and re-extract rather than hand the probe a wrong matrix.
    except Exception:
        cache = None
        key = None

    X = embed_fn()

    if cache is not None and key is not None:
        try:
            cache.put(
                key,
                np.asarray(X, dtype="float32"),
                np.zeros(len(X), dtype="int64"),
            )
        except Exception:
            pass  # cache write is best-effort; never fail the bench on it
    return X
