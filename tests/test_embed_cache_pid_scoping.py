"""Regression test for the embedding-cache race that corrupted 716 benchmark rows.

The FastESM ``embed_dataset`` performs a non-atomic ``torch.save`` after a
read-modify-write on ``embeddings.pth``. When multiple benchmark subprocesses
targeting the same model path ran concurrently (3 seeds x 2 splits), the file
could be torn mid-write and subsequent ``torch.load`` raised
``PytorchStreamReader failed reading zip archive``.

The fix in ``protein_benchmark_suite.embed_sequences`` appends the PID to the
configured ``embed_save_path`` so concurrent subprocesses never share a cache
file. This test fakes a stub ``embed_dataset`` and asserts the computed
``save_path`` is process-unique while the originally configured
``embed_save_path`` is left untouched.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protein_benchmark_suite as pbs  # noqa: E402


class _StubEmbedModel:
    """Stub standing in for a FastESM model exposing ``embed_dataset``."""

    tokenizer = object()
    config = types.SimpleNamespace(hidden_size=8)

    def __init__(self) -> None:
        self.last_save_path: str | None = None

    def embed_dataset(self, **kwargs: object) -> dict[str, torch.Tensor]:
        self.last_save_path = kwargs["save_path"]  # type: ignore[assignment]
        seqs = kwargs["sequences"]  # type: ignore[assignment]
        return {s: torch.zeros(8) for s in seqs}  # type: ignore[misc]


def test_embed_cache_path_is_pid_scoped(tmp_path: Path) -> None:
    configured = str(tmp_path / "ns" / "embeddings.pth")
    model = _StubEmbedModel()

    embs = pbs.embed_sequences(
        (object(), model),
        is_sbert=False,
        sequences=["AAAA", "BBBB"],
        device="cpu",
        embed_save_path=configured,
    )

    assert isinstance(embs, np.ndarray)
    assert model.last_save_path is not None
    actual = model.last_save_path
    pid_suffix = f".pid{os.getpid()}.pth"
    assert actual.endswith(pid_suffix), (
        f"cache path must be PID-scoped to avoid cross-process corruption; got {actual}"
    )
    assert actual != configured, "unpatched shared path would re-enable the race"
    assert not os.path.exists(configured), (
        "stub must not touch the shared configured path"
    )


def test_embed_cache_disabled_uses_tempdir(tmp_path: Path) -> None:
    model = _StubEmbedModel()

    pbs.embed_sequences(
        (object(), model),
        is_sbert=False,
        sequences=["AAAA"],
        device="cpu",
        embed_save_path=None,
    )

    assert model.last_save_path is not None
    assert "_no_cache_embeddings.pth" in model.last_save_path
    assert f".pid{os.getpid()}" not in model.last_save_path


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
