"""Embedding cache must survive across processes without re-opening the race.

``tests/test_embed_cache_pid_scoping.py`` records why the cache path is
PID-scoped: FastESM's ``embed_dataset`` does a non-atomic ``torch.save`` after a
read-modify-write, and concurrent benchmark workers tore the file mid-write,
corrupting 716 rows. PID scoping fixed that by giving every process its own
file -- but it also meant no process ever benefited from another's work, so a
two-split sweep embedded every sequence twice and a restart re-embedded
everything from zero.

These tests pin the middle ground: seed the per-process file from the shared
cache before embedding (a plain read of a complete file), and publish it back
with an atomic rename afterwards. Neither step can produce a torn file.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protein_benchmark_suite as pbs  # noqa: E402


class _RecordingEmbedModel:
    """Stub that reports what was already in its save_path when it was called."""

    tokenizer = object()
    config = types.SimpleNamespace(hidden_size=8)

    def __init__(self) -> None:
        self.preloaded: dict[str, torch.Tensor] | None = None
        self.last_save_path: str | None = None

    def embed_dataset(self, **kwargs: object) -> dict[str, torch.Tensor]:
        """Mimic embed_dataset: load whatever is at save_path, add, save back."""

        save_path = str(kwargs["save_path"])
        self.last_save_path = save_path
        self.preloaded = (
            torch.load(save_path, weights_only=True)
            if os.path.exists(save_path)
            else None
        )
        embeddings = dict(self.preloaded or {})
        for seq in kwargs["sequences"]:  # type: ignore[union-attr]
            embeddings[str(seq)] = torch.zeros(8)
        if kwargs["save"]:
            torch.save(embeddings, save_path)
        return embeddings


def test_embedding_cache_is_seeded_from_a_previous_process(tmp_path: Path) -> None:
    """A shared cache left by an earlier process must be visible to this one.

    Without this, each split of a sweep re-embeds every sequence from scratch.
    """

    canonical = tmp_path / "ns" / "embeddings.pth"
    canonical.parent.mkdir(parents=True)
    torch.save({"AAAA": torch.ones(8)}, canonical)
    model = _RecordingEmbedModel()

    pbs.embed_sequences(
        (object(), model),
        is_sbert=False,
        sequences=["BBBB"],
        device="cpu",
        embed_save_path=str(canonical),
    )

    assert model.preloaded is not None, "process started with an empty cache"
    assert "AAAA" in model.preloaded


def test_embedding_cache_is_published_back_for_the_next_process(
    tmp_path: Path,
) -> None:
    """Work done here must land in the shared cache, and leave no pid file behind."""

    canonical = tmp_path / "ns" / "embeddings.pth"
    canonical.parent.mkdir(parents=True)
    model = _RecordingEmbedModel()

    pbs.embed_sequences(
        (object(), model),
        is_sbert=False,
        sequences=["BBBB"],
        device="cpu",
        embed_save_path=str(canonical),
    )

    assert canonical.is_file(), "shared cache was never published"
    assert "BBBB" in torch.load(canonical, weights_only=True)
    assert model.last_save_path is not None
    assert not os.path.exists(model.last_save_path), "pid-scoped file was left behind"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
