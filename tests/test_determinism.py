"""Seeding must cover every RNG the benchmark can reach.

BENCHMARK_SEED was threaded into sklearn's random_state and datasets.shuffle,
which covers the probes and the subsampling. It does not cover torch, numpy's
global RNG, or stdlib random -- so anything reaching for those (dropout in a
fine-tune, an unseeded permutation, a shuffled dataloader) was free to vary
run to run while the run still called itself seeded.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protein_benchmark_suite as pbs  # noqa: E402


def test_seed_all_seeds_stdlib_random():
    pbs.seed_all(1234)
    got = [random.random() for _ in range(3)]
    random.seed(1234)
    assert got == [random.random() for _ in range(3)]


def test_seed_all_seeds_numpy_global_rng():
    pbs.seed_all(1234)
    got = np.random.rand(3).tolist()
    np.random.seed(1234)
    assert got == np.random.rand(3).tolist()


def test_seed_all_seeds_torch():
    pbs.seed_all(1234)
    assert torch.initial_seed() == 1234
    got = torch.randn(3)
    pbs.seed_all(1234)
    assert torch.equal(got, torch.randn(3))


def test_seed_all_is_reachable_from_a_different_seed():
    """Re-seeding must actually re-seed, not no-op after the first call."""
    pbs.seed_all(1)
    first = torch.randn(3)
    pbs.seed_all(2)
    assert not torch.equal(first, torch.randn(3))
    pbs.seed_all(1)
    assert torch.equal(first, torch.randn(3))
