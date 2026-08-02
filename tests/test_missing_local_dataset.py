"""A task whose local dataset was never built must say how to build it.

Four tasks read from `data/<name>/`, produced by `scripts/prep_*.py` rather than
downloaded. On a fresh clone those directories do not exist, and two of the four
are in the default presets -- so this is the first thing a new user hits, not an
edge case.

Left to `datasets`, the failure reads "doesn't exist on the Hub or cannot be
accessed", which sends you looking for a deleted HuggingFace dataset instead of
at a prep script sitting in the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protein_benchmark_suite as pbs  # noqa: E402
from benchmark_tasks import TASKS  # noqa: E402

LOCAL_TASKS = [k for k, c in TASKS.items() if str(c.dataset).startswith("data/")]


def test_repo_still_has_local_dataset_tasks():
    """If this fails the tasks were reworked and this whole file can go."""
    assert LOCAL_TASKS, "expected at least one data/-backed task"


@pytest.mark.parametrize("task_key", LOCAL_TASKS)
def test_missing_local_dataset_names_the_prep_script(task_key, tmp_path, monkeypatch):
    cfg = TASKS[task_key]
    # Point resolution somewhere empty so the dataset is definitively absent.
    monkeypatch.setattr(pbs, "_resolve_local_dataset_path", lambda name: None)

    with pytest.raises(FileNotFoundError) as excinfo:
        pbs.require_local_dataset(cfg.dataset)

    message = str(excinfo.value)
    assert cfg.dataset in message
    assert "scripts/" in message, "must point at the prep script"
    assert "prep_" in message
