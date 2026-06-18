"""Tests for final_benchmark_report._validate — the self-validation guard that
stops the capped/uncapped DMS-indel mixup (the #1 historical results confusion).

The report lives in results/ (gitignored), so load it by absolute path.
Hermetic: _validate(res=...) is pointed at a tmp dir, never the real results.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPORT = Path("/data/proteva/plm/results/final_benchmark_report.py")


def _load():
    spec = importlib.util.spec_from_file_location("_fbr_under_test", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk(p: Path, text="{}\n"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_fails_when_capped_sits_beside_uncapped(tmp_path):
    mod = _load()
    _mk(tmp_path / "bench" / "idx_full_vanilla" / "x.jsonl")
    _mk(tmp_path / "bench" / "pgym_van_indel" / "y.jsonl")   # capped, loose -> must FAIL
    with pytest.raises(RuntimeError, match="capped"):
        mod._validate(res=str(tmp_path))


def test_passes_when_only_uncapped_present(tmp_path):
    mod = _load()
    _mk(tmp_path / "bench" / "idx_full_vanilla" / "x.jsonl")
    warns = mod._validate(res=str(tmp_path))           # no capped dirs -> no raise
    assert isinstance(warns, list)


def test_archived_capped_does_not_trip_the_guard(tmp_path):
    mod = _load()
    _mk(tmp_path / "bench" / "idx_full_vanilla" / "x.jsonl")
    _mk(tmp_path / "bench" / "_archived_capped_indels" / "pgym_van_indel" / "y.jsonl")
    warns = mod._validate(res=str(tmp_path))           # archived copy is fine
    assert isinstance(warns, list)


def test_warns_when_no_uncapped_indels_yet(tmp_path):
    mod = _load()
    (tmp_path / "bench").mkdir(parents=True)            # nothing written yet
    warns = mod._validate(res=str(tmp_path))
    assert any("indel" in w.lower() for w in warns)
