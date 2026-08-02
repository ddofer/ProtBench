"""A torch.compile checkpoint must never benchmark as a random model.

A model saved while ``torch.compile``-wrapped has EVERY state-dict key prefixed
``_orig_mod.``. ``from_pretrained`` then matches nothing and silently returns a
randomly-initialised model, which benchmarks as plausible-looking garbage
(AUC pinned at 0.500, F1 0.000) rather than crashing.

This has now bitten repeatedly. Real incident (2026-07-21): the v6-rtd re-bench
targeted ``esmc_stage2_v6_rtd/checkpoint-105000`` (652/652 keys prefixed). The
pre-bench strip raised ModuleNotFoundError, an ``except Exception`` swallowed it,
and ~3.5 GPU-hours produced chance-level numbers that read as "the model is
broken". The model was fine.

The load path already had a gate — ``_WRAPPER_PREFIXES`` — but it only listed the
JEPA-specific ``_orig_mod.student.`` / ``_orig_mod.teacher.`` forms, so a plain
``_orig_mod.``-prefixed checkpoint sailed straight through it.

These tests pin the gate on the PLAIN prefix, so the guard holds for any future
model/version regardless of which launcher (or none) ran the strip first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from model_utils import (  # noqa: E402
    _assert_no_wrapper_prefixes,
    _validate_local_checkpoint_integrity,
)


def _write_ckpt(dirpath: Path, keys: list[str]) -> Path:
    """Minimal single-file safetensors checkpoint with the given key names."""
    import torch
    from safetensors.torch import save_file

    dirpath.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: torch.zeros(2, 2) for k in keys},
        str(dirpath / "model.safetensors"),
        metadata={"format": "pt"},
    )
    return dirpath


class _CleanKeyModel:
    """Loaded model with normal (unprefixed) keys — what HF gives you after it
    silently ignores a fully-prefixed checkpoint."""

    def state_dict(self):
        return {"encoder.layers.0.weight": None, "mlm_head.bias": None}


def test_validate_names_orig_mod_as_the_cause_not_architecture_mismatch(tmp_path):
    """The exact v6-rtd failure must be diagnosed as a '_orig_mod' prefix problem.

    Without the plain prefix in ``_WRAPPER_PREFIXES`` this still raised — but via
    the generic key-overlap gate, reporting "architecture mismatch or an incorrect
    export". That sends you hunting a broken model when the checkpoint just needs
    stripping. Pin the actionable diagnosis, not merely "it raised".
    """
    ckpt = _write_ckpt(
        tmp_path / "compiled",
        ["_orig_mod.encoder.layers.0.weight", "_orig_mod.mlm_head.bias"],
    )

    # Match the PREFIX gate's wording, not the substring "_orig_mod" — pytest's
    # tmp_path embeds the test name, so "_orig_mod" appears in the error's path
    # and would match either gate (a tautology that passes without the fix).
    with pytest.raises(RuntimeError, match="wrapper-prefixed keys found"):
        _validate_local_checkpoint_integrity(str(ckpt), _CleanKeyModel())


def test_assert_no_wrapper_prefixes_rejects_plain_orig_mod(tmp_path):
    """The bypass path (SentenceTransformer loads) must reject it too."""
    ckpt = _write_ckpt(tmp_path / "compiled2", ["_orig_mod.encoder.weight"])

    with pytest.raises(RuntimeError, match="_orig_mod"):
        _assert_no_wrapper_prefixes(str(ckpt))


def test_clean_checkpoint_still_passes(tmp_path):
    """Regression: a normal (stripped) checkpoint must NOT raise on the prefix gate."""
    ckpt = _write_ckpt(
        tmp_path / "clean", ["encoder.layers.0.weight", "mlm_head.bias"]
    )

    _assert_no_wrapper_prefixes(str(ckpt))  # must not raise
