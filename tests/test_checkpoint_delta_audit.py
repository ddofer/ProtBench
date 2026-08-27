from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from checkpoint_delta_audit import audit_checkpoints, parse_variant, render_markdown


def _save(path: Path, tensors: dict[str, np.ndarray]) -> None:
    from safetensors.numpy import save_file

    save_file(tensors, path)


def test_checkpoint_delta_audit_identifies_only_zeroed_tensors(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.safetensors"
    variant = tmp_path / "variant.safetensors"
    _save(
        baseline,
        {
            "encoder.weight": np.ones((2, 2), dtype=np.float32),
            "encoder.blocks.0.xsa_alpha": np.asarray([0.2], dtype=np.float32),
        },
    )
    _save(
        variant,
        {
            "encoder.weight": np.ones((2, 2), dtype=np.float32),
            "encoder.blocks.0.xsa_alpha": np.zeros(1, dtype=np.float32),
        },
    )

    payload = audit_checkpoints(baseline, [("xsa_off", variant)])
    item = payload["variants"]["xsa_off"]
    assert item["changed_tensor_count"] == 1
    assert item["changed_categories"] == {"xsa_alpha": 1}
    assert item["all_changed_tensors_zeroed"]
    assert "not separately" in render_markdown(payload)


def test_checkpoint_delta_audit_rejects_key_and_tag_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.safetensors"
    variant = tmp_path / "variant.safetensors"
    _save(baseline, {"a": np.ones(1, dtype=np.float32)})
    _save(variant, {"b": np.ones(1, dtype=np.float32)})
    with pytest.raises(ValueError, match="tensor keys differ"):
        audit_checkpoints(baseline, [("variant", variant)])
    with pytest.raises(ValueError, match="duplicate variant tag"):
        audit_checkpoints(baseline, [("same", baseline), ("same", baseline)])
    assert parse_variant("xsa=/tmp/x.safetensors") == (
        "xsa",
        Path("/tmp/x.safetensors"),
    )
