"""Tests for proteva model-type detection in model_utils.detect_model_type."""

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from model_utils import detect_model_type


@pytest.fixture()
def neutral_ckpt_dir():
    """A temp checkpoint dir whose path contains no model-family keyword.

    ``pytest``'s ``tmp_path`` embeds the test-function name in the directory,
    which would leak the substring "proteva" into the path and trip the
    name-based detector — defeating a test that targets the *config* branch.
    """
    d = Path(tempfile.gettempdir()) / f"ckpt-{uuid.uuid4().hex[:8]}"
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_detect_proteva_by_name():
    # Substring match only -- no such repo is fetched, so a placeholder id keeps
    # the test from naming a private checkpoint.
    assert detect_model_type("example-org/proteva-120m") == "proteva"


def test_detect_proteva_by_local_config_model_type(neutral_ckpt_dir):
    cfg = {
        "architectures": ["ProtevaForPretraining"],
        "model_type": "proteva",
        "encoder_config": {"flash_attn_mode": "fa2-varlen", "hidden_size": 640},
    }
    (neutral_ckpt_dir / "config.json").write_text(json.dumps(cfg))
    assert detect_model_type(str(neutral_ckpt_dir)) == "proteva"


def test_detect_proteva_by_local_config_architecture_only(neutral_ckpt_dir):
    # Even without model_type, the ProtevaForPretraining architecture marks it.
    cfg = {"architectures": ["ProtevaForPretraining"], "hidden_size": 640}
    (neutral_ckpt_dir / "config.json").write_text(json.dumps(cfg))
    assert detect_model_type(str(neutral_ckpt_dir)) == "proteva"


def test_non_proteva_config_still_standard(neutral_ckpt_dir):
    # A plain BERT config (neutral path) must stay "standard".
    cfg = {"architectures": ["BertModel"], "model_type": "bert"}
    (neutral_ckpt_dir / "config.json").write_text(json.dumps(cfg))
    assert detect_model_type(str(neutral_ckpt_dir)) == "standard"
