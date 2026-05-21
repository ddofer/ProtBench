"""Smoke test for finetune_residue.py — skipped by default.

Run with ``pytest -m slow`` to actually fetch AMPLIFY-120M + the SS3
dataset and run a tiny CPU training loop.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_finetune_residue_ss3_smoke(tmp_path):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable,
        "-m",
        "finetune_residue",
        "--model_name",
        "chandar-lab/AMPLIFY_120M",
        "--task",
        "ss3",
        "--mode",
        "probe",
        "--max_length",
        "32",
        "--max_train_samples",
        "8",
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "2",
        "--per_device_eval_batch_size",
        "2",
        "--output_dir",
        str(tmp_path),
    ]
    result = subprocess.run(
        cmd, cwd=str(_BENCH), env=env, capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, f"stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    jsonls = list(Path(tmp_path).glob("*.jsonl"))
    assert jsonls, "no JSONL output produced"
    assert jsonls[0].read_text().strip(), "JSONL file is empty"
