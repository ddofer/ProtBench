"""Precision/attention dispatch for the frozen-probe forward.

The linear probe measures frozen-representation quality; for a fair cross-model
comparison every model must run at the SAME precision. Near-degenerate-sequence
tasks (GB1 point mutants, deep mutational scans, ProteinGym DMS substitutions,
MLM masked-marginal) carry signal BELOW the bf16 noise floor, so the probe
defaults to fp32. FlashAttention-2 (``fa2-varlen``) is bf16/fp16-only, so an
fp32 forward MUST fall back to the dense SDPA path (``flash_attn_mode="off"``),
which is bit-identical to AMPLIFY in fp32 (verified). bf16 keeps the fast flash
kernel.

This pins that policy so the Proteva loader stops force-casting bf16 and instead
honors the requested ``torch_dtype`` like every other model family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import torch  # noqa: E402

from protein_benchmark_suite import resolve_proteva_runtime  # noqa: E402


def test_bf16_keeps_flash_varlen_kernel():
    """Explicit bf16 → bf16 weights + the fast fa2-varlen flash kernel."""
    weight_dtype, flash_mode = resolve_proteva_runtime(torch.bfloat16, is_cpu=False)
    assert weight_dtype == torch.bfloat16
    assert flash_mode == "fa2-varlen"


def test_fp32_default_uses_dense_sdpa_not_flash():
    """fp32 (torch_dtype None, the CLI default) → fp32 weights + SDPA ('off').

    fa2-varlen is bf16-only; fp32 cannot use it, so the dense path is required.
    This is the fix: stop force-casting Proteva to bf16 on GPU."""
    weight_dtype, flash_mode = resolve_proteva_runtime(None, is_cpu=False)
    assert weight_dtype == torch.float32
    assert flash_mode == "off"


def test_explicit_fp32_dtype_also_dense():
    weight_dtype, flash_mode = resolve_proteva_runtime(torch.float32, is_cpu=False)
    assert weight_dtype == torch.float32
    assert flash_mode == "off"


def test_cpu_is_always_fp32_dense():
    """CPU has no flash_attn + bf16 matmul is unsupported/slow → fp32 + 'off'
    regardless of the requested dtype (preserves the existing CPU contract)."""
    for dt in (None, torch.bfloat16, torch.float32):
        weight_dtype, flash_mode = resolve_proteva_runtime(dt, is_cpu=True)
        assert weight_dtype == torch.float32
        assert flash_mode == "off"
