"""LoRA/finetune eval must keep only the logits tensor.

Some encoders return more than logits from the classification head. ESM-C
(``Synthyra/ESMplusplus_small``) returns ``(logits, hidden_states, ...)``; Proteva
returns a bare logits tensor — which is why LoRA ran fine for v5/trained-5ep and
died only for vanilla.

HF's Trainer accumulates EVERY returned tensor across the eval set, so the raw
tuple reaches ``compute_metrics`` and:

  np.array(preds_logits, dtype=np.float64)
  ValueError: setting an array element with a sequence. The requested array has an
  inhomogeneous shape after 2 dimensions. The detected shape was (2, 736) + ...

(the leading "2" is the TUPLE length, not a batch dim). Accumulating every hidden
state also pins the whole eval set's activations in memory — the OOM seen on the
first vanilla LoRA attempt.

``preprocess_logits_for_metrics`` is the supported HF hook to reduce outputs
*before* accumulation, fixing both the ragged array and the memory blow-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from _hf_finetune_common import keep_logits_only  # noqa: E402


def test_keeps_only_logits_from_multi_tensor_output():
    """(logits, hidden_states) -> logits. The ESM-C case that broke vanilla LoRA."""
    logits = torch.zeros(4, 3)
    hidden = torch.zeros(4, 17, 8)

    assert keep_logits_only((logits, hidden), labels=None) is logits


def test_passes_through_a_bare_logits_tensor():
    """Proteva's single-tensor output must be untouched."""
    logits = torch.zeros(4, 3)

    assert keep_logits_only(logits, labels=None) is logits


def test_reduced_output_is_no_longer_ragged_for_numpy():
    """The end result: metrics can build an array. Raw tuple raises; reduced does not."""
    logits = torch.zeros(4, 3)
    hidden = torch.zeros(4, 17, 8)

    with pytest.raises(ValueError):  # the actual production failure
        np.array((logits.numpy(), hidden.numpy()), dtype=np.float64)

    reduced = keep_logits_only((logits, hidden), labels=None)
    assert np.array(reduced.numpy(), dtype=np.float64).shape == (4, 3)
