"""Which models `resolve_mlm_head` accepts.

The generic fallback must demand positive evidence of a MASKED LM. A causal LM
also owns an `lm_head` and returns `.logits`, but its logits at position t are
the distribution for token t+1 -- accepting one yields a silently
one-residue-shifted contact map and mis-positioned variant-effect scores, with
no error anywhere.

Offline: nn.Module stubs, no weights.
"""

import pytest
import torch
from torch import nn

from wt_test_time_training import resolve_mlm_head


class _Config:
    def __init__(self, model_type, architectures=None):
        self.model_type = model_type
        self.architectures = architectures


class _Tokenizer:
    mask_token_id = 32
    all_special_ids = [0, 1, 2, 32]


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Base(nn.Module):
    """Minimal HF-shaped encoder: `.encoder.layer` blocks plus a head."""

    def __init__(self, model_type, architectures=None):
        super().__init__()
        self.config = _Config(model_type, architectures)
        self.esm = nn.Module()
        self.esm.encoder = nn.Module()
        self.esm.encoder.layer = nn.ModuleList([nn.Linear(4, 4)])

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch, tokens = input_ids.shape
        return _Out(torch.zeros(batch, tokens, 33))


class EsmForMaskedLM(_Base):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.lm_head = nn.Linear(4, 33)


class LlamaForCausalLM(_Base):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.lm_head = nn.Linear(4, 33)


class WeirdlyNamedWrapper(_Base):
    """A MaskedLM whose class name does not follow the convention; it must be
    recognised through `config.architectures` instead."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lm_head = nn.Linear(4, 33)


class SequenceClassifier(_Base):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.classifier = nn.Linear(4, 2)


class EncoderOnly(_Base):
    pass


def test_masked_lm_resolves_and_returns_full_length_logits():
    model = EsmForMaskedLM(model_type="esm")
    refs = resolve_mlm_head(model, _Tokenizer())
    logits = refs.forward_logits(torch.zeros(1, 5, dtype=torch.long), None)
    assert logits.shape == (1, 5, 33)
    assert refs.mask_token_id == 32


def test_masked_lm_recognised_via_config_architectures():
    """Some checkpoints wrap the model in a differently-named class."""
    model = WeirdlyNamedWrapper(model_type="esm", architectures=["EsmForMaskedLM"])
    assert resolve_mlm_head(model, _Tokenizer()) is not None


def test_weirdly_named_masked_lm_without_architectures_is_rejected():
    """No positive evidence, so refuse rather than guess."""
    with pytest.raises(ValueError):
        resolve_mlm_head(WeirdlyNamedWrapper(model_type="esm"), _Tokenizer())


def test_causal_lm_is_rejected():
    with pytest.raises(ValueError, match="masked"):
        resolve_mlm_head(LlamaForCausalLM(model_type="llama"), _Tokenizer())


def test_sequence_classifier_is_rejected():
    with pytest.raises(ValueError):
        resolve_mlm_head(SequenceClassifier(model_type="esm"), _Tokenizer())


def test_encoder_only_export_is_rejected():
    with pytest.raises(ValueError):
        resolve_mlm_head(EncoderOnly(model_type="esm"), _Tokenizer())


def test_missing_mask_token_is_rejected():
    class NoMask:
        mask_token_id = None
        all_special_ids = []

    with pytest.raises(ValueError, match="mask_token_id"):
        resolve_mlm_head(EsmForMaskedLM(model_type="esm"), NoMask())


@pytest.mark.parametrize("model_type", ["AMPLIFY", "amplify"])
def test_amplify_is_matched_case_insensitively(model_type):
    """AMPLIFY needs an ADDITIVE attention mask and length padded to a multiple
    of 8. Falling through to the generic branch hands it a 0/1 mask instead --
    the exact two things its own branch exists to fix. Every other AMPLIFY
    detector in the repo already matches case-insensitively."""

    class Amplify(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Config(model_type)
            self.transformer_encoder = nn.ModuleList([nn.Linear(4, 4)])
            self.decoder = nn.Linear(4, 33)

    refs = resolve_mlm_head(Amplify(), _Tokenizer())
    assert refs.head_module is not None
    assert len(refs.encoder_blocks) == 1
