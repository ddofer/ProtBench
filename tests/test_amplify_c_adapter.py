"""AMPLIFY-C must see the attention the bench's AMPLIFY path means to give it.

hf_amplify_c's remote code feeds ``attention_mask.bool()`` to SDPA (True =
attend) and its ``forward`` takes no ``return_dict``. The bench builds the
xformers-era AMPLIFY's additive 0/-inf mask, which ``.bool()`` inverts: real
tokens attend only to padding, and nothing raises.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from model_utils import _prepare_amplify_inputs, adapt_amplify_c


class _AmplifyCLike(torch.nn.Module):
    """hf_amplify_c's calling convention in one attention layer."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.encoder = torch.nn.Embedding(8, 4)
        self.layer_norm = torch.nn.RMSNorm(4)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.encoder(input_ids)[:, None]
        mask = None if attention_mask is None else attention_mask[:, None, None, :].bool()
        attn = torch.nn.functional.scaled_dot_product_attention(x, x, x, attn_mask=mask)
        return SimpleNamespace(hidden_states=[(x + attn)[:, 0]])


class _ProteinTokenizerLike:
    def truncate(self, encoded_inputs, max_length=None, random_truncate=True):
        return random_truncate


def test_bench_additive_mask_gives_unpadded_hidden_states():
    model, tokenizer = _AmplifyCLike(), _ProteinTokenizerLike()
    short = torch.tensor([[3, 4, 5]])
    reference = model(input_ids=short).hidden_states[-1]
    adapt_amplify_c(model, tokenizer)

    ids, additive, _, _ = _prepare_amplify_inputs(
        torch.tensor([[3, 4, 5, 0], [6, 7, 1, 2]]), torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    )
    out = model(input_ids=ids, attention_mask=additive, output_hidden_states=True, return_dict=True)

    torch.testing.assert_close(out.hidden_states[-1][:1, :3], reference)


def test_native_zero_one_mask_passes_through():
    model = _AmplifyCLike()
    ids, mask = torch.tensor([[3, 4, 5, 0]]), torch.tensor([[1, 1, 1, 0]])
    expected = model(input_ids=ids, attention_mask=mask).hidden_states[-1]
    adapt_amplify_c(model, _ProteinTokenizerLike())
    torch.testing.assert_close(model(input_ids=ids, attention_mask=mask).hidden_states[-1], expected)


def test_final_norm_alias_and_deterministic_truncation():
    model, tokenizer = _AmplifyCLike(), _ProteinTokenizerLike()
    adapt_amplify_c(model, tokenizer)
    assert model.layer_norm_2 is model.layer_norm
    assert tokenizer.truncate({}, max_length=4, random_truncate=True) is False


def test_xformers_amplify_is_left_alone(monkeypatch):
    monkeypatch.setattr(
        sys.modules[_AmplifyCLike.__module__], "memory_efficient_attention", object(), raising=False
    )
    model, tokenizer = _AmplifyCLike(), _ProteinTokenizerLike()
    adapt_amplify_c(model, tokenizer)
    assert "forward" not in vars(model)
    assert not hasattr(model, "layer_norm_2")
    assert tokenizer.truncate({}, random_truncate=True) is True
