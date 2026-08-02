"""Out-of-vocabulary residues must not raise.

FastPLM's tokenizer raises KeyError on any character absent from the vocabulary.
The suite catches that per task, writes it to an Error column and still exits 0,
so the failure is silent -- CATH lookup69k domain 9pcyA00 (a NUL byte at residue
99) errored the entire cath_eat task that way.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_utils import _FallbackVocab, patch_unknown_residue_tokens  # noqa: E402


class _StrictTokenizer:
    """Stand-in for FastPLM's tokenizer: raises on an unknown token."""

    def __init__(self):
        self._token_to_id = {"<unk>": 3, "A": 5, "L": 4, "X": 24}
        self.unk_token_id = 3

    def convert(self, token):
        try:
            return self._token_to_id[token]
        except KeyError:
            raise KeyError(token) from None


def test_unknown_token_raises_before_patch():
    tok = _StrictTokenizer()
    with pytest.raises(KeyError):
        tok.convert("\x00\x00")


@pytest.mark.parametrize("token", ["\x00\x00", "J", "|", "#", "\x00"])
def test_unknown_tokens_map_to_X_after_patch(token):
    tok = _StrictTokenizer()
    patch_unknown_residue_tokens(tok)
    # 'X', not <unk>: ESM2 was pretrained with 'X' as the unknown-residue symbol.
    assert tok.convert(token) == 24


def test_known_tokens_are_untouched():
    tok = _StrictTokenizer()
    patch_unknown_residue_tokens(tok)
    assert tok.convert("A") == 5
    assert tok.convert("L") == 4


def test_reported_vocab_does_not_grow():
    """__missing__ fires only on [], so the embedding matrix stays consistent."""
    tok = _StrictTokenizer()
    patch_unknown_residue_tokens(tok)
    assert "J" not in tok._token_to_id
    assert len(tok._token_to_id) == 4


def test_idempotent():
    tok = _StrictTokenizer()
    patch_unknown_residue_tokens(tok)
    first = tok._token_to_id
    patch_unknown_residue_tokens(tok)
    assert tok._token_to_id is first


def test_survives_pickling_into_a_worker():
    import pickle

    tok = _StrictTokenizer()
    patch_unknown_residue_tokens(tok)
    revived = pickle.loads(pickle.dumps(tok._token_to_id))
    assert isinstance(revived, _FallbackVocab)
    assert revived["\x00\x00"] == 24


def test_no_op_when_there_is_no_fallback_id():
    class _NoX:
        _token_to_id = {"A": 5}
        unk_token_id = None

    tok = _NoX()
    patch_unknown_residue_tokens(tok)
    assert not isinstance(tok._token_to_id, _FallbackVocab)
