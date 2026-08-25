"""Longest-first batching in the generic HF path must not reorder its output."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_benchmark_suite import embed_sequences  # noqa: E402

TINY = "facebook/esm2_t6_8M_UR50D"
# Unsorted and unevenly sized on purpose: sorting must genuinely permute these, or the test cannot
# tell a correct scatter from none at all. Lengths here are 22, 72, 4, 33, 16.
SEQS = [
    "MKTAYIAKQRQISFVKSHFSRQ",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHS",
    "GSHM",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "GSHMLEDPQTRAWCEV",
]


@pytest.fixture(scope="module")
def hf_bundle():
    """A raw (tokenizer, model) tuple -- the shape that reaches the generic batched loop.

    Loading this checkpoint through load_model returns a SentenceTransformer, which takes the SBERT
    branch and batches internally, so it would never exercise the code under test.
    """
    try:
        from transformers import AutoModel, AutoTokenizer

        return AutoTokenizer.from_pretrained(TINY), AutoModel.from_pretrained(TINY).eval()
    except Exception as exc:  # no network / no cache
        pytest.skip(f"cannot load {TINY}: {exc}")


def test_batched_embeddings_keep_input_order(hf_bundle):
    """Each row must belong to the sequence at the same index of the input.

    The loop batches longest-first to cut padding waste, then scatters results back. A scatter that
    is off -- or simply left in sorted order -- still returns the right *set* of vectors, so shapes
    and norms look fine while every embedding is attached to the wrong protein. Comparing against
    one-sequence-per-batch is what catches it, since no reordering can happen there.
    """
    kw = dict(model_obj=hf_bundle, is_sbert=False, device="cpu", max_length=128)
    batched = embed_sequences(sequences=SEQS, batch_size=4, **kw)
    one_at_a_time = np.stack(
        [embed_sequences(sequences=[s], batch_size=1, **kw)[0] for s in SEQS]
    )

    assert batched.shape == one_at_a_time.shape
    for i, seq in enumerate(SEQS):
        assert np.allclose(batched[i], one_at_a_time[i], atol=1e-4), f"row {i} ({len(seq)} aa)"
