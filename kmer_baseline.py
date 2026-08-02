"""K-mer frequency vectors: the no-learning baseline.

Answers "does the pretrained model beat counting amino acid triplets?", which
is the floor any representation claim has to clear. Use it like any other
model, so it goes through the same probes, splits, metrics and CSV writer:

    python protein_benchmark_suite.py -m kmer --tasks solubility -p linear
    python protein_benchmark_suite.py -m kmer5 --tasks solubility   # k=5

The vocabulary is *fixed* (all k-mers over the 20 standard amino acids), not
fitted. A fitted vectorizer would derive its vocabulary from whichever split
it saw first, so train and test would end up with different columns -- or,
worse, silently agree in width while disagreeing on what each column means.
Fixed vocabulary also means no information crosses from test back into train.

Counts are normalised to frequencies. Without that, the vector length tracks
sequence length and the probe learns to predict from length alone, which for
several tasks is a real (and misleading) signal.

Dimensionality is 20**k: 400 / 8_000 / 160_000 for k=2 / 3 / 4. k>=4 is dense
and large -- pair it with --max_samples.
"""

from __future__ import annotations

import functools
import itertools
import re

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_K = 3
_NON_STANDARD = re.compile(f"[^{AMINO_ACIDS}]")


def parse_kmer_model_name(model_name: str) -> int | None:
    """``kmer`` -> DEFAULT_K, ``kmer4`` -> 4, anything else -> None."""
    m = re.fullmatch(r"kmer(\d*)", model_name.strip().lower())
    if m is None:
        return None
    return int(m.group(1)) if m.group(1) else DEFAULT_K


@functools.lru_cache(maxsize=None)
def _vocab(k: int) -> dict[str, int]:
    """All 20**k k-mers, in a fixed order. Cached: callers hit this per split."""
    return {
        "".join(t): i for i, t in enumerate(itertools.product(AMINO_ACIDS, repeat=k))
    }


def kmer_features(sequences: list[str], k: int = DEFAULT_K) -> np.ndarray:
    """(n_sequences, 20**k) float32 matrix of k-mer frequencies."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # 20**6 x 10k sequences is 2.5 TB. Fail with the arithmetic rather than
    # let someone discover it as an OOM twenty minutes into a sweep.
    gib = 20**k * max(len(sequences), 1) * 4 / 1024**3
    if gib > 16:
        raise ValueError(
            f"k={k} on {len(sequences):,} sequences needs {gib:,.0f} GiB "
            f"({20**k:,} dims, dense float32). Use a smaller k, or cap rows "
            f"with --max_samples."
        )

    vocab = _vocab(k)
    out = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
    for row, seq in enumerate(sequences):
        # X/B/Z/U and gaps break the sequence into segments rather than being
        # deleted. Deleting them would splice the flanks together and invent a
        # k-mer that spans the gap -- "AXA" would contribute an "AA" that is not
        # in the sequence.
        for seg in _NON_STANDARD.split(str(seq).upper()):
            for i in range(len(seg) - k + 1):
                out[row, vocab[seg[i : i + k]]] += 1.0
        total = out[row].sum()
        if total:
            out[row] /= total
    return out
