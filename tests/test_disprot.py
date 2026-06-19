"""Tests for the DisProt residue-level disorder benchmark addition."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH / "scripts"))

from benchmark_tasks import FAST_TASKS, TASKS, VERY_FAST_TASKS  # noqa: E402
from prep_disprot import build_residue_labels  # noqa: E402
from token_classification_probe import (  # noqa: E402
    EmbeddingCache,
    evaluate_token_classification,
)
from test_token_classification_probe import _TinyEncoder, _TinyTokenizer  # noqa: E402


def test_labels_one_based_inclusive():
    # single disorder region [2,4] (1-based inclusive) -> 0-based residues 1,2,3
    labels = build_residue_labels(6, [2], [4], ["disorder"])
    assert labels == [0, 1, 1, 1, 0, 0]
    # two disjoint disorder regions
    labels = build_residue_labels(7, [2, 5], [3, 6], ["disorder", "disorder"])
    assert labels == [0, 1, 1, 0, 1, 1, 0]


def test_non_disorder_terms_excluded():
    # only the 'disorder' region counts; 'protein binding' span is ignored
    labels = build_residue_labels(6, [2, 4], [3, 5], ["disorder", "protein binding"])
    assert labels == [0, 1, 1, 0, 0, 0]


def test_labels_reconstruct_disorder_content():
    # DP00003: length 529, two disorder regions [294,334] & [454,464] -> 52 residues.
    labels = build_residue_labels(
        529, [294, 454], [334, 464], ["disorder", "disorder"]
    )
    assert sum(labels) == 52
    assert abs(sum(labels) / 529 - 0.09829867674858224) < 1e-3


def test_labels_clip_out_of_range_end():
    labels = build_residue_labels(5, [3], [99], ["disorder"])  # end past length -> clip
    assert labels == [0, 0, 1, 1, 1]


def test_disprot_task_config():
    cfg = TASKS["disprot"]
    assert cfg.problem_type == "token_classification"
    assert cfg.main_metric == "MCC"
    assert cfg.label_col == "disorder_labels"
    assert cfg.dataset == "data/disprot"
    assert cfg.validation_split == "validation"
    assert cfg.test_split == "test"


def test_disprot_in_fast_not_very_fast():
    # Full benchmarks (FAST set drives run_full_bench.sh) include disprot;
    # the --very-fast scout subset excludes it.
    assert "disprot" in FAST_TASKS
    assert "disprot" not in VERY_FAST_TASKS


def test_disprot_probe_returns_mcc(tmp_path):
    """Tiny-encoder smoke through the real residue probe path -> MCC present."""
    enc = _TinyEncoder(hidden=16, seed=13)
    tok = _TinyTokenizer()
    rng = np.random.RandomState(0)
    aas = list("ACDEFGHIKLMNPQRSTVWY")

    def toy(n, length):
        seqs, labs = [], []
        for _ in range(n):
            chars = rng.choice(aas, size=length).tolist()
            # linearly separable: disorder iff residue in second half of alphabet
            labs.append([1 if aas.index(c) >= 10 else 0 for c in chars])
            seqs.append("".join(chars))
        return seqs, labs

    tr_seqs, tr_labs = toy(40, 20)
    te_seqs, te_labs = toy(16, 20)

    metrics = evaluate_token_classification(
        cfg=TASKS["disprot"],
        encoder=enc,
        tokenizer=tok,
        train_sequences=tr_seqs,
        train_labels=tr_labs,
        test_sequences=te_seqs,
        test_labels=te_labs,
        device="cpu",
        batch_size=8,
        max_length=64,
        cache=EmbeddingCache(tmp_path),
        model_hash="unit-test-disprot",
    )
    assert "MCC" in metrics and isinstance(metrics["MCC"], float)
    assert -1.0 <= metrics["MCC"] <= 1.0
