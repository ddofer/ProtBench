"""TDD tests for residue-level (token-classification) linear-probe path.

Wires SS3 / Disorder benchmarks into the existing linear-probe pipeline
by extracting per-residue hidden states once per (model, task, split)
into a parquet/npz cache, then fitting an sklearn LogisticRegression
linear probe over the cached residue embeddings.

The tests use a tiny synthetic encoder (single ``torch.nn.Embedding`` +
projection) and a stub task config — no AMPLIFY/HF downloads needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Make ``plm/bench`` importable; conftest already does this but be defensive.
_BENCH = Path(__file__).resolve().parent.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from benchmark_tasks import TaskConfig  # noqa: E402

# Under test (created during GREEN phase).
from token_classification_probe import (  # noqa: E402
    EmbeddingCache,
    cache_key,
    evaluate_token_classification,
    extract_residue_embeddings,
    fit_residue_linear_probe,
)


# ---------------------------------------------------------------------------
# Tiny synthetic encoder + tokenizer
# ---------------------------------------------------------------------------


class _TinyEncoder:
    """Stand-in for an HF model: embeds each token id into a small vector."""

    def __init__(self, hidden: int = 8, vocab: int = 32, seed: int = 0):
        import torch

        gen = torch.Generator().manual_seed(seed)
        self.hidden = hidden
        # (vocab, hidden) embedding table
        self.W = torch.randn(vocab, hidden, generator=gen)
        # Mimic AMPLIFY/HF API: model.config.model_type used by the suite.
        self.config = type("Cfg", (), {"model_type": "tiny"})()

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        import torch

        # (B, L, H)
        last_hidden_state = self.W[input_ids]
        return type(
            "Output",
            (),
            {
                "last_hidden_state": last_hidden_state,
                "hidden_states": (last_hidden_state,),
            },
        )()

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        import torch

        return iter([torch.zeros(1)])


class _FakeAmplifyEncoder(_TinyEncoder):
    """AMPLIFY-like encoder that, like the real model, REJECTS a boolean
    attention_mask. The residue path must convert it to an additive
    (float, -inf-on-pad) mask first — otherwise this raises exactly the
    error the real AMPLIFY raised in the bench run.

    Also exposes ``layer_norm_2`` (AMPLIFY's final norm, applied by the fix)
    and a float param so ``_prepare_amplify_inputs`` can match dtype.
    """

    def __init__(self, hidden: int = 8, vocab: int = 32, seed: int = 0):
        super().__init__(hidden=hidden, vocab=vocab, seed=seed)
        self.config = type("Cfg", (), {"model_type": "AMPLIFY"})()

    def layer_norm_2(self, hidden):  # identity is enough for the unit test
        return hidden

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        assert (
            attention_mask is not None
            and attention_mask.dtype.is_floating_point
        ), "AMPLIFY expects an additive attention_mask."
        return super().__call__(input_ids, attention_mask=attention_mask, **kwargs)

    def parameters(self):
        import torch

        return iter([torch.zeros(1, dtype=torch.float32)])


class _TinyTokenizer:
    """Char-level tokenizer: AA -> id starting at 4 (0..3 reserved)."""

    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2

    def __init__(self):
        # 20 standard AAs at positions 4..23
        self._alpha = "ACDEFGHIKLMNPQRSTVWY"
        self._tok = {a: i + 4 for i, a in enumerate(self._alpha)}

    def __call__(
        self,
        sequences,
        return_tensors=None,
        padding=True,
        truncation=True,
        max_length=128,
        return_special_tokens_mask=False,
    ):
        import torch

        if isinstance(sequences, str):
            sequences = [sequences]
        ids: List[List[int]] = []
        stm: List[List[int]] = []
        for s in sequences:
            s = s[: max_length - 2]
            row = (
                [self.cls_token_id]
                + [self._tok.get(c, 3) for c in s]
                + [self.sep_token_id]
            )
            ids.append(row)
            stm.append([1] + [0] * len(s) + [1])
        # right-pad
        max_len = max(len(r) for r in ids)
        attn = []
        for i in range(len(ids)):
            pad = max_len - len(ids[i])
            attn.append([1] * len(ids[i]) + [0] * pad)
            ids[i] = ids[i] + [self.pad_token_id] * pad
            stm[i] = stm[i] + [1] * pad
        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
        if return_special_tokens_mask:
            out["special_tokens_mask"] = torch.tensor(stm, dtype=torch.long)
        return out


@dataclass
class _Sample:
    sequence: str
    labels: List[int]


def _toy_ss3_dataset(seed: int = 0, n: int = 32, length: int = 24) -> List[_Sample]:
    rng = np.random.RandomState(seed)
    samples: List[_Sample] = []
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    for _ in range(n):
        seq_chars = rng.choice(aas, size=length).tolist()
        labels = [aas.index(c) % 3 for c in seq_chars]
        samples.append(_Sample("".join(seq_chars), labels))
    return samples


def _toy_conservation_dataset(seed: int = 0, n: int = 32, length: int = 20) -> List[_Sample]:
    """9-class conservation dataset (labels 1-9) matching the conservation_flip format."""
    rng = np.random.RandomState(seed)
    samples: List[_Sample] = []
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    for _ in range(n):
        seq_chars = rng.choice(aas, size=length).tolist()
        # Labels 1-9 keyed from AA index modulo 9, plus 1 (mirrors FLIP conservation scores)
        labels = [aas.index(c) % 9 + 1 for c in seq_chars]
        samples.append(_Sample("".join(seq_chars), labels))
    return samples


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_residue_embeddings_shape():
    """Stack of per-residue embeddings should equal sum of per-protein lengths."""
    enc = _TinyEncoder(hidden=8)
    tok = _TinyTokenizer()
    samples = [_Sample("ACDEFG", [0, 1, 2, 0, 1, 2]), _Sample("AAAA", [0, 0, 0, 0])]
    X, y = extract_residue_embeddings(
        encoder=enc,
        tokenizer=tok,
        sequences=[s.sequence for s in samples],
        labels=[s.labels for s in samples],
        device="cpu",
        batch_size=2,
        max_length=64,
    )
    assert X.shape == (10, 8), f"expected (10, 8), got {X.shape}"
    assert y.shape == (10,)
    # Labels preserved in order
    np.testing.assert_array_equal(y, np.array([0, 1, 2, 0, 1, 2, 0, 0, 0, 0]))


def test_padding_excluded():
    """Padding and special-token positions must NOT appear in the output."""
    enc = _TinyEncoder(hidden=8)
    tok = _TinyTokenizer()
    # Two sequences of length 3 and 6 -> total non-special residues = 9
    samples = [_Sample("ACD", [0, 1, 2]), _Sample("EFGHIK", [2, 1, 0, 2, 1, 0])]
    X, y = extract_residue_embeddings(
        encoder=enc,
        tokenizer=tok,
        sequences=[s.sequence for s in samples],
        labels=[s.labels for s in samples],
        device="cpu",
        batch_size=2,
        max_length=32,
    )
    assert X.shape[0] == 9
    assert y.shape == (9,)


def test_logreg_fit_predict_ss3():
    """Linear probe on a tiny deterministic SS3 dataset beats chance (>0.4)."""
    enc = _TinyEncoder(hidden=16, seed=7)
    tok = _TinyTokenizer()
    train = _toy_ss3_dataset(seed=1, n=64, length=20)
    test = _toy_ss3_dataset(seed=2, n=32, length=20)

    X_tr, y_tr = extract_residue_embeddings(
        encoder=enc,
        tokenizer=tok,
        sequences=[s.sequence for s in train],
        labels=[s.labels for s in train],
        device="cpu",
        batch_size=8,
        max_length=64,
    )
    X_te, y_te = extract_residue_embeddings(
        encoder=enc,
        tokenizer=tok,
        sequences=[s.sequence for s in test],
        labels=[s.labels for s in test],
        device="cpu",
        batch_size=8,
        max_length=64,
    )
    probe = fit_residue_linear_probe(X_tr, y_tr, problem_type="multiclass")
    preds = probe.predict(X_te)
    acc = float(np.mean(preds == y_te))
    # 3-class chance = ~0.33; the synthetic mapping is fully linear so this
    # should comfortably clear 0.4.
    assert acc > 0.4, f"accuracy {acc} not above chance"


def test_embedding_cache_roundtrip(tmp_path):
    """Write embeddings to cache, read back byte-exact."""
    cache = EmbeddingCache(tmp_path)
    X = np.random.RandomState(0).randn(17, 11).astype("float32")
    y = np.arange(17, dtype="int64")
    key = "test_model_hash::ss3::train"
    cache.put(key, X, y)
    X2, y2 = cache.get(key)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)


def test_cache_key_uses_model_hash():
    """Different model hashes -> different cache keys (no collisions)."""
    k1 = cache_key("model_aaa", "ss3", "train")
    k2 = cache_key("model_bbb", "ss3", "train")
    assert k1 != k2, "cache key must depend on model hash"
    # Stable across calls
    assert cache_key("model_aaa", "ss3", "train") == k1
    # Different task -> different key
    assert cache_key("model_aaa", "disorder", "train") != k1
    # Different split -> different key
    assert cache_key("model_aaa", "ss3", "test") != k1


def test_task_exception_resolved(tmp_path):
    """A synthetic SS3 task config goes through the residue-probe path
    and returns a results dict with the main_metric populated (not the
    historic ``task_exception='token_classification'`` error path)."""
    enc = _TinyEncoder(hidden=16, seed=11)
    tok = _TinyTokenizer()
    train = _toy_ss3_dataset(seed=3, n=64, length=18)
    test = _toy_ss3_dataset(seed=4, n=32, length=18)

    cfg = TaskConfig(
        name="SS3 (test)",
        dataset="local://synthetic",
        input_map={"seq": "sequence"},
        label_col="labels",
        problem_type="token_classification",
        main_metric="Accuracy",
    )

    metrics = evaluate_token_classification(
        cfg=cfg,
        encoder=enc,
        tokenizer=tok,
        train_sequences=[s.sequence for s in train],
        train_labels=[s.labels for s in train],
        test_sequences=[s.sequence for s in test],
        test_labels=[s.labels for s in test],
        device="cpu",
        batch_size=8,
        max_length=64,
        cache=EmbeddingCache(tmp_path),
        model_hash="unit-test",
    )
    assert "Accuracy" in metrics, f"metrics missing Accuracy: {metrics}"
    assert isinstance(metrics["Accuracy"], float)
    # Sanity bounds
    assert 0.0 <= metrics["Accuracy"] <= 1.0


def test_conservation_9class_labels_decoded_correctly():
    """Labels 1-9 (conservation_flip format) pass through _decode_residue_label
    as a plain list[int] and produce 9-class F1_Macro metric in [0, 1]."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from protein_benchmark_suite import _decode_residue_label

    raw = [1, 3, 5, 7, 9, 2, 4, 6, 8]
    decoded = _decode_residue_label("Residue Conservation (FLIP)", "conservation_labels", raw)
    assert decoded == [1, 3, 5, 7, 9, 2, 4, 6, 8], f"unexpected decode: {decoded}"


def test_conservation_9class_probe_returns_f1_macro(tmp_path):
    """9-class conservation probe returns F1_Macro in [0, 1] and beats trivial baseline."""
    enc = _TinyEncoder(hidden=16, seed=99)
    tok = _TinyTokenizer()
    train = _toy_conservation_dataset(seed=5, n=80, length=20)
    test = _toy_conservation_dataset(seed=6, n=40, length=20)

    cfg = TaskConfig(
        name="Residue Conservation (FLIP)",
        dataset="data/conservation_flip",
        input_map={"seq": "sequence"},
        label_col="conservation_labels",
        problem_type="token_classification",
        main_metric="F1_Macro",
    )

    metrics = evaluate_token_classification(
        cfg=cfg,
        encoder=enc,
        tokenizer=tok,
        train_sequences=[s.sequence for s in train],
        train_labels=[s.labels for s in train],
        test_sequences=[s.sequence for s in test],
        test_labels=[s.labels for s in test],
        device="cpu",
        batch_size=8,
        max_length=64,
        cache=EmbeddingCache(tmp_path),
        model_hash="unit-test-conservation",
    )
    assert "F1_Macro" in metrics, f"F1_Macro missing from metrics: {metrics}"
    f1 = metrics["F1_Macro"]
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0, f"F1_Macro out of range: {f1}"
    # Chance for 9-class = ~0.11; deterministic mapping should beat it.
    assert f1 > 0.11, f"F1_Macro {f1:.3f} not above 9-class chance"


def test_fast_task_lists_include_conservation():
    """conservation_flip must appear in both FAST_TASKS and VERY_FAST_TASKS."""
    from benchmark_tasks import FAST_TASKS, VERY_FAST_TASKS

    assert "conservation_flip" in FAST_TASKS, "conservation_flip missing from FAST_TASKS"
    assert "conservation_flip" in VERY_FAST_TASKS, "conservation_flip missing from VERY_FAST_TASKS"
    assert "ss3" in FAST_TASKS
    assert "ss3" in VERY_FAST_TASKS


def test_amplify_residue_embeddings_use_additive_mask():
    """Regression: AMPLIFY needs an ADDITIVE attention mask in the residue path.

    A boolean mask raised ``AMPLIFY expects an additive attention_mask`` and
    blanked SS3 / Residue Conservation for the AMPLIFY baseline. The fake
    AMPLIFY encoder reproduces that assertion; a clean extraction proves the
    residue path now converts the mask (mirroring the sequence-embed path)."""
    enc = _FakeAmplifyEncoder(hidden=8)
    tok = _TinyTokenizer()
    samples = [_Sample("ACDEFG", [0, 1, 2, 0, 1, 2]), _Sample("AAAA", [0, 0, 0, 0])]
    X, y = extract_residue_embeddings(
        encoder=enc,
        tokenizer=tok,
        sequences=[s.sequence for s in samples],
        labels=[s.labels for s in samples],
        device="cpu",
        batch_size=2,
        max_length=64,
    )
    # 6 + 4 non-special residues, 8-dim hidden, all finite.
    assert X.shape == (10, 8), f"expected (10, 8), got {X.shape}"
    assert y.shape == (10,)
    assert np.isfinite(X).all()
