"""Reusable frozen linear probes for external PTM benchmarks.

The caller owns model loading and embedding extraction.  This module owns the
deterministic residue sampling, L2-normalized probe fitting, predictions, and a
residue-identity baseline so every model uses exactly the same protocol.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from ptm_benchmark import PTMSitePrediction


@dataclass(frozen=True)
class FrozenSiteExample:
    row_id: str
    sequence: str
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.row_id or not self.sequence:
            raise ValueError("row_id and sequence must be non-empty")
        if len(self.sequence) != len(self.labels):
            raise ValueError("site labels must align one-to-one with the sequence")
        if any(label not in (0, 1) for label in self.labels):
            raise ValueError("site labels must be binary")


@dataclass(frozen=True)
class FrozenSequenceExample:
    row_id: str
    sequence: str
    label: int

    def __post_init__(self) -> None:
        if not self.row_id or not self.sequence:
            raise ValueError("row_id and sequence must be non-empty")
        if self.label not in (0, 1):
            raise ValueError("sequence label must be binary")


def _normalized(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, np.finfo(np.float32).eps)


def sampled_fit_positions(
    example: FrozenSiteExample,
    *,
    seed: int,
    negatives_per_positive: int,
) -> list[int]:
    """Keep all positives and deterministic hard-cap sampling of negatives.

    Negative-only proteins retain one residue, avoiding a fit set composed only
    of proteins already known to contain a PTM site.
    """

    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")
    positives = [index for index, label in enumerate(example.labels) if label]
    negatives = [index for index, label in enumerate(example.labels) if not label]
    n_negative = min(
        len(negatives),
        max(1, negatives_per_positive * len(positives)),
    )
    chosen_negatives = sorted(
        negatives,
        key=lambda index: hashlib.sha256(
            f"{seed}\0{example.row_id}\0{index}".encode()
        ).digest(),
    )[:n_negative]
    return sorted([*positives, *chosen_negatives])


def fit_site_probe(
    examples: Sequence[FrozenSiteExample],
    embeddings: Iterable[np.ndarray],
    *,
    seed: int = 1337,
    negatives_per_positive: int = 5,
) -> tuple[LogisticRegression, dict[str, int]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    seen = 0
    for example, hidden in zip(examples, embeddings, strict=True):
        if len(hidden) != len(example.sequence):
            raise ValueError(
                f"embedding alignment mismatch for {example.row_id}: "
                f"{len(hidden)} != {len(example.sequence)}"
            )
        positions = sampled_fit_positions(
            example,
            seed=seed,
            negatives_per_positive=negatives_per_positive,
        )
        features.append(_normalized(hidden[positions]))
        labels.extend(example.labels[position] for position in positions)
        seen += 1
    if seen != len(examples):
        raise ValueError(f"embedding iterator yielded {seen}/{len(examples)} proteins")
    if not features or len(set(labels)) != 2:
        raise ValueError("site probe fit data must contain both classes")
    x = np.concatenate(features)
    y = np.asarray(labels, dtype=np.int8)
    probe = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    ).fit(x, y)
    return probe, {
        "fit_proteins": len(examples),
        "fit_residues": len(y),
        "fit_positive": int(y.sum()),
        "fit_negative": int(len(y) - y.sum()),
    }


def predict_site_probe(
    examples: Sequence[FrozenSiteExample],
    embeddings: Iterable[np.ndarray],
    probe: LogisticRegression,
) -> list[PTMSitePrediction]:
    predictions: list[PTMSitePrediction] = []
    seen = 0
    for example, hidden in zip(examples, embeddings, strict=True):
        if len(hidden) != len(example.sequence):
            raise ValueError(
                f"embedding alignment mismatch for {example.row_id}: "
                f"{len(hidden)} != {len(example.sequence)}"
            )
        scores = probe.predict_proba(_normalized(hidden))[:, 1]
        predictions.extend(
            PTMSitePrediction(example.row_id, position, label, float(scores[position]))
            for position, label in enumerate(example.labels)
        )
        seen += 1
    if seen != len(examples):
        raise ValueError(f"embedding iterator yielded {seen}/{len(examples)} proteins")
    return predictions


def fit_residue_identity_baseline(
    examples: Sequence[FrozenSiteExample],
) -> dict[str, float]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for example in examples:
        for residue, label in zip(example.sequence, example.labels, strict=True):
            counts[residue][label] += 1
    return {
        residue: (values[1] + 1) / (values[0] + values[1] + 2)
        for residue, values in counts.items()
    }


def predict_residue_identity_baseline(
    examples: Sequence[FrozenSiteExample], probabilities: dict[str, float]
) -> list[PTMSitePrediction]:
    return [
        PTMSitePrediction(
            example.row_id,
            position,
            label,
            probabilities.get(residue, 0.5),
        )
        for example in examples
        for position, (residue, label) in enumerate(
            zip(example.sequence, example.labels, strict=True)
        )
    ]


def fit_sequence_probe(
    examples: Sequence[FrozenSequenceExample],
    embeddings: np.ndarray,
    *,
    seed: int = 1337,
) -> LogisticRegression:
    if len(embeddings) != len(examples):
        raise ValueError("one sequence embedding is required per example")
    labels = np.asarray([example.label for example in examples], dtype=np.int8)
    if len(set(labels.tolist())) != 2:
        raise ValueError("sequence probe fit data must contain both classes")
    return LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    ).fit(_normalized(embeddings), labels)


def predict_sequence_probe(
    examples: Sequence[FrozenSequenceExample],
    embeddings: np.ndarray,
    probe: LogisticRegression,
    *,
    position: int,
) -> list[PTMSitePrediction]:
    if len(embeddings) != len(examples):
        raise ValueError("one sequence embedding is required per example")
    scores = probe.predict_proba(_normalized(embeddings))[:, 1]
    return [
        PTMSitePrediction(example.row_id, position, example.label, float(score))
        for example, score in zip(examples, scores, strict=True)
    ]
