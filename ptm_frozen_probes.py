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
from typing import Literal

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
    candidate_residues: frozenset[str] | None = None,
    outside_candidate_positives: Literal["error", "ignore"] = "error",
) -> list[int]:
    """Keep all positives and deterministic hard-cap sampling of negatives.

    Negative-only proteins retain one residue, avoiding a fit set composed only
    of proteins already known to contain a PTM site.
    """

    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")
    if candidate_residues is not None and not candidate_residues:
        raise ValueError("candidate_residues must not be empty")
    if outside_candidate_positives not in {"error", "ignore"}:
        raise ValueError("outside_candidate_positives must be 'error' or 'ignore'")
    positive_outside_candidates = [
        index
        for index, (residue, label) in enumerate(
            zip(example.sequence, example.labels, strict=True)
        )
        if label
        and candidate_residues is not None
        and residue not in candidate_residues
    ]
    if positive_outside_candidates and outside_candidate_positives == "error":
        raise ValueError(
            f"{example.row_id}: positive labels outside candidate residues at "
            f"{positive_outside_candidates[:5]}"
        )
    eligible = [
        index
        for index, residue in enumerate(example.sequence)
        if candidate_residues is None or residue in candidate_residues
    ]
    positives = [index for index in eligible if example.labels[index]]
    negatives = [index for index in eligible if not example.labels[index]]
    if not eligible:
        return []
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
    candidate_residues: frozenset[str] | None = None,
    outside_candidate_positives: Literal["error", "ignore"] = "error",
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
            candidate_residues=candidate_residues,
            outside_candidate_positives=outside_candidate_positives,
        )
        if positions:
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
    *,
    candidate_residues: frozenset[str] | None = None,
) -> list[PTMSitePrediction]:
    predictions: list[PTMSitePrediction] = []
    seen = 0
    for example, hidden in zip(examples, embeddings, strict=True):
        if len(hidden) != len(example.sequence):
            raise ValueError(
                f"embedding alignment mismatch for {example.row_id}: "
                f"{len(hidden)} != {len(example.sequence)}"
            )
        positions = [
            position
            for position, residue in enumerate(example.sequence)
            if candidate_residues is None or residue in candidate_residues
        ]
        if not positions:
            seen += 1
            continue
        scores = probe.predict_proba(_normalized(hidden[positions]))[:, 1]
        predictions.extend(
            PTMSitePrediction(
                example.row_id,
                position,
                example.labels[position],
                float(score),
            )
            for position, score in zip(positions, scores, strict=True)
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


def fit_center_residue_baseline(
    examples: Sequence[FrozenSequenceExample],
) -> dict[str, float]:
    """Fit the honest P(label | centre residue) baseline for window tasks."""

    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for example in examples:
        counts[example.sequence[len(example.sequence) // 2]][example.label] += 1
    return {
        residue: (values[1] + 1) / (values[0] + values[1] + 2)
        for residue, values in counts.items()
    }


def predict_center_residue_baseline(
    examples: Sequence[FrozenSequenceExample], probabilities: dict[str, float]
) -> list[PTMSitePrediction]:
    """Score one centre site per sequence with a fitted residue-only prior."""

    return [
        PTMSitePrediction(
            example.row_id,
            len(example.sequence) // 2,
            example.label,
            probabilities.get(example.sequence[len(example.sequence) // 2], 0.5),
        )
        for example in examples
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
