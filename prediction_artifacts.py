"""Compact, deterministic prediction artifacts for post-hoc audit/rescoring."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256("".join(sequence.split()).upper().encode()).hexdigest()


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def write_query_fasta(path: Path, sequences: Sequence[str]) -> None:
    """Write one outside-Git homology query per unique prediction sequence."""

    unique: dict[str, str] = {}
    for sequence in sequences:
        normalized = "".join(sequence.split()).upper()
        unique.setdefault(_sequence_sha256(normalized), normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for sequence_hash in sorted(unique):
            handle.write(f">{sequence_hash}\n{unique[sequence_hash]}\n")


def write_sequence_predictions(
    path: Path,
    *,
    sequences: Sequence[str],
    labels: Sequence[Any],
    predictions: Sequence[Any],
    scores: Sequence[Any] | None,
    metadata: dict[str, Any],
    query_fasta_path: Path | None = None,
) -> None:
    if len(sequences) != len(labels) or len(labels) != len(predictions):
        raise ValueError("sequence prediction lengths differ")
    if scores is not None and len(scores) != len(labels):
        raise ValueError("score and label lengths differ")
    if query_fasta_path is not None:
        write_query_fasta(query_fasta_path, sequences)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write(
            json.dumps(
                {"schema_version": 1, "record_type": "metadata", **metadata},
                sort_keys=True,
            )
            + "\n"
        )
        for index, (sequence, label, prediction) in enumerate(
            zip(sequences, labels, predictions, strict=True)
        ):
            row = {
                "example_id": index,
                "sequence_sha256": _sequence_sha256(sequence),
                "label": _scalar(label),
                "prediction": _scalar(prediction),
                "score": None if scores is None else _scalar(scores[index]),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_prediction_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or rows[0].get("record_type") != "metadata":
        raise ValueError(f"missing metadata record: {path}")
    return rows[0], rows[1:]


def reproduce_classification_metrics(path: Path) -> dict[str, float]:
    """Recompute point metrics from a persisted binary or multiclass artifact."""

    metadata, rows = read_prediction_rows(path)
    problem_type = metadata.get("problem_type")
    if problem_type not in {"binary", "multiclass"}:
        raise ValueError(f"unsupported prediction problem type: {problem_type}")
    if not rows:
        raise ValueError("no prediction rows")
    from protein_benchmark_suite import classification_metrics

    labels = np.asarray([row["label"] for row in rows])
    predictions = np.asarray([row["prediction"] for row in rows])
    metrics = classification_metrics(problem_type, labels, predictions)
    if problem_type == "binary" and all(row["score"] is not None for row in rows):
        from sklearn.metrics import average_precision_score, roc_auc_score

        scores = np.asarray([row["score"] for row in rows], dtype=float)
        try:
            metrics["AUC"] = float(roc_auc_score(labels, scores))
            metrics["AP"] = float(average_precision_score(labels, scores))
        except ValueError:
            pass
    return metrics


def _residue_score(scores: Any, i: int) -> Any:
    if scores is None:
        return None
    row = np.asarray(scores[i], dtype=float).ravel()
    return round(float(row[-1]), 4) if row.size <= 2 else [round(float(v), 4) for v in row]


def write_residue_predictions(
    path: Path,
    *,
    sequences: Sequence[str],
    groups: Sequence[int],
    labels: Sequence[Any],
    predictions: Sequence[Any],
    metadata: dict[str, Any],
    scores: Any = None,
    query_fasta_path: Path | None = None,
) -> None:
    """``scores``: predict_proba rows aligned to ``metadata["classes"]``; stored
    as the positive-class probability for two classes, else the full row."""
    if len(groups) != len(labels) or len(labels) != len(predictions):
        raise ValueError("residue prediction lengths differ")
    if scores is not None and len(scores) != len(labels):
        raise ValueError("score and label lengths differ")
    positions: dict[int, int] = {}
    if query_fasta_path is not None:
        write_query_fasta(query_fasta_path, sequences)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write(
            json.dumps(
                {"schema_version": 1, "record_type": "metadata", **metadata},
                sort_keys=True,
            )
            + "\n"
        )
        for i, (group, label, prediction) in enumerate(
            zip(groups, labels, predictions, strict=True)
        ):
            example_id = int(group)
            position = positions.get(example_id, 0)
            positions[example_id] = position + 1
            row = {
                "example_id": example_id,
                "position": position,
                "sequence_sha256": _sequence_sha256(sequences[example_id]),
                "label": _scalar(label),
                "prediction": _scalar(prediction),
                "score": _residue_score(scores, i),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
