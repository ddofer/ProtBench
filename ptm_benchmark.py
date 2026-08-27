"""Reusable records, adapter contract, metrics, and persistence for PTM tasks.

Model-specific loading belongs in a thin repository adapter.  The records and
scoring here are deliberately model-agnostic, so Proteva, vanilla backbones, and
future PTM models can be compared without duplicating metric code.
"""

from __future__ import annotations

import json
import math
import gzip
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)


@dataclass(frozen=True)
class PTMSequenceInput:
    """One protein and its optional residue-aligned PTM targets."""

    row_id: str
    sequence: str
    ptm_any: tuple[int, ...] | None = None
    ptm_token_ids: tuple[int, ...] | None = None
    ptm_class_masks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.row_id or not self.sequence:
            raise ValueError("row_id and sequence must be non-empty")
        length = len(self.sequence)
        for name in ("ptm_any", "ptm_token_ids", "ptm_class_masks"):
            values = getattr(self, name)
            if values is not None and len(values) != length:
                raise ValueError(f"{name} length {len(values)} != sequence length {length}")
        if self.ptm_any is not None and any(value not in (0, 1) for value in self.ptm_any):
            raise ValueError("ptm_any must be binary")


@dataclass(frozen=True)
class PTMSitePrediction:
    row_id: str
    position: int
    label: int
    score: float


@dataclass(frozen=True)
class PTMTypePrediction:
    row_id: str
    position: int
    target_type: str
    ranked_types: tuple[str, ...] | None
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class PTMAdapterCapabilities:
    native_ptm_tokens: bool
    canonicalized_inputs: bool
    direct_ptm_any: bool
    type_ranking: bool


class PTMModelAdapter(Protocol):
    """Minimal model adapter used by a repository-specific PTM launcher."""

    model_id: str
    capabilities: PTMAdapterCapabilities

    def predict(
        self, inputs: Sequence[PTMSequenceInput]
    ) -> tuple[list[PTMSitePrediction], list[PTMTypePrediction]]:
        """Return residue-site scores and/or supported PTM-type rankings."""


def _decode_track(value: object, *, dtype: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        array = np.frombuffer(value, dtype=np.dtype(dtype))
    else:
        array = np.asarray(value, dtype=np.dtype(dtype))
    return tuple(int(item) for item in array)


def structured_ptm_input(row: dict[str, object]) -> PTMSequenceInput:
    """Decode Proteva-compatible packed tracks into a model-neutral record."""

    required = {"row_id", "sequence"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"missing PTM input fields: {sorted(missing)}")
    sequence = "".join(str(row["sequence"]).split()).upper()
    return PTMSequenceInput(
        row_id=str(row["row_id"]),
        sequence=sequence,
        ptm_any=_decode_track(row.get("ptm_any_u8"), dtype="u1"),
        ptm_token_ids=_decode_track(row.get("ptm_token_id"), dtype="<u2"),
        ptm_class_masks=_decode_track(row.get("ptm_class_mask"), dtype="<u2"),
    )


def score_ptm_sites(
    predictions: Iterable[PTMSitePrediction], *, threshold: float = 0.5
) -> dict[str, float | int | None]:
    """Score direct residue-level ``ptm_any`` probabilities."""

    records = list(predictions)
    if not records:
        return {
            "n_sites": 0,
            "n_positive": 0,
            "auprc": None,
            "auroc": None,
            "mcc": None,
            "f1": None,
        }
    labels = np.asarray([record.label for record in records], dtype=np.int8)
    scores = np.asarray([record.score for record in records], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("PTM site scores must all be finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("PTM site labels must be binary")
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & (labels == 1)))
    fp = int(np.count_nonzero(predicted & (labels == 0)))
    fn = int(np.count_nonzero(~predicted & (labels == 1)))
    f1_denominator = 2 * tp + fp + fn
    both_classes = np.unique(labels).size == 2
    return {
        "n_sites": len(records),
        "n_positive": int(labels.sum()),
        "auprc": float(average_precision_score(labels, scores)) if labels.any() else None,
        "auroc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "mcc": float(matthews_corrcoef(labels, predicted)) if both_classes else None,
        "f1": (2 * tp / f1_denominator) if f1_denominator else 0.0,
        "threshold": threshold,
    }


def score_ptm_types(
    predictions: Iterable[PTMTypePrediction],
) -> dict[str, object]:
    """Score PTM type Top-1/Top-3/MRR while preserving unsupported coverage."""

    records = list(predictions)
    supported = [record for record in records if record.ranked_types is not None]
    unsupported = Counter(
        record.target_type
        for record in records
        if record.ranked_types is None
    )
    unsupported_reasons = Counter(
        record.unsupported_reason or "unspecified"
        for record in records
        if record.ranked_types is None
    )
    top1 = 0
    top3 = 0
    reciprocal_rank = 0.0
    for record in supported:
        assert record.ranked_types is not None
        ranking = record.ranked_types
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"duplicate ranked PTM types for {record.row_id}:{record.position}")
        if ranking and ranking[0] == record.target_type:
            top1 += 1
        if record.target_type in ranking[:3]:
            top3 += 1
        try:
            rank = ranking.index(record.target_type) + 1
        except ValueError:
            rank = 0
        reciprocal_rank += 1.0 / rank if rank else 0.0

    n_total = len(records)
    n_supported = len(supported)
    return {
        "n_total": n_total,
        "n_supported": n_supported,
        "mapping_coverage": n_supported / n_total if n_total else 0.0,
        "top1": top1 / n_supported if n_supported else None,
        "top3": top3 / n_supported if n_supported else None,
        "mrr": reciprocal_rank / n_supported if n_supported else None,
        "unsupported_types": dict(sorted(unsupported.items())),
        "unsupported_reasons": dict(sorted(unsupported_reasons.items())),
    }


def score_ptm_predictions(
    site_predictions: Iterable[PTMSitePrediction],
    type_predictions: Iterable[PTMTypePrediction],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    return {
        "site": score_ptm_sites(site_predictions, threshold=threshold),
        "type": score_ptm_types(type_predictions),
    }


def write_ptm_predictions(
    path: Path,
    *,
    site_predictions: Iterable[PTMSitePrediction],
    type_predictions: Iterable[PTMTypePrediction],
    metadata: dict[str, object] | None = None,
) -> None:
    """Persist deterministic JSONL records for score reproduction/stratification."""

    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"kind": "metadata", "payload": metadata or {}}, sort_keys=True)
            + "\n"
        )
        for site_record in sorted(
            site_predictions, key=lambda item: (item.row_id, item.position)
        ):
            handle.write(
                json.dumps(
                    {"kind": "site", "payload": asdict(site_record)}, sort_keys=True
                )
                + "\n"
            )
        for type_record in sorted(
            type_predictions, key=lambda item: (item.row_id, item.position)
        ):
            handle.write(
                json.dumps(
                    {"kind": "type", "payload": asdict(type_record)}, sort_keys=True
                )
                + "\n"
            )


def read_ptm_predictions(
    path: Path,
) -> tuple[dict[str, object], list[PTMSitePrediction], list[PTMTypePrediction]]:
    metadata: dict[str, object] = {}
    sites: list[PTMSitePrediction] = []
    types: list[PTMTypePrediction] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            kind = row.get("kind")
            payload = row.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: payload is not an object")
            if kind == "metadata":
                metadata = payload
            elif kind == "site":
                sites.append(PTMSitePrediction(**payload))
            elif kind == "type":
                ranked = payload.get("ranked_types")
                payload["ranked_types"] = None if ranked is None else tuple(ranked)
                types.append(PTMTypePrediction(**payload))
            else:
                raise ValueError(f"{path}:{line_number}: unknown prediction kind {kind!r}")
    return metadata, sites, types


def assert_same_metrics(left: object, right: object, *, atol: float = 1e-12) -> None:
    """Recursively assert metric equivalence after prediction persistence."""

    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"metric keys differ: {left.keys()} != {right.keys()}")
        for key in left:
            assert_same_metrics(left[key], right[key], atol=atol)
        return
    if isinstance(left, float) and isinstance(right, float):
        if not math.isclose(left, right, abs_tol=atol, rel_tol=0.0):
            raise AssertionError(f"metric values differ: {left} != {right}")
        return
    if left != right:
        raise AssertionError(f"metric values differ: {left!r} != {right!r}")
