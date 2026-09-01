#!/usr/bin/env python3
"""Recompute a saved sequence/residue prediction artifact in identity strata."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from prediction_artifacts import read_prediction_rows, reproduce_classification_metrics

STRATA = ("exact", "90-<100", "50-<90", "30-<50", "<30_or_no_hit")


def _identity_table(path: Path) -> dict[str, str]:
    with path.open() as handle:
        rows = {
            row["query_id"]: row["stratum"]
            for row in csv.DictReader(handle, delimiter="\t")
        }
    if not rows:
        raise ValueError(f"empty identity table: {path}")
    return rows


def stratify(path: Path, identity_path: Path) -> dict[str, object]:
    metadata, rows = read_prediction_rows(path)
    identities = _identity_table(identity_path)
    missing = {str(row["sequence_sha256"]) for row in rows} - set(identities)
    if missing:
        raise ValueError(
            f"prediction sequences missing identity: {sorted(missing)[:3]}"
        )
    groups = defaultdict(list)
    for row in rows:
        groups[identities[str(row["sequence_sha256"])]].append(row)
    return {
        "schema_version": 1,
        "prediction": str(path),
        "prediction_metadata": metadata,
        "identity_table": str(identity_path),
        "full": reproduce_classification_metrics(path),
        "strata": {
            stratum: _metrics_for_rows(metadata, groups[stratum]) for stratum in STRATA
        },
        "protein_counts": {
            stratum: len({row["sequence_sha256"] for row in groups[stratum]})
            for stratum in STRATA
        },
    }


def _metrics_for_rows(
    metadata: dict[str, object], rows: list[dict[str, object]]
) -> dict[str, float]:
    # This compact local implementation mirrors ``reproduce_classification_metrics``
    # without creating a second artifact just to score a stratum.
    import numpy as np

    from protein_benchmark_suite import classification_metrics

    problem_type = str(metadata["problem_type"])
    labels = np.asarray([row["label"] for row in rows])
    predictions = np.asarray([row["prediction"] for row in rows])
    metrics = classification_metrics(problem_type, labels, predictions)
    if (
        problem_type == "binary"
        and len(np.unique(labels)) == 2
        and all(row["score"] is not None for row in rows)
    ):
        from sklearn.metrics import average_precision_score, roc_auc_score

        scores = np.asarray([row["score"] for row in rows], dtype=float)
        try:
            metrics["AUC"] = float(roc_auc_score(labels, scores))
            metrics["AP"] = float(average_precision_score(labels, scores))
        except ValueError:
            pass
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = stratify(args.prediction, args.identity)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
