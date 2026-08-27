from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scope_stratify import (
    Identity,
    build_records,
    load_identity_table,
    paired_delta,
    parse_expected_full,
    score_directory,
    score_records,
    sequence_hash,
)


def _identities(sequences: list[str]) -> dict[str, Identity]:
    out = {}
    for index, sequence in enumerate(sequences):
        query_hash = sequence_hash(sequence)
        out[query_hash] = Identity(
            query_hash=query_hash,
            max_identity=[1.0, 0.95, 0.5, 0.0][index],
            exact=index == 0,
            ge_90=index < 2,
            ge_30=index < 3,
        )
    return out


def test_scope_records_score_and_paired_delta_are_query_aligned() -> None:
    sequences = ["AAAA", "AAAC", "KKKK", "KKKL"]
    labels = np.asarray(["a", "a", "b", "b"])
    identities = _identities(sequences)
    good = np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    bad = np.asarray([[1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    candidate = build_records(
        model_tag="candidate",
        model_name="/candidate",
        domain_ids=["d1", "d2", "d3", "d4"],
        sequences=sequences,
        labels=labels,
        embeddings=good,
        identities=identities,
    )
    baseline = build_records(
        model_tag="baseline",
        model_name="/baseline",
        domain_ids=["d1", "d2", "d3", "d4"],
        sequences=sequences,
        labels=labels,
        embeddings=bad,
        identities=identities,
    )
    score = score_records(candidate, n_boot=100)
    assert score["strata"]["full"]["n"] == 4
    assert score["strata"]["lt90"]["n"] == 2
    assert score["strata"]["full"]["recall_at_10"] == 1.0
    delta = paired_delta(candidate[::-1], baseline, n_boot=100)
    assert delta["candidate"] == "candidate"
    assert delta["strata"]["full"]["recall_at_10_delta"] == 0.0
    assert delta["strata"]["full"]["map_delta"] > 0
    assert candidate[0]["schema_version"] == 2
    assert candidate[0]["embedding_execution"] == {}


def test_load_identity_table_validates_flag_nesting(tmp_path: Path) -> None:
    path = tmp_path / "identity.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_hash", "max_identity", "exact", "ge_90", "ge_30"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_hash": "q1",
                "max_identity": 0.5,
                "exact": 0,
                "ge_90": 0,
                "ge_30": 1,
            }
        )
    loaded = load_identity_table(path, enforce_counts=False)
    assert loaded["q1"].max_identity == 0.5


def test_score_directory_validates_historical_full_metrics(tmp_path: Path) -> None:
    sequences = ["AAAA", "AAAC", "KKKK", "KKKL"]
    labels = np.asarray(["a", "a", "b", "b"])
    records = build_records(
        model_tag="candidate",
        model_name="/candidate",
        domain_ids=["d1", "d2", "d3", "d4"],
        sequences=sequences,
        labels=labels,
        embeddings=np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]]),
        identities=_identities(sequences),
        embedding_execution={"batch_size": 1},
    )
    per_query = tmp_path / "per_query"
    per_query.mkdir()
    with (per_query / "candidate.jsonl").open("w") as handle:
        for record in records * (2207 // 4):
            handle.write(json.dumps(record) + "\n")
        for record in records[: 2207 % 4]:
            handle.write(json.dumps(record) + "\n")
    # Repeated hashes are acceptable to the scorer but make this a deliberately
    # synthetic full-metric validation fixture.
    _, _, validation = score_directory(
        tmp_path,
        [],
        n_boot=10,
        expected_full={"candidate": (1.0, 1.0)},
    )
    assert validation["passed"]
    _, _, mismatch = score_directory(
        tmp_path,
        [],
        n_boot=10,
        expected_full={"candidate": (0.0, 1.0)},
    )
    assert not mismatch["passed"]
    assert parse_expected_full(["candidate=1.0,0.5"]) == {"candidate": (1.0, 0.5)}
