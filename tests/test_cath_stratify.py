"""CATH per-query persistence and corpus-identity stratification tests."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from cath_levels import score_levels
from cath_stratify import (
    CATH_LEVELS,
    CathIdentity,
    build_prediction_records,
    cath_query_hash,
    load_identity_table,
    read_prediction_jsonl,
    score_identity_strata,
    sequence_sha256,
    write_prediction_jsonl,
)


def _labels(values: list[str]) -> dict[str, list[str]]:
    """Expand dotted H labels into their C/A/T/H prefixes."""

    output = {level: [] for level in CATH_LEVELS}
    for value in values:
        parts = value.split(".")
        for index, level in enumerate(CATH_LEVELS, start=1):
            output[level].append(".".join(parts[:index]))
    return output


def _prediction_fixture() -> tuple[
    list[dict], dict[str, CathIdentity], dict, dict, np.ndarray
]:
    sequences = ["ACDE", "FGHI", "KLMN", "PQRS"]
    query_ids = [f"domain{index}" for index in range(4)]
    lookup_ids = ["lookup0", "lookup1"]
    lookup_labels = _labels(["1.10.8.10", "2.20.9.20"])
    query_labels = _labels(["1.10.8.10", "2.20.9.20", "2.20.9.20", "1.10.8.10"])
    nearest = np.asarray([0, 0, 1, 1])
    identities = {}
    for index, (domain_id, sequence) in enumerate(zip(query_ids, sequences)):
        query_hash = cath_query_hash(sequence)
        identities[query_hash] = CathIdentity(
            query_hash=query_hash,
            domain_id=domain_id,
            sequence_sha256=sequence_sha256(sequence),
            in_test300=True,
            in_test219=True,
            in_test_h=True,
            max_identity=(1.0, 1.0, 0.95, 0.80)[index],
            ge_90=index < 3,
            exact=index < 2,
        )
    records = build_prediction_records(
        model_tag="fixture",
        model_name="/models/fixture",
        query_ids=query_ids,
        query_sequences=sequences,
        query_labels=query_labels,
        lookup_ids=lookup_ids,
        lookup_labels=lookup_labels,
        nearest_indices=nearest,
        identities=identities,
    )
    return records, identities, query_labels, lookup_labels, nearest


def test_per_query_full_metrics_reproduce_and_strata_rescore() -> None:
    """Persisted correctness exactly reproduces the unstratified C/A/T/H result."""

    records, identities, query_labels, lookup_labels, nearest = _prediction_fixture()
    baseline = score_levels(query_labels, lookup_labels, nearest, n_boot=100)
    stratified = score_identity_strata(records, identities=identities, n_boot=100)

    for level in CATH_LEVELS:
        full = stratified["levels"][level]["full"]
        assert full["n"] == baseline[level]["n_answerable"] == 4
        assert full["accuracy"] == baseline[level]["accuracy"] == 0.5
    h_level = stratified["levels"]["cath_h"]
    assert h_level["exact"]["n"] == 2
    assert h_level["non_exact"]["n"] == 2
    assert h_level["ge_90"]["accuracy"] == pytest.approx(2 / 3)
    assert h_level["lt_90"]["accuracy"] == 0.0
    assert h_level["lt_90"]["diagnostic"] is True


def test_prediction_jsonl_is_deterministic_and_self_contained(tmp_path: Path) -> None:
    """JSONL can be rescored without embeddings or a separate identity table."""

    records, identities, _, _, _ = _prediction_fixture()
    path = tmp_path / "predictions.jsonl"
    write_prediction_jsonl(path, records)
    first = path.read_bytes()
    write_prediction_jsonl(path, records)
    assert path.read_bytes() == first

    loaded = read_prediction_jsonl(path)
    embedded = score_identity_strata(loaded, n_boot=100)
    external = score_identity_strata(loaded, identities=identities, n_boot=100)
    assert embedded == external


def test_h_membership_must_match_answerable_mask() -> None:
    """The 150-row H subset cannot silently drift from EAT's masking rule."""

    records, identities, _, _, _ = _prediction_fixture()
    first_hash = records[0]["query_hash"]
    first = identities[first_hash]
    identities[first_hash] = CathIdentity(**{**first.__dict__, "in_test_h": False})

    with pytest.raises(ValueError, match="H-answerable membership mismatch"):
        _prediction_fixture_with_identities(identities)


def _prediction_fixture_with_identities(
    identities: dict[str, CathIdentity],
) -> list[dict]:
    """Rebuild the synthetic predictions with caller-supplied identity rows."""

    sequences = ["ACDE", "FGHI", "KLMN", "PQRS"]
    return build_prediction_records(
        model_tag="fixture",
        model_name="/models/fixture",
        query_ids=[f"domain{index}" for index in range(4)],
        query_sequences=sequences,
        query_labels=_labels(["1.10.8.10", "2.20.9.20", "2.20.9.20", "1.10.8.10"]),
        lookup_ids=["lookup0", "lookup1"],
        lookup_labels=_labels(["1.10.8.10", "2.20.9.20"]),
        nearest_indices=np.asarray([0, 0, 1, 1]),
        identities=identities,
    )


def test_identity_table_validates_flags_and_split_relationships(tmp_path: Path) -> None:
    """The audit table cannot carry inconsistent threshold or split flags."""

    sequence = "ACDE"
    path = tmp_path / "identity.tsv"
    header = [
        "query_hash",
        "domain_id",
        "sequence_sha256",
        "in_test300",
        "in_test219",
        "in_test_h",
        "max_identity",
        "ge_90",
        "exact",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerow(
            [
                cath_query_hash(sequence),
                "domain0",
                sequence_sha256(sequence),
                1,
                0,
                1,
                0.95,
                0,
                0,
            ]
        )
    with pytest.raises(ValueError, match="inconsistent ge_90 flag"):
        load_identity_table(path, enforce_expected_counts=False)
