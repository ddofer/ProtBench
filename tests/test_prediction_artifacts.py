from __future__ import annotations

from prediction_artifacts import (
    read_prediction_rows,
    reproduce_classification_metrics,
    write_sequence_predictions,
)
from scripts.stratify_prediction_artifacts import stratify


def test_sequence_predictions_are_deterministic_and_have_no_plain_sequences(
    tmp_path,
) -> None:
    path = tmp_path / "predictions.jsonl.gz"
    queries = tmp_path / "queries.fasta"
    write_sequence_predictions(
        path,
        sequences=["ASK", "MKK"],
        labels=[1, 0],
        predictions=[1, 1],
        scores=[0.9, 0.7],
        metadata={"task": "solubility", "problem_type": "binary"},
        query_fasta_path=queries,
    )
    metadata, rows = read_prediction_rows(path)
    assert metadata["task"] == "solubility"
    assert rows[0] == {
        "example_id": 0,
        "label": 1,
        "prediction": 1,
        "score": 0.9,
        "sequence_sha256": "6d6125cc4538aaec9dbef490ab1091a6cb4af5348f96a5cb0bfeeeda6edfebbe",
    }
    assert "sequence" not in rows[0]
    reproduced = reproduce_classification_metrics(path)
    assert reproduced["Accuracy"] == 0.5
    assert reproduced["AUC"] == 1.0
    assert "ASK" in queries.read_text()
    assert (
        "6d6125cc4538aaec9dbef490ab1091a6cb4af5348f96a5cb0bfeeeda6edfebbe"
        in queries.read_text()
    )
    identity = tmp_path / "identity.tsv"
    identity.write_text(
        "query_id\tstratum\n"
        "6d6125cc4538aaec9dbef490ab1091a6cb4af5348f96a5cb0bfeeeda6edfebbe\texact\n"
        "029c2d46a6c11e345b8d42343e8cdcefb07dd079b7dff423d91981b92ec922d6\t<30_or_no_hit\n"
    )
    report = stratify(path, identity)
    assert report["full"] == reproduced
    assert report["protein_counts"]["exact"] == 1
