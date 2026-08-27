"""SCOPe-40 retrieval stratified by pretraining-corpus sequence identity.

Frozen embeddings are scored all-vs-all with the canonical cosine ranking. One
JSONL row per query stores Recall@10 and average precision, so every stratum and
paired model delta can be recomputed without loading a model again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bootstrap_ci import boot_ci, per_query_metrics
from protein_benchmark_suite import (
    _model_cache_namespace,
    cosine_ranking,
    embed_sequences,
    load_model,
)
from seq_embed_cache import cached_embed_sequences

LOGGER = logging.getLogger("scope_stratify")
DATASET = "tattabio/scope40_test"
DATASET_REVISION = "0c9e085c1883336839ecdfbac788fadb68218087"
EXPECTED_COUNTS = {"queries": 2207, "exact": 1225, "ge_90": 2028, "ge_30": 2206}


@dataclass(frozen=True)
class Identity:
    query_hash: str
    max_identity: float
    exact: bool
    ge_90: bool
    ge_30: bool


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256("".join(sequence.split()).upper().encode()).hexdigest()


def load_identity_table(path: Path, *, enforce_counts: bool = True) -> dict[str, Identity]:
    identities = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query_hash = row["query_hash"]
            identity = Identity(
                query_hash=query_hash,
                max_identity=float(row["max_identity"]),
                exact=row["exact"] == "1",
                ge_90=row["ge_90"] == "1",
                ge_30=row["ge_30"] == "1",
            )
            if query_hash in identities:
                raise ValueError(f"duplicate SCOPe identity {query_hash}")
            if identity.exact and not identity.ge_90:
                raise ValueError(f"inconsistent exact/ge90 flags for {query_hash}")
            if identity.ge_90 and not identity.ge_30:
                raise ValueError(f"inconsistent ge90/ge30 flags for {query_hash}")
            identities[query_hash] = identity
    counts = {
        "queries": len(identities),
        "exact": sum(identity.exact for identity in identities.values()),
        "ge_90": sum(identity.ge_90 for identity in identities.values()),
        "ge_30": sum(identity.ge_30 for identity in identities.values()),
    }
    if enforce_counts and counts != EXPECTED_COUNTS:
        raise ValueError(f"SCOPe identity counts {counts} != {EXPECTED_COUNTS}")
    return identities


def load_scope():
    from datasets import load_dataset

    dataset = load_dataset(DATASET, revision=DATASET_REVISION, split="train")
    return (
        [str(value) for value in dataset["id"]],
        [str(value) for value in dataset["sequence"]],
        np.asarray(dataset["family"], dtype=str),
    )


def embed_arm(
    model_name: str,
    sequences: list[str],
    batch_size: int,
    max_length: int,
    cache_dir: Path | None = None,
    proteva_flash_off_mode: str = "dense",
) -> tuple[np.ndarray, dict[str, Any]]:
    model_obj, is_sbert, device = load_model(model_name, device="cuda")
    hf_model = model_obj[1] if isinstance(model_obj, tuple) else model_obj
    encoder_config = getattr(getattr(hf_model, "encoder", None), "config", None)
    execution = {
        "batch_size": batch_size,
        "max_length": max_length,
        "pooling": "mean",
        "l2_normalize": False,
        "device": str(device),
        "flash_attn_mode": getattr(encoder_config, "flash_attn_mode", None),
        "proteva_flash_off_mode": proteva_flash_off_mode,
    }
    cache_root = None
    if cache_dir is not None:
        cache_root = str(cache_dir / _model_cache_namespace(model_name) / "seq_cache")
    # Batch size is intentionally part of the key. Mathematically equivalent
    # padded and singleton inference can differ at the last few FP32 bits, which
    # is enough to flip a near-tied all-vs-all retrieval rank.
    cfg_key = json.dumps(execution, sort_keys=True)
    model_holder = [model_obj]

    def compute():
        return embed_sequences(
            model_holder[0],
            is_sbert,
            sequences,
            device,
            batch_size=batch_size,
            max_length=max_length,
            proteva_flash_off_mode=proteva_flash_off_mode,
        )

    embeddings = cached_embed_sequences(
        compute,
        sequences,
        cache_root=cache_root,
        cfg_key=cfg_key,
    )
    model_holder.clear()
    del model_obj
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.cuda.empty_cache()
    return np.asarray(embeddings), execution


def build_records(
    *,
    model_tag: str,
    model_name: str,
    domain_ids: list[str],
    sequences: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    identities: dict[str, Identity],
    embedding_execution: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ranking = cosine_ranking(embeddings)
    metrics = per_query_metrics(ranking, labels)
    records = []
    for index, (domain_id, sequence, label) in enumerate(
        zip(domain_ids, sequences, labels, strict=True)
    ):
        query_hash = sequence_hash(sequence)
        identity = identities.get(query_hash)
        if identity is None:
            raise ValueError(f"SCOPe query {domain_id}/{query_hash} missing identity")
        records.append(
            {
                "schema_version": 2,
                "model_tag": model_tag,
                "model": model_name,
                "dataset": DATASET,
                "dataset_revision": DATASET_REVISION,
                "domain_id": domain_id,
                "query_hash": query_hash,
                "family": str(label),
                "embedding_execution": embedding_execution or {},
                "max_corpus_identity": identity.max_identity,
                "exact": identity.exact,
                "ge_90": identity.ge_90,
                "ge_30": identity.ge_30,
                "eligible": bool(metrics["eligible"][index]),
                "hit1": float(metrics["hit1"][index]),
                "hit10": float(metrics["hit10"][index]),
                "hit30": float(metrics["hit30"][index]),
                "average_precision": float(metrics["ap"][index]),
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _strata_mask(records: list[dict[str, Any]], stratum: str) -> np.ndarray:
    if stratum == "full":
        return np.ones(len(records), dtype=bool)
    field, expected = {
        "exact": ("exact", True),
        "non_exact": ("exact", False),
        "ge90": ("ge_90", True),
        "lt90": ("ge_90", False),
        "ge30": ("ge_30", True),
        "lt30": ("ge_30", False),
    }[stratum]
    return np.asarray([bool(record[field]) is expected for record in records])


STRATA = ("full", "exact", "non_exact", "ge90", "lt90", "ge30", "lt30")


def score_records(records: list[dict[str, Any]], n_boot: int = 10_000) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot score empty SCOPe records")
    result = {"model_tag": records[0]["model_tag"], "model": records[0]["model"], "strata": {}}
    for stratum in STRATA:
        mask = _strata_mask(records, stratum)
        hit10 = np.asarray([record["hit10"] for record in records], dtype=float)[mask]
        ap = np.asarray([record["average_precision"] for record in records], dtype=float)[mask]
        eligible = np.asarray([record["eligible"] for record in records], dtype=bool)[mask]
        recall_mean, recall_low, recall_high = boot_ci(hit10, n_boot=n_boot)
        map_mean, map_low, map_high = boot_ci(ap, n_boot=n_boot)
        result["strata"][stratum] = {
            "n": int(mask.sum()),
            "n_eligible": int(eligible.sum()),
            "recall_at_10": recall_mean,
            "recall_at_10_ci95": [recall_low, recall_high],
            "map": map_mean,
            "map_ci95": [map_low, map_high],
            "diagnostic": int(mask.sum()) < 20,
        }
    return result


def paired_delta(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    n_boot: int = 10_000,
) -> dict[str, Any]:
    candidate_map = {record["query_hash"]: record for record in candidate}
    baseline_map = {record["query_hash"]: record for record in baseline}
    if set(candidate_map) != set(baseline_map):
        raise ValueError("candidate/baseline SCOPe queries differ")
    ordered = sorted(candidate_map)
    candidate = [candidate_map[key] for key in ordered]
    baseline = [baseline_map[key] for key in ordered]
    result = {
        "candidate": candidate[0]["model_tag"],
        "baseline": baseline[0]["model_tag"],
        "strata": {},
    }
    for stratum in STRATA:
        mask = _strata_mask(candidate, stratum)
        cand_recall = np.asarray([record["hit10"] for record in candidate], dtype=float)[mask]
        base_recall = np.asarray([record["hit10"] for record in baseline], dtype=float)[mask]
        cand_ap = np.asarray(
            [record["average_precision"] for record in candidate], dtype=float
        )[mask]
        base_ap = np.asarray(
            [record["average_precision"] for record in baseline], dtype=float
        )[mask]
        recall = boot_ci(cand_recall - base_recall, n_boot=n_boot)
        map_delta = boot_ci(cand_ap - base_ap, n_boot=n_boot)
        result["strata"][stratum] = {
            "n": int(mask.sum()),
            "recall_at_10_delta": recall[0],
            "recall_at_10_delta_ci95": [recall[1], recall[2]],
            "map_delta": map_delta[0],
            "map_delta_ci95": [map_delta[1], map_delta[2]],
            "diagnostic": int(mask.sum()) < 20,
        }
    return result


def _safe_tag(tag: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("._")
    if not safe:
        raise ValueError(f"unsafe empty model tag from {tag!r}")
    return safe


def render_markdown(
    scores: dict[str, Any],
    comparisons: list[dict[str, Any]],
    validation: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# SCOPe-40 corpus-identity stratification",
        "",
        "All-vs-all cosine retrieval. Existing checkpoints are homology-exposed; strata",
        "are diagnostic and no SCOPe sequence was added to a training denylist.",
        "",
        "| Model | Stratum | n | Recall@10 (95% CI) | MAP (95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    if validation and validation["models"]:
        validation_items = sorted(validation["models"].items())
        first_validation = validation_items[0][1]
        lines[5:5] = [
            "## Historical full-metric validation",
            "",
            "| Model | Recall@10 observed / expected | MAP observed / expected | Pass |",
            "|---|---:|---:|:---:|",
            *[
                f"| {tag} | {item['observed_recall_at_10']:.6f} / "
                f"{item['expected_recall_at_10']:.6f} | {item['observed_map']:.6f} / "
                f"{item['expected_map']:.6f} | {'yes' if item['passed'] else 'NO'} |"
                for tag, item in validation_items
            ],
            "",
            (
                "Pass uses the configured absolute tolerances: "
                f"Recall@10 ≤ {first_validation['recall_at_10_tolerance']:.6f} "
                "(about one of 2,207 queries) and "
                f"MAP ≤ {first_validation['map_tolerance']:.6f}."
            ),
            "",
        ]
    for tag, score in sorted(scores.items()):
        for stratum in STRATA:
            item = score["strata"][stratum]
            flag = "*" if item["diagnostic"] else ""
            lines.append(
                f"| {tag} | {stratum}{flag} | {item['n']} | "
                f"{item['recall_at_10']:.4f} [{item['recall_at_10_ci95'][0]:.4f}, "
                f"{item['recall_at_10_ci95'][1]:.4f}] | {item['map']:.4f} "
                f"[{item['map_ci95'][0]:.4f}, {item['map_ci95'][1]:.4f}] |"
            )
    for comparison in comparisons:
        lines += [
            "",
            f"## Paired delta: {comparison['candidate']} - {comparison['baseline']}",
            "",
            "| Stratum | n | Δ Recall@10 (95% CI) | Δ MAP (95% CI) |",
            "|---|---:|---:|---:|",
        ]
        for stratum in STRATA:
            item = comparison["strata"][stratum]
            flag = "*" if item["diagnostic"] else ""
            lines.append(
                f"| {stratum}{flag} | {item['n']} | {item['recall_at_10_delta']:+.4f} "
                f"[{item['recall_at_10_delta_ci95'][0]:+.4f}, "
                f"{item['recall_at_10_delta_ci95'][1]:+.4f}] | "
                f"{item['map_delta']:+.4f} [{item['map_delta_ci95'][0]:+.4f}, "
                f"{item['map_delta_ci95'][1]:+.4f}] |"
            )
    lines += ["", "`*` n < 20; diagnostic only."]
    return "\n".join(lines) + "\n"


def score_directory(
    out_dir: Path,
    comparisons_requested: list[tuple[str, str]],
    n_boot: int,
    expected_full: dict[str, tuple[float, float]] | None = None,
    recall_tolerance: float = 0.5 / 2207,
    map_tolerance: float = 1e-4,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    records_by_tag = {}
    scores = {}
    for path in sorted((out_dir / "per_query").glob("*.jsonl")):
        records = read_jsonl(path)
        if len(records) != 2207:
            raise ValueError(f"{path}: expected 2207 records, got {len(records)}")
        tag = str(records[0]["model_tag"])
        records_by_tag[tag] = records
        scores[tag] = score_records(records, n_boot=n_boot)
    comparisons = []
    for candidate, baseline in comparisons_requested:
        if candidate not in records_by_tag or baseline not in records_by_tag:
            raise ValueError(f"comparison models missing: {candidate}={baseline}")
        comparisons.append(
            paired_delta(records_by_tag[candidate], records_by_tag[baseline], n_boot=n_boot)
        )
    validation: dict[str, Any] = {"passed": True, "models": {}}
    for tag, (expected_recall, expected_map) in (expected_full or {}).items():
        if tag not in scores:
            raise ValueError(f"expected full metrics model missing: {tag}")
        observed = scores[tag]["strata"]["full"]
        recall_delta = float(observed["recall_at_10"] - expected_recall)
        map_delta = float(observed["map"] - expected_map)
        passed = abs(recall_delta) <= recall_tolerance and abs(map_delta) <= map_tolerance
        validation["models"][tag] = {
            "expected_recall_at_10": expected_recall,
            "observed_recall_at_10": observed["recall_at_10"],
            "recall_at_10_delta": recall_delta,
            "recall_at_10_tolerance": recall_tolerance,
            "expected_map": expected_map,
            "observed_map": observed["map"],
            "map_delta": map_delta,
            "map_tolerance": map_tolerance,
            "passed": passed,
        }
        validation["passed"] = validation["passed"] and passed
    payload = {
        "schema_version": 1,
        "dataset": {"id": DATASET, "revision": DATASET_REVISION, "split": "train"},
        "interpretation": "homology-exposed diagnostic; not clean absolute performance",
        "models": scores,
        "comparisons": comparisons,
        "historical_full_metric_validation": validation,
    }
    (out_dir / "scope_strata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "SCOPE_STRATA.md").write_text(
        render_markdown(scores, comparisons, validation)
    )
    return scores, comparisons, validation


def parse_expected_full(values: list[str]) -> dict[str, tuple[float, float]]:
    expected = {}
    for value in values:
        tag, metrics = value.split("=", 1)
        recall, map_value = metrics.split(",", 1)
        if tag in expected:
            raise ValueError(f"duplicate expected-full tag: {tag}")
        expected[tag] = (float(recall), float(map_value))
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", metavar="TAG=PATH")
    parser.add_argument("--identity-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparison", action="append", default=[], metavar="CANDIDATE=BASE")
    parser.add_argument("--rescore-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--proteva-flash-off-mode",
        choices=("dense", "legacy_single_packed"),
        default="dense",
        help="use legacy packed singleton inference only for historical comparability",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        help="optional explicit content-addressed cache (disabled by default)",
    )
    parser.add_argument(
        "--expected-full",
        action="append",
        default=[],
        metavar="TAG=RECALL10,MAP",
        help="fail if full metrics do not reproduce a historical result",
    )
    parser.add_argument("--recall-tolerance", type=float, default=0.5 / 2207)
    parser.add_argument("--map-tolerance", type=float, default=1e-4)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    comparisons = [tuple(value.split("=", 1)) for value in args.comparison]
    args.out.mkdir(parents=True, exist_ok=True)
    identities = load_identity_table(args.identity_table)
    failures = {}
    if not args.rescore_only:
        if not args.models:
            raise SystemExit("--models is required unless --rescore-only")
        domain_ids, sequences, labels = load_scope()
        observed_hashes = {sequence_hash(sequence) for sequence in sequences}
        if observed_hashes != set(identities):
            raise SystemExit("pinned SCOPe dataset differs from identity table")
        for spec in args.models:
            tag, model_name = spec.split("=", 1)
            try:
                embeddings, execution = embed_arm(
                    model_name,
                    sequences,
                    args.batch_size,
                    args.max_length,
                    args.embedding_cache_dir,
                    args.proteva_flash_off_mode,
                )
                records = build_records(
                    model_tag=tag,
                    model_name=model_name,
                    domain_ids=domain_ids,
                    sequences=sequences,
                    labels=labels,
                    embeddings=embeddings,
                    identities=identities,
                    embedding_execution=execution,
                )
                write_jsonl(args.out / "per_query" / f"{_safe_tag(tag)}.jsonl", records)
            except Exception as exc:
                LOGGER.exception("%s failed", tag)
                failures[tag] = {"error_type": type(exc).__name__, "error": str(exc)}
    (args.out / "scope_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n"
    )
    _, _, validation = score_directory(
        args.out,
        comparisons,
        args.n_boot,
        expected_full=parse_expected_full(args.expected_full),
        recall_tolerance=args.recall_tolerance,
        map_tolerance=args.map_tolerance,
    )
    return int(bool(failures) or not validation["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
