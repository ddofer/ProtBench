#!/usr/bin/env python
"""Full C/A/T/H accuracy table on the CATH midnight-zone benchmark.

The `cath_eat` task in the suite reports the H (homologous superfamily) level
only. That is the headline number, but the paper's argument is the *shape* of
the C -> A -> T -> H progression: embeddings hold up at coarse levels where
alignment collapses. This produces the whole row, so our models can be set
beside Heinzinger et al. 2022 Table 1:

    Method              C     A     T     H
    MMseqs2             52    36    29    35
    HMMER (profiles)    70    60    59    77
    ProtT5 (raw)        84    67    57    64
    ProtTucker(ProtT5)  89    75    64    76

Method, matching EAT: mean-pooled per-protein embeddings, 1 nearest neighbour by
Euclidean distance in the lookup set, transfer its label.

Two details that look like they need code and do not:

* **The cascade is free.** The paper counts a hit at H only if C, A and T were
  also right. Labels here are dotted *prefix* strings (`cath_a` is "c.a",
  `cath_h` is "c.a.t.h"), so exact match at a level already implies every
  coarser level matched. Plain equality gives the paper's rule.
* **Masking is per level.** A query whose label at some level appears nowhere in
  the lookup set cannot be answered correctly by any method, so the paper drops
  it from that level rather than charging every method for an impossible case.
  Expected survivors: C 219, A 219, T 210, H 150 -- asserted, not assumed.

Bootstrap follows the paper (1,000 resamples, x1.96 for a 95% CI), not EAT's
code, whose `compute_err` defaults to 10,000 and returns a bare standard error
that the tables then multiply by hand.

Usage:
    python cath_levels.py                          # the two default ESM-2 arms
    python cath_levels.py --models base=/path/to/model ...
    python cath_levels.py --identity-table /path/to/cath_eat_query_identity.tsv
    python cath_levels.py --rescore-only --identity-table /path/to/table.tsv
    python cath_levels.py --merge-from /path/to/model_outputs ...
    python cath_levels.py --selfcheck              # no GPU, no network
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

from cath_stratify import (
    CATH_LEVELS,
    answerable_mask,
    bootstrap_ci,
    build_prediction_records,
    file_sha256,
    load_identity_table,
    read_prediction_jsonl,
    render_strata_markdown,
    score_identity_strata,
    write_prediction_jsonl,
)

logger = logging.getLogger("cath_levels")

DATASET = "GrimSqueaker/cath43-eat"
DATASET_REVISION = "26c3a3e28d559488f13dc0a85641b7c37329cf65"
LOOKUP_SPLIT = "lookup"
QUERY_SPLIT = "test219"
LEVELS = CATH_LEVELS

# From the paper's masking rule, reproduced independently from Rostlab/EAT's raw
# FASTAs. A mismatch means the dataset or the masking changed, not that a model
# did badly -- so it is a hard failure.
EXPECTED_ANSWERABLE = {"cath_c": 219, "cath_a": 219, "cath_t": 210, "cath_h": 150}
# Same queries under the stricter "label must be in the lookup set" reading.
EXPECTED_ANSWERABLE_STRICT = {
    "cath_c": 219,
    "cath_a": 219,
    "cath_t": 208,
    "cath_h": 150,
}

# Heinzinger et al. 2022, Table 1, accuracy on test219 -> lookup69k.
PAPER = {
    "Random": (29, 9, 1, 0),
    "MMseqs2 (sequence)": (52, 36, 29, 35),
    "HMMER (CATH-Gene3D profiles)": (70, 60, 59, 77),
    "ProtBERT (raw)": (67, 38, 22, 18),
    "ESM-1b (raw)": (79, 61, 50, 57),
    "ProtT5 (raw)": (84, 67, 57, 64),
    "ProtTucker(ESM-1b)": (87, 68, 59, 70),
    "ProtTucker(ProtT5)": (89, 75, 64, 76),
}

# No default arms: this script scores whatever models --models names, hub ids
# or local paths alike. A couple of small public hub models so `python
# cath_levels.py` with no arguments does something rather than nothing.
DEFAULT_ARMS = [
    ("esm2_8m", "facebook/esm2_t6_8M_UR50D"),
    ("esm2_650m", "facebook/esm2_t33_650M_UR50D"),
]


def nearest_neighbour(
    query: np.ndarray, lookup: np.ndarray, chunk: int = 64
) -> np.ndarray:
    """Index into `lookup` of the Euclidean-nearest row for each row of `query`.

    ||q - l||^2 = ||q||^2 - 2 q.l + ||l||^2; the ||q||^2 term is constant within
    a row and drops out of the argmin, so only the cross term and ||l||^2 matter.
    Chunked over queries so the (n_query, n_lookup) matrix never lands whole.
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    lookup = np.ascontiguousarray(lookup, dtype=np.float32)
    lookup_sq = np.einsum("ij,ij->i", lookup, lookup)
    out = np.empty(len(query), dtype=np.int64)
    for i in range(0, len(query), chunk):
        block = query[i : i + chunk]
        scores = lookup_sq[None, :] - 2.0 * (block @ lookup.T)
        out[i : i + chunk] = np.argmin(scores, axis=1)
    return out


def score_levels(
    query_labels: dict[str, list[str]],
    lookup_labels: dict[str, list[str]],
    nn_idx: np.ndarray,
    n_boot: int = 1000,
) -> dict:
    """Per-level accuracy over answerable queries, with a bootstrap CI."""
    out = {}
    for level in LEVELS:
        truth = np.asarray(query_labels[level])
        pool = np.asarray(lookup_labels[level])
        predicted = pool[nn_idx]

        # The paper's rule (EAT's `mask_singletons`): a query is dropped at a
        # level when it is the ONLY protein carrying that label across
        # lookup UNION test -- i.e. the label's count there is 1, the query
        # itself. Reproduces their denominators exactly: 219 / 219 / 210 / 150.
        answerable = answerable_mask(truth.tolist(), pool.tolist())

        # Stricter reading: the label must actually be IN the lookup set, since
        # a label shared only with another query still cannot be transferred.
        # Identical at C, A and H; two queries apart at T (208 vs 210). Reported
        # so the choice is visible, but the paper's rule is what we score on, and
        # it charges every method equally for those two.
        in_lookup = set(pool.tolist())
        n_strict = int(sum(1 for t in truth if t in in_lookup))

        correct = (predicted[answerable] == truth[answerable]).astype(np.float64)
        out[level] = {
            "accuracy": float(correct.mean()) if len(correct) else float("nan"),
            "ci95": bootstrap_ci(correct, n_boot=n_boot),
            "n_answerable": int(answerable.sum()),
            "n_answerable_strict": n_strict,
            "n_total": len(truth),
        }
    return out


def load_splits():
    from datasets import load_dataset

    ds = load_dataset(DATASET, revision=DATASET_REVISION)
    lookup, query = ds[LOOKUP_SPLIT], ds[QUERY_SPLIT]
    lookup_labels = {lv: list(lookup[lv]) for lv in LEVELS}
    query_labels = {lv: list(query[lv]) for lv in LEVELS}
    return (
        list(lookup["id"]),
        list(query["id"]),
        list(lookup["sequence"]),
        list(query["sequence"]),
        lookup_labels,
        query_labels,
    )


def embed_arm(
    model_name: str, seqs_lookup, seqs_query, batch_size: int, max_length: int
):
    """Embed both splits, reusing the suite's on-disk cache when it hits.

    The cache key is a hash of the exact sequence list plus an embed-config
    string, so a miss costs a re-embed and a hit is provably the same input --
    there is no way to be handed another model's vectors.
    """
    from protein_benchmark_suite import (
        _model_cache_namespace,
        embed_sequences,
        load_model,
    )
    from seq_embed_cache import cached_embed_sequences

    cache_root = str(
        Path("embed_cache") / _model_cache_namespace(model_name) / "seq_cache"
    )
    # Must match the suite's _cfg_key exactly or the Phase 1 lookup cache misses. The suite
    # interpolates its amp_dtype variable directly, and --amp_dtype fp32 (the default) leaves that
    # variable as None rather than a torch dtype -- so the token it writes is "dt=None", not
    # "dt=None". Writing fp32 here missed the cache on every run and silently re-embedded all
    # 69,605 CATH lookup sequences.
    cfg_key = f"trunk|l2=0|ml={max_length}|dt=None"

    obj, is_sbert, device = load_model(model_name, device="cuda")

    def _embed(seqs, _model=obj):
        return embed_sequences(
            _model, is_sbert, seqs, device, batch_size=batch_size, max_length=max_length
        )

    x_lookup = cached_embed_sequences(
        lambda: _embed(seqs_lookup), seqs_lookup, cache_root=cache_root, cfg_key=cfg_key
    )
    x_query = cached_embed_sequences(
        lambda: _embed(seqs_query), seqs_query, cache_root=cache_root, cfg_key=cfg_key
    )
    del obj
    try:
        import torch
    except ImportError:
        logger.debug("torch unavailable while clearing the embedding cache")
    else:
        torch.cuda.empty_cache()
    return np.asarray(x_lookup), np.asarray(x_query)


def render_markdown(results: dict) -> str:
    lines = [
        "# CATH v4.3 midnight-zone annotation transfer (C/A/T/H)",
        "",
        "1-NN Euclidean over mean-pooled per-protein embeddings, lookup69k -> test219,",
        "scored per CATH level over the queries answerable at that level.",
        "Errors are 95% CIs (1.96 x bootstrap SE, 1,000 resamples).",
        "",
        "## Our models",
        "",
        "| Model | C | A | T | H |",
        "|---|---|---|---|---|",
    ]
    for tag, res in results.items():
        cells = []
        for lv in LEVELS:
            r = res["levels"][lv]
            cells.append(f"{100 * r['accuracy']:.1f} ± {100 * r['ci95']:.1f}")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")

    n = next(iter(results.values()))["levels"] if results else None
    if n:
        counts = " / ".join(
            f"{lv[-1].upper()} {n[lv]['n_answerable']}" for lv in LEVELS
        )
        lines += ["", f"Answerable queries per level: {counts} (of 219).", ""]

    lines += [
        "## Heinzinger et al. 2022, Table 1",
        "",
        "Same splits and same scoring, but their models and their embedding code.",
        "A row here is NOT a like-for-like comparison against a row above; the",
        "like-for-like comparison is each ProtSent arm against its own frozen base.",
        "",
        "| Method | C | A | T | H |",
        "|---|---|---|---|---|",
    ]
    for name, (c, a, t, h) in PAPER.items():
        lines.append(f"| {name} | {c} | {a} | {t} | {h} |")
    return "\n".join(lines) + "\n"


def selfcheck() -> int:
    """Synthetic data whose 1-NN answer is known by construction."""
    # Four lookup points on the axes; queries placed nearest a chosen one.
    lookup = np.eye(4, dtype=np.float32) * 10.0
    query = np.array([[9.0, 0, 0, 0], [0, 9.0, 0, 0], [0, 0, 0, 9.0]], dtype=np.float32)
    nn = nearest_neighbour(query, lookup, chunk=2)
    assert nn.tolist() == [0, 1, 3], nn.tolist()

    # Chunking must not change the answer.
    for chunk in (1, 2, 3, 100):
        assert nearest_neighbour(query, lookup, chunk=chunk).tolist() == [0, 1, 3]

    # Scoring: lookup holds labels 1.10.8.10, 2.20.9.20, 3.30.7.30, 4.40.6.40.
    def lab(dotted):
        p = dotted.split(".")
        return {
            "cath_c": p[0],
            "cath_a": ".".join(p[:2]),
            "cath_t": ".".join(p[:3]),
            "cath_h": dotted,
        }

    def stack(dots):
        rows = [lab(d) for d in dots]
        return {lv: [r[lv] for r in rows] for lv in LEVELS}

    lookup_labels = stack(["1.10.8.10", "2.20.9.20", "3.30.7.30", "4.40.6.40"])
    # q0 -> lookup0 (exact, right at every level)
    # q1 -> lookup1 but truth is 2.20.9.99: right at C/A/T, wrong at H
    # q2 -> lookup3 but truth is 4.40.6.40: right everywhere
    query_labels = stack(["1.10.8.10", "2.20.9.99", "4.40.6.40"])
    res = score_levels(query_labels, lookup_labels, nn, n_boot=200)

    # 2.20.9.99 is a singleton across lookup+test, so q1 is masked at H only.
    assert res["cath_h"]["n_answerable"] == 2, res["cath_h"]
    assert res["cath_t"]["n_answerable"] == 3, res["cath_t"]

    # The paper's rule counts a label shared by two QUERIES as answerable even
    # when it is absent from the lookup set; the strict reading does not. This is
    # the entire 210-vs-208 gap at T, so pin both.
    pair = stack(["5.50.5.50", "5.50.5.50"])
    pair_res = score_levels(pair, lookup_labels, np.array([0, 0]), n_boot=200)
    assert pair_res["cath_h"]["n_answerable"] == 2, pair_res["cath_h"]
    assert pair_res["cath_h"]["n_answerable_strict"] == 0, pair_res["cath_h"]
    # Neither can be answered from the lookup set, so both are scored wrong.
    assert pair_res["cath_h"]["accuracy"] == 0.0, pair_res["cath_h"]
    assert res["cath_h"]["accuracy"] == 1.0, res["cath_h"]
    assert res["cath_t"]["accuracy"] == 1.0, res["cath_t"]
    assert res["cath_c"]["accuracy"] == 1.0, res["cath_c"]

    # A wrong prediction at a coarse level must propagate: prefix labels mean an
    # H match implies C/A/T matched, so a C miss cannot coexist with an H hit.
    wrong = stack(["9.99.9.99", "2.20.9.20", "4.40.6.40"])
    wrong_lookup = stack(["9.99.9.99", "2.20.9.20", "3.30.7.30", "4.40.6.40"])
    r2 = score_levels(wrong, wrong_lookup, np.array([1, 1, 3]), n_boot=200)
    assert r2["cath_c"]["accuracy"] == 2 / 3, r2["cath_c"]
    assert r2["cath_h"]["accuracy"] == 2 / 3, r2["cath_h"]

    # CI is a non-negative, finite half-width; a unanimous vector has zero spread.
    assert bootstrap_ci(np.ones(50), n_boot=200) == 0.0
    assert 0.0 < bootstrap_ci(np.array([0.0, 1.0] * 25), n_boot=200) < 0.5

    print("selfcheck OK")
    return 0


def _prediction_path(out_dir: Path, tag: str) -> Path:
    """Return a filesystem-safe, deterministic prediction path for ``tag``."""

    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("._")
    if not safe_tag:
        raise ValueError(f"model tag has no filesystem-safe characters: {tag!r}")
    return out_dir / "per_query" / f"{safe_tag}.jsonl"


def _write_strata_outputs(
    out_dir: Path,
    results: dict,
    identity_table: Path,
) -> None:
    """Persist stratified JSON and Markdown with identity-table provenance."""

    payload = {
        "schema_version": 1,
        "dataset": {"id": DATASET, "revision": DATASET_REVISION, "split": QUERY_SPLIT},
        "identity_table": str(identity_table),
        "identity_table_sha256": file_sha256(identity_table),
        "interpretation": (
            "Existing checkpoints are CATH-corpus-contaminated; strata are diagnostic only."
        ),
        "models": results,
    }
    (out_dir / "cath_strata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "CATH_STRATA.md").write_text(render_strata_markdown(results))


def _rescore_predictions(out_dir: Path, identity_table: Path, n_boot: int) -> int:
    """Rescore persisted predictions without loading a model or embeddings."""

    identities = load_identity_table(identity_table)
    prediction_paths = sorted((out_dir / "per_query").glob("*.jsonl"))
    if not prediction_paths:
        raise SystemExit(f"no prediction JSONL files under {out_dir / 'per_query'}")
    results = {}
    for path in prediction_paths:
        records = read_prediction_jsonl(path)
        if len(records) != 219:
            raise SystemExit(
                f"{path}: expected 219 per-query rows, found {len(records)}"
            )
        score = score_identity_strata(records, identities=identities, n_boot=n_boot)
        tag = score["model_tag"]
        if tag in results:
            raise SystemExit(f"duplicate model tag in persisted predictions: {tag}")
        results[tag] = score
        logger.info("rescored %s from %s", tag, path)
    _write_strata_outputs(out_dir, results, identity_table)
    logger.info(
        "wrote %s and %s", out_dir / "cath_strata.json", out_dir / "CATH_STRATA.md"
    )
    return 0


def _merge_model_outputs(
    out_dir: Path,
    input_dirs: list[Path],
    identity_table: Path,
    n_boot: int,
    *,
    expected_predictions: int = 219,
) -> int:
    """Merge independently run model directories without recomputing embeddings."""

    identities = load_identity_table(identity_table)
    results = {}
    strata_results = {}
    failures = {}
    for input_dir in input_dirs:
        levels_path = input_dir / "cath_levels.json"
        if not levels_path.exists():
            raise SystemExit(f"missing {levels_path}")
        input_results = json.loads(levels_path.read_text())
        failure_path = input_dir / "cath_failures.json"
        if failure_path.exists():
            input_failures = json.loads(failure_path.read_text())
            overlap = failures.keys() & input_failures.keys()
            if overlap:
                raise SystemExit(f"duplicate failed model tags: {sorted(overlap)}")
            failures.update(input_failures)

        for tag, result in input_results.items():
            if tag in results:
                raise SystemExit(f"duplicate successful model tag: {tag}")
            prediction_path = Path(result["per_query_predictions"])
            if not prediction_path.is_absolute():
                prediction_path = input_dir / prediction_path
            records = read_prediction_jsonl(prediction_path)
            if len(records) != expected_predictions:
                raise SystemExit(
                    f"{prediction_path}: expected {expected_predictions} per-query rows, "
                    f"found {len(records)}"
                )
            central_prediction_path = _prediction_path(out_dir, tag)
            write_prediction_jsonl(central_prediction_path, records)
            result["per_query_predictions"] = str(central_prediction_path)
            strata = score_identity_strata(
                records,
                identities=identities,
                n_boot=n_boot,
            )
            for level in LEVELS:
                full = strata["levels"][level]["full"]
                expected = result["levels"][level]
                if full["n"] != expected["n_answerable"] or not np.isclose(
                    full["accuracy"], expected["accuracy"]
                ):
                    raise SystemExit(f"merged full metric mismatch for {tag}/{level}")
            results[tag] = result
            strata_results[tag] = strata

    (out_dir / "cath_levels.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "CATH_LEVELS.md").write_text(render_markdown(results))
    (out_dir / "cath_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n"
    )
    _write_strata_outputs(out_dir, strata_results, identity_table)
    logger.info(
        "merged %d successful models and %d failures into %s",
        len(results),
        len(failures),
        out_dir,
    )
    return int(bool(failures))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="+",
        metavar="TAG=PATH",
        help="Arms to score, as tag=hub_id or tag=/local/path; defaults to two small "
        "public ESM2 arms so a bare run does something.",
    )
    ap.add_argument("--out", default="results/cath_eat")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument(
        "--identity-table",
        "--identity_table",
        type=Path,
        default=Path(os.environ["CATH_IDENTITY_TABLE"])
        if os.environ.get("CATH_IDENTITY_TABLE")
        else None,
        help="Proteva cath_eat_query_identity.tsv. Enables exact/90% stratification.",
    )
    ap.add_argument(
        "--rescore-only",
        action="store_true",
        help="Rescore existing OUT/per_query/*.jsonl; never load a model or embeddings.",
    )
    ap.add_argument(
        "--merge-from",
        nargs="+",
        type=Path,
        help="Merge per-model output directories and rescore their JSONL; no embedding.",
    )
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.selfcheck:
        return selfcheck()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_from:
        if args.rescore_only or args.models:
            raise SystemExit(
                "--merge-from cannot be combined with --rescore-only or --models"
            )
        if args.identity_table is None:
            raise SystemExit("--merge-from requires --identity-table")
        return _merge_model_outputs(
            out_dir,
            args.merge_from,
            args.identity_table,
            args.n_boot,
        )
    if args.rescore_only:
        if args.identity_table is None:
            raise SystemExit("--rescore-only requires --identity-table")
        return _rescore_predictions(out_dir, args.identity_table, args.n_boot)

    arms = DEFAULT_ARMS
    if args.models:
        arms = [tuple(m.split("=", 1)) for m in args.models]

    lookup_ids, query_ids, seqs_lookup, seqs_query, lookup_labels, query_labels = (
        load_splits()
    )
    logger.info("lookup %d, queries %d", len(seqs_lookup), len(seqs_query))

    identities = (
        load_identity_table(args.identity_table) if args.identity_table else None
    )
    results = {}
    strata_results = {}
    failures = {}

    for tag, model_name in arms:
        logger.info("=== %s (%s)", tag, model_name)
        try:
            x_lookup, x_query = embed_arm(
                model_name, seqs_lookup, seqs_query, args.batch_size, args.max_length
            )
        except Exception as exc:  # noqa: BLE001  # one bad arm must not lose the others
            logger.error("%s FAILED: %s: %s", tag, type(exc).__name__, exc)
            failures[tag] = {
                "model": model_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            continue

        nn = nearest_neighbour(x_query, x_lookup)
        levels = score_levels(query_labels, lookup_labels, nn, n_boot=args.n_boot)

        got = {lv: levels[lv]["n_answerable"] for lv in LEVELS}
        if got != EXPECTED_ANSWERABLE:
            raise SystemExit(
                f"answerable counts {got} != expected {EXPECTED_ANSWERABLE}; "
                "the masking or the dataset changed -- refusing to report."
            )

        records = build_prediction_records(
            model_tag=tag,
            model_name=model_name,
            query_ids=query_ids,
            query_sequences=seqs_query,
            query_labels=query_labels,
            lookup_ids=lookup_ids,
            lookup_labels=lookup_labels,
            nearest_indices=nn,
            identities=identities,
        )
        prediction_path = _prediction_path(out_dir, tag)
        write_prediction_jsonl(prediction_path, records)

        results[tag] = {
            "model": model_name,
            "dim": int(x_lookup.shape[1]),
            "levels": levels,
            "per_query_predictions": str(prediction_path),
        }
        if identities is not None:
            strata = score_identity_strata(
                records, identities=identities, n_boot=args.n_boot
            )
            for level in LEVELS:
                full = strata["levels"][level]["full"]
                if full["n"] != levels[level]["n_answerable"] or not np.isclose(
                    full["accuracy"], levels[level]["accuracy"]
                ):
                    raise RuntimeError(
                        f"per-query full metric failed to reproduce {tag}/{level}"
                    )
            strata_results[tag] = strata
        logger.info(
            "%s  C %.1f  A %.1f  T %.1f  H %.1f",
            tag,
            *[100 * levels[lv]["accuracy"] for lv in LEVELS],
        )

    (out_dir / "cath_levels.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "CATH_LEVELS.md").write_text(render_markdown(results))
    (out_dir / "cath_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n"
    )
    logger.info(
        "wrote %s and %s", out_dir / "cath_levels.json", out_dir / "CATH_LEVELS.md"
    )
    if strata_results and args.identity_table is not None:
        _write_strata_outputs(out_dir, strata_results, args.identity_table)
        logger.info(
            "wrote %s and %s", out_dir / "cath_strata.json", out_dir / "CATH_STRATA.md"
        )

    # Cross-check against the suite's own cath_eat run: two independent code
    # paths must agree on H, or one of them is wrong.
    for tag, res in results.items():
        csv = sorted((out_dir / tag).glob("*.csv")) if (out_dir / tag).is_dir() else []
        if not csv:
            continue
        import pandas as pd

        d = pd.read_csv(csv[0])
        if "Error" in d.columns:
            d = d[d["Error"].isna()]
        if not len(d):
            continue
        suite_h = float(d.iloc[-1]["Accuracy"])
        ours_h = res["levels"]["cath_h"]["accuracy"]
        # The suite rounds metrics to 5 decimals on write, so compare at that
        # precision -- a 1e-6 tolerance flags 0.40667 vs 0.4066666... as a
        # mismatch when the two paths in fact agree exactly.
        flag = "OK" if round(suite_h, 5) == round(ours_h, 5) else "MISMATCH"
        logger.info(
            "%s  H cross-check: suite %.4f vs levels %.4f  %s",
            tag,
            suite_h,
            ours_h,
            flag,
        )

    return int(bool(failures))


if __name__ == "__main__":
    os.environ.setdefault("PROTEIN_BENCH_ATTN_IMPLEMENTATION", "sdpa")
    sys.exit(main())
