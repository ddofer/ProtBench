#!/usr/bin/env python
"""HMMER (phmmer) alignment baseline over the benchmark, scored identically to MMseqs2.

phmmer builds an implicit profile from each single query sequence and searches
it against the target set, which makes it a more sensitive remote-homology
detector than MMseqs2's k-mer prefilter. It is the harder alignment baseline of
the two, which is why it is worth running rather than assuming it would lose.

**The scoring code is not duplicated here.** Both engines emit the same
structure -- `hits: {query_idx: [(target_idx, bitscore), ...]}` ranked best
first -- and both are scored by `mmseqs_baseline.score_task`. If the two
baselines were scored by two copies of the metric code, the comparison between
them would be worthless. `--selfcheck` asserts the property directly.

Conventions inherited from that shared path: self-matches excluded, **queries
with no hit count as failures and are never dropped**, per-class score = max
bitscore over that class's training sequences, 1-NN by bitscore for regression,
`hit_coverage` recorded per task.

Search direction matches MMseqs2: query = test sequences, target = train
sequences; for retrieval, all-vs-all on the gallery.

Usage:
    python hmmer_baseline.py --task scope40_retrieval   # one benchmark row
    python hmmer_baseline.py                            # legacy SCOPe-40 CI report
    python hmmer_baseline.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mmseqs_baseline import load_task, score_task  # noqa: E402  the shared scoring path

logger = logging.getLogger("hmmer_baseline")

# Parity with mmseqs_baseline.SEARCH_FLAGS: -e 10 reporting threshold, and at
# most 300 targets kept per query (MMseqs2's --max-seqs 300).
E_VALUE = 10.0
MAX_HITS = 300
NON_RESIDUE = re.compile(r"[^A-Z]")


def load_scope() -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    ds = load_dataset("tattabio/scope40_test", split="train")
    return list(ds["sequence"]), list(ds["family"])


def _digitize(seqs: list[str], alphabet):
    """Sequences -> digital, named by their index.

    Anything that is not a letter becomes X, which is how MMseqs2 treats the
    same characters (e.g. the `|` separator in peptide_hla inputs). Easel raises
    on them instead of coercing, so the coercion has to be explicit -- and it
    has to match MMseqs2 or the two baselines are not reading the same input.
    """
    from pyhmmer.easel import TextSequence

    return [
        TextSequence(
            sequence=NON_RESIDUE.sub("X", s.upper()) or "X", name=str(i).encode()
        ).digitize(alphabet)
        for i, s in enumerate(seqs)
    ]


def phmmer_hits(
    queries: list[str],
    targets: list[str],
    cpus: int = 48,
    E: float = E_VALUE,
    max_hits: int = MAX_HITS,
) -> dict[int, list[tuple[int, float]]]:
    """{query_idx: [(target_idx, bitscore), ...]}, bitscore descending.

    Same structure `mmseqs_baseline.read_hits` returns, so it feeds the same
    scorer. Queries phmmer reports nothing for are simply absent from the dict;
    the scorer counts them as failures.
    """
    import pyhmmer
    from pyhmmer.easel import Alphabet, DigitalSequenceBlock

    alphabet = Alphabet.amino()
    q_dig = _digitize(queries, alphabet)
    t_block = DigitalSequenceBlock(
        alphabet,
        q_dig if targets is queries else _digitize(targets, alphabet),
    )

    hits: dict[int, list[tuple[int, float]]] = {}
    for q, top in enumerate(pyhmmer.hmmer.phmmer(q_dig, t_block, cpus=cpus, E=E)):
        scored = [
            (int(h.name if isinstance(h.name, str) else h.name.decode()), float(h.score))
            for h in top
        ]
        if scored:
            scored.sort(key=lambda x: -x[1])
            hits[q] = scored[:max_hits]
    return hits


def phmmer_ranking(seqs: list[str], cpus: int = 32, E: float = E_VALUE) -> np.ndarray:
    """All-vs-all phmmer as a full gallery ordering, self removed.

    Only used by the bootstrap-CI report, which wants a dense ranking matrix.
    Queries with no hit keep an arbitrary tail, so their positives land beyond
    any K that is scored -- the "no hit is a failure" convention.
    """
    hits = phmmer_hits(seqs, seqs, cpus=cpus, E=E, max_hits=len(seqs))
    n = len(seqs)
    ranking = np.zeros((n, n - 1), dtype=int)
    n_no_hit = 0
    for q in range(n):
        ranked = [t for t, _ in hits.get(q, ()) if t != q]
        if not ranked:
            n_no_hit += 1
        seen = set(ranked)
        ranking[q] = np.array(
            ranked + [t for t in range(n) if t != q and t not in seen], dtype=int
        )[: n - 1]
    print(f"queries with no phmmer hit at all: {n_no_hit}/{n}")
    return ranking


def run_task(args) -> int:
    """One benchmark task -> one row in the HMMER baseline JSON."""
    cfg, (train_seqs, train_labels), (test_seqs, test_labels) = load_task(
        args.task, args.max_samples
    )
    logger.info(
        "%s [%s]: %d train / %d test sequences",
        cfg.name, cfg.problem_type, len(train_seqs), len(test_seqs),
    )

    # Same direction MMseqs2 searches: test queries against train targets, except
    # retrieval, which is all-vs-all over the single gallery.
    targets = test_seqs if cfg.problem_type == "retrieval" else train_seqs
    t0 = time.time()
    hits = phmmer_hits(test_seqs, targets, cpus=args.cpus)
    elapsed = time.time() - t0
    logger.info("%d/%d queries got >=1 hit in %.1fs", len(hits), len(test_seqs), elapsed)

    metrics = score_task(cfg, hits, train_labels, test_labels)

    row = {
        "Model": "HMMER phmmer (alignment baseline)",
        "Task": cfg.name,
        "task_key": args.task,
        "main_metric": cfg.main_metric,
        "problem_type": cfg.problem_type,
        "eval_split": "test",  # prepare_data(eval_split="test"); NOT the suite default
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "hmmer_flags": f"phmmer -E {E_VALUE} --max-hits {MAX_HITS}, rank by bitscore desc",
        "max_samples": args.max_samples,  # null = full task, no subsampling
        "search_seconds": round(elapsed, 1),
        **metrics,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else []
    existing = [r for r in existing if r.get("task_key") != args.task]
    existing.append(row)
    out.write_text(json.dumps(existing, indent=2))

    print(json.dumps(row, indent=2))
    return 0


def scope40_ci_report(args) -> int:
    """SCOPe-40 retrieval with bootstrap CIs -- the original single-task report."""
    from bootstrap_ci import boot_ci, per_query_metrics

    seqs, labels = load_scope()
    labels = np.asarray(labels)
    print(f"{len(seqs)} SCOPe-40 sequences; phmmer all-vs-all ...", flush=True)

    ranking = phmmer_ranking(seqs, cpus=args.cpus)
    m = per_query_metrics(ranking, labels)
    el = m["eligible"]

    report = {
        "method": "phmmer (HMMER3, pyhmmer)",
        "flags": {"E": E_VALUE, "rank_by": "bitscore desc", "self_excluded": True,
                  "no_hit_counts_as_failure": True},
        "n_queries": len(seqs),
        "n_eligible": int(el.sum()),
        "all_queries": {},
        "eligible": {},
    }
    print(f"\n{'metric':7s} {'all':>8s} {'eligible':>9s}  95% CI (eligible)")
    for metric in ("hit1", "hit10", "hit30", "ap"):
        allv = float(m[metric].mean())
        mean, lo, hi = boot_ci(m[metric][el])
        report["all_queries"][metric] = allv
        report["eligible"][metric] = {"mean": mean, "lo": lo, "hi": hi}
        print(f"{metric:7s} {allv:8.4f} {mean:9.4f}  [{lo:.4f}, {hi:.4f}]")

    out = Path(args.scope_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default=None,
                    help="Any key from benchmark_tasks.TASKS. Omit for the "
                         "SCOPe-40 bootstrap-CI report.")
    ap.add_argument("--cpus", type=int, default=48)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Subsample cap passed to prepare_data (default: no cap). "
                         "Recorded in the output row.")
    ap.add_argument("--output", default="results/benchmarks/hmmer_baseline.json")
    ap.add_argument("--scope_out", default="results/benchmarks/hmmer_scope40.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    return scope40_ci_report(args) if args.task is None else run_task(args)


def _selfcheck() -> None:
    """The property that makes the HMMER/MMseqs2 comparison meaningful: the same
    `hits` dict scores identically whichever engine produced it."""
    import tempfile
    from types import SimpleNamespace

    import mmseqs_baseline

    # 1. One scoring path, not two copies.
    assert score_task is mmseqs_baseline.score_task

    # 2. Real phmmer output -> hits dict; that same dict written as an MMseqs2 hit
    #    table and read back through the MMseqs2 reader must score identically.
    a1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    a2 = a1[:50] + "C" + a1[51:]
    b1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFELRRQACGRSDLAWQVQNMLHRYPQVVDMLRRLGLDPQAVE"
    b2 = b1[:-1] + "D"
    seqs, labels = [a1, a2, b1, b2], ["A", "A", "B", "B"]

    hmmer_hits = phmmer_hits(seqs, seqs, cpus=2)
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        for q, hs in hmmer_hits.items():
            for rank, (t, bits) in enumerate(hs):
                # query target fident alnlen evalue bits; E-value monotone in rank
                # so read_hits' (evalue asc, bits desc) key reproduces this order.
                fh.write(f"{q}\t{t}\t1.0\t100\t{10.0 ** -(50 - rank)}\t{bits}\n")
        tsv = fh.name
    mmseqs_hits = mmseqs_baseline.read_hits(Path(tsv))
    assert mmseqs_hits == hmmer_hits, (mmseqs_hits, hmmer_hits)

    for ptype, main_metric in (("retrieval", "Recall@1"), ("multiclass", "AUC")):
        cfg = SimpleNamespace(problem_type=ptype, main_metric=main_metric)
        via_hmmer = score_task(cfg, hmmer_hits, labels, labels)
        via_mmseqs = score_task(cfg, mmseqs_hits, labels, labels)
        assert via_hmmer == via_mmseqs, (ptype, via_hmmer, via_mmseqs)
        assert "hit_coverage" in via_hmmer, via_hmmer

    # 3. phmmer must actually find the near-identical partner first, and a query
    #    with no hit must score zero rather than crash or drop out.
    from bootstrap_ci import per_query_metrics

    m = per_query_metrics(phmmer_ranking(seqs, cpus=2), np.array(labels))
    assert m["eligible"].all(), m["eligible"]
    assert m["hit1"].tolist() == [1.0] * 4, (
        f"phmmer did not rank near-identical partners first: {m['hit1']}"
    )
    lone = "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
    m2 = per_query_metrics(phmmer_ranking([a1, a2, lone], cpus=2),
                           np.array(["A", "A", "Z"]))
    assert m2["eligible"].tolist() == [True, True, False]
    assert m2["hit1"][2] == 0.0

    # 4. Retrieval hit_coverage must not count a query's own self-hit.
    cfg = SimpleNamespace(problem_type="retrieval", main_metric="Recall@1")
    only_self = score_task(cfg, {0: [(0, 99.0)], 1: [(1, 99.0)]}, ["A", "A"], ["A", "A"])
    assert only_self["hit_coverage"] == 0.0, only_self

    # 5. Non-residue characters are coerced, not crashed on.
    assert phmmer_hits(["ACDEF|GHIK", a1], [a1, a2], cpus=2), "sanitisation failed"

    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
