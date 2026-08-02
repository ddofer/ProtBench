#!/usr/bin/env python
"""MMseqs2-only baseline for the two structural benchmark tasks.

Answers the reviewer question "how much of ProtSent's structural performance is
just sequence similarity?" by scoring the SAME tasks with alignment instead of
embeddings, under the SAME metric definitions:

  scope40_retrieval  family-level Recall@{1,10,30}, self-match excluded --
                     identical to evaluate_retrieval() in
                     protein_benchmark_suite.py:1863-1907, with cosine-NN rank
                     replaced by MMseqs2 bitscore rank.
  remote_homology    multiclass fold prediction. Per-class score = max bitscore
                     over that class's training sequences, which yields a dense
                     score vector, so the task's AUC main metric stays
                     comparable instead of degenerating to hard 1-NN accuracy.

Queries with no hit at all are genuine misses (recall 0 / lowest-rank class),
not dropped rows -- that failure to retrieve is the point of the baseline.

Usage:
    uv run --no-sync python mmseqs_baseline.py --task scope40_retrieval
    uv run --no-sync python mmseqs_baseline.py --task remote_homology
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("mmseqs_baseline")

# $MMSEQS_BIN wins, else whatever is on PATH. ProtSent vendored a copy under
# tools/mmseqs/, which did not survive the move here; mmseqs is a system tool,
# not something this repo should ship.
MMSEQS = Path(os.environ.get("MMSEQS_BIN") or shutil.which("mmseqs") or "mmseqs")
# Sensitive search: this is a homology-detection baseline, so recall matters far
# more than runtime. Both task sets are small (~10-15k sequences).
SEARCH_FLAGS = [
    "-s", "7.5",
    "-e", "10",
    "--max-seqs", "300",
    "--alignment-mode", "3",
    "--format-output", "query,target,fident,alnlen,evalue,bits",
]


def write_fasta(seqs: list[str], path: Path) -> None:
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")


def easy_search(query: Path, target: Path, out_tsv: Path, threads: int) -> None:
    """Run mmseqs easy-search; raises with the full command on failure."""
    tmp = Path(tempfile.mkdtemp(prefix="mmseqs_bl_", dir=out_tsv.parent))
    cmd = [
        str(MMSEQS), "easy-search", str(query), str(target), str(out_tsv), str(tmp),
        *SEARCH_FLAGS, "--threads", str(threads), "--remove-tmp-files", "-v", "3",
    ]
    logger.info("running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=24 * 3600)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"mmseqs failed ({exc.returncode}): {' '.join(cmd)}") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def read_hits(tsv: Path) -> dict[int, list[tuple[int, float]]]:
    """query_idx -> [(target_idx, bits), ...] in rank order.

    Ranked by **E-value ascending, then bitscore descending** -- the conventional
    ordering for a sequence-search baseline, and the one the rebuttal reports.
    Bitscore alone gives a different (and, for SCOPe-40, markedly more optimistic)
    ranking, so the tie-break matters and is stated explicitly rather than left to
    whatever order MMseqs2 emitted.

    The returned score stays the bitscore, since downstream class scoring wants a
    higher-is-better magnitude rather than an E-value.
    """
    hits: dict[int, list[tuple[int, float, float]]] = {}
    with open(tsv) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            # query, target, fident, alnlen, evalue, bits
            hits.setdefault(int(parts[0]), []).append(
                (int(parts[1]), float(parts[4]), float(parts[5]))
            )
    ranked: dict[int, list[tuple[int, float]]] = {}
    for q, v in hits.items():
        v.sort(key=lambda t: (t[1], -t[2]))
        ranked[q] = [(tgt, bits) for tgt, _evalue, bits in v]
    return ranked


def load_task(task: str, max_samples: int | None = None):
    """Load train/test exactly as the benchmark suite does with --eval_split test.

    Delegates to protein_benchmark_suite.prepare_data so the baseline sees byte
    identical inputs to the model side -- including whitespace stripping,
    split_column / auto_split handling and label parsing. Anything else would
    compare the two systems on different data.
    """
    from benchmark_tasks import TASKS
    from protein_benchmark_suite import prepare_data

    cfg = TASKS[task]
    tr_seqs, tr_labels, te_seqs, te_labels, _, meta = prepare_data(
        cfg, max_samples=max_samples, eval_split="test"
    )
    logger.info("split metadata: %s", meta)
    return cfg, (tr_seqs, tr_labels), (te_seqs, te_labels)


def eval_retrieval(hits, labels, k_list=(1, 10, 30)) -> dict[str, float]:
    """Family-level Recall@K and MAP, self excluded.

    Recall@K mirrors evaluate_retrieval() (protein_benchmark_suite.py:1863-1907)
    so the baseline and the model are scored identically.

    Reported over two query populations, because they answer different questions
    and the gap between them is large:

      all queries      every query counts, including the ~N with no same-family
                       partner anywhere in the gallery. Those are unachievable by
                       any method, so this depresses every system equally.
      eligible queries queries that have at least one non-self same-family protein
                       in the gallery, i.e. the ones where retrieval is possible
                       at all. This is the fairer denominator for comparing
                       methods, and the one to quote when a paper reports
                       "eligible" or "achievable" recall.

    MAP is average precision over the ranked list, averaged over queries, with
    self-matches removed and unretrieved relevant items contributing zero.
    """
    n = len(labels)
    labels = list(labels)
    # A query is eligible iff some OTHER gallery protein shares its family.
    counts: dict = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    eligible = [q for q in range(n) if counts[labels[q]] > 1]

    def _recall_at(qs, k):
        return float(
            np.mean([
                any(
                    labels[t] == labels[q]
                    for t in [t for t, _ in hits.get(q, []) if t != q][:k]
                )
                for q in qs
            ])
        ) if qs else float("nan")

    def _map(qs):
        aps = []
        for q in qs:
            n_rel = counts[labels[q]] - 1  # exclude self
            if n_rel <= 0:
                aps.append(0.0)
                continue
            ranked = [t for t, _ in hits.get(q, []) if t != q]
            found = 0
            precision_sum = 0.0
            for rank, t in enumerate(ranked, start=1):
                if labels[t] == labels[q]:
                    found += 1
                    precision_sum += found / rank
            aps.append(precision_sum / n_rel)  # unretrieved relevants score 0
        return float(np.mean(aps)) if aps else float("nan")

    out: dict[str, float] = {}
    for k in k_list:
        out[f"Recall@{k}"] = _recall_at(range(n), k)
    out["MAP"] = _map(range(n))
    for k in k_list:
        out[f"eligible_Recall@{k}"] = _recall_at(eligible, k)
    out["eligible_MAP"] = _map(eligible)
    out["n_queries"] = n
    out["n_eligible_queries"] = len(eligible)
    return out


def eval_regression(hits, train_labels, test_labels) -> dict[str, float]:
    """1-NN by bitscore: predict the target value of the best-aligning train seq.

    Queries with no hit fall back to the training mean -- the least-informative
    guess available, so they neither help nor are silently dropped.
    """
    from scipy.stats import spearmanr

    y_train = np.asarray(train_labels, dtype=np.float64)
    y_true = np.asarray(test_labels, dtype=np.float64)
    fallback = float(y_train.mean())
    y_pred = np.array(
        [y_train[hits[q][0][0]] if hits.get(q) else fallback
         for q in range(len(y_true))]
    )
    covered = np.array([bool(hits.get(q)) for q in range(len(y_true))])
    rho = spearmanr(y_true, y_pred).statistic
    return {
        "Spearman": float(rho) if rho == rho else None,
        "MSE": float(np.mean((y_true - y_pred) ** 2)),
        "hit_coverage": float(covered.mean()),
    }


def eval_multilabel(hits, train_labels, test_labels) -> dict[str, float]:
    """1-NN by bitscore: copy the best-aligning train sequence's label set."""
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    mlb.fit([list(x) for x in train_labels] + [list(x) for x in test_labels])
    Y = mlb.transform([list(x) for x in test_labels])
    empty = [()] * 0
    pred = [
        train_labels[hits[q][0][0]] if hits.get(q) else empty
        for q in range(len(test_labels))
    ]
    P = mlb.transform([list(x) for x in pred])
    covered = np.array([bool(hits.get(q)) for q in range(len(test_labels))])
    return {
        "F1_Macro": float(f1_score(Y, P, average="macro", zero_division=0)),
        "F1_Micro": float(f1_score(Y, P, average="micro", zero_division=0)),
        "hit_coverage": float(covered.mean()),
    }


def eval_multiclass(hits, train_labels, test_labels) -> dict[str, float]:
    """Per-class score = max bitscore over that class's training sequences."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    classes = sorted(set(train_labels) | set(test_labels))
    cls_idx = {c: i for i, c in enumerate(classes)}
    scores = np.zeros((len(test_labels), len(classes)), dtype=np.float32)
    for q in range(len(test_labels)):
        for t, bits in hits.get(q, []):
            j = cls_idx[train_labels[t]]
            if bits > scores[q, j]:
                scores[q, j] = bits

    y_true = np.array([cls_idx[c] for c in test_labels])
    y_pred = scores.argmax(axis=1)
    covered = (scores.max(axis=1) > 0)

    # Rows with no hit have an all-zero score vector; argmax picks class 0
    # arbitrarily. Left in deliberately -- "no homolog found" is a real error
    # mode of a search baseline and must count against it.
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hit_coverage": float(covered.mean()),
    }
    # Macro one-vs-rest AUC computed per class directly. sklearn's
    # multi_class="ovr" insists the score matrix be probabilities summing to 1,
    # which a per-class max-bitscore matrix is not; normalising it into a
    # softmax only distorts the ranking (bitscores span hundreds, so the softmax
    # saturates to one-hot). AUC only cares about the ordering within each class
    # column, so the raw scores are the right input.
    aucs = []
    for j in range(len(classes)):
        pos = y_true == j
        if pos.any() and not pos.all():  # AUC undefined for a single-class column
            aucs.append(roc_auc_score(pos, scores[:, j]))
    metrics["AUC"] = float(np.mean(aucs)) if aucs else None
    metrics["AUC_n_classes"] = len(aucs)
    return metrics


def score_task(cfg, hits, train_labels, test_labels) -> dict:
    """The one scoring path. Any search engine that produces the same `hits`
    dict -- {query_idx: [(target_idx, bitscore), ...]} ranked best-first -- gets
    scored identically here.

    `hmmer_baseline.py` imports this rather than reimplementing the evaluators:
    two alignment baselines scored by two copies of the metric code would make
    the comparison between them meaningless.
    """
    if cfg.problem_type == "retrieval":
        metrics = eval_retrieval(hits, test_labels)
    elif cfg.problem_type == "regression":
        metrics = eval_regression(hits, train_labels, test_labels)
    elif cfg.problem_type == "multilabel":
        metrics = eval_multilabel(hits, train_labels, test_labels)
    else:
        metrics = eval_multiclass(hits, train_labels, test_labels)
    # Retrieval is the one evaluator that does not report coverage itself, and
    # it is all-vs-all, so a query's self-hit must not count as coverage.
    metrics.setdefault(
        "hit_coverage",
        float(np.mean([
            any(t != q for t, _ in hits.get(q, ())) for q in range(len(test_labels))
        ])),
    )
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True,
                    help="Any key from benchmark_tasks.TASKS")
    ap.add_argument("--work_dir", default="/storage/users/ddofer/data/mmseqs_baseline")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Subsample cap passed to prepare_data (default: no cap)")
    ap.add_argument("--output", default="results/benchmarks/mmseqs_baseline.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not MMSEQS.exists():
        logger.error(
            "mmseqs binary not found (looked at %s). Install it (conda install -c "
            "bioconda mmseqs2, or the static build from github.com/soedinglab/"
            "MMseqs2/releases) and put it on PATH, or point $MMSEQS_BIN at it.",
            MMSEQS,
        )
        return 1

    work = Path(args.work_dir) / args.task
    work.mkdir(parents=True, exist_ok=True)

    cfg, (train_seqs, train_labels), (test_seqs, test_labels) = load_task(
        args.task, args.max_samples
    )
    logger.info("%s [%s]: %d train / %d test sequences",
                cfg.name, cfg.problem_type, len(train_seqs), len(test_seqs))

    q_fa, t_fa = work / "query.fasta", work / "target.fasta"
    write_fasta(test_seqs, q_fa)
    write_fasta(train_seqs, t_fa)

    tsv = work / "hits.tsv"
    if tsv.exists() and tsv.stat().st_size > 0:
        logger.info("reusing existing hits: %s", tsv)
    else:
        easy_search(q_fa, t_fa, tsv, args.threads)
    hits = read_hits(tsv)
    logger.info("%d/%d queries got >=1 hit", len(hits), len(test_seqs))

    metrics = score_task(cfg, hits, train_labels, test_labels)

    row = {
        "Model": "MMseqs2 (alignment baseline)",
        "Task": cfg.name,
        "task_key": args.task,
        "main_metric": cfg.main_metric,
        "problem_type": cfg.problem_type,
        "eval_split": "test",  # prepare_data(eval_split="test"); NOT the suite default
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "mmseqs_flags": " ".join(SEARCH_FLAGS),
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


def _selfcheck() -> None:
    """Metric functions must agree with the suite's definitions."""
    # Recall/MAP: q0 and q2 share family "A"; q1 is alone in "B".
    labels = ["A", "B", "A"]
    hits = {0: [(0, 99.0), (1, 5.0), (2, 4.0)], 1: [(1, 99.0), (0, 3.0)], 2: []}
    r = eval_retrieval(hits, labels, k_list=(1, 2))
    # q0@1 -> best non-self is 1 ("B"): miss. q0@2 -> includes 2 ("A"): hit.
    # q1 never matches (no other "B"). q2 has no hits at all: miss.
    assert r["Recall@1"] == 0.0, r
    assert abs(r["Recall@2"] - 1 / 3) < 1e-9, r
    # Only q0 and q2 have a same-family partner, so 2 of 3 are eligible.
    assert r["n_eligible_queries"] == 2, r
    assert r["n_queries"] == 3, r
    # Eligible Recall@2: q0 hits, q2 has no hits -> 1/2.
    assert abs(r["eligible_Recall@2"] - 0.5) < 1e-9, r
    # MAP over all 3: q0 finds its 1 relevant at rank 2 -> AP 0.5; q1 0; q2 0.
    assert abs(r["MAP"] - (0.5 / 3)) < 1e-9, r
    assert abs(r["eligible_MAP"] - 0.25) < 1e-9, r

    # Rank order must be E-value ascending, then bitscore descending.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        # target 7 has a worse bitscore but a far better E-value -> must rank first
        fh.write("0\t7\t0.9\t100\t1e-40\t50\n")
        fh.write("0\t8\t0.9\t100\t1e-3\t900\n")
        tmp = fh.name
    ranked = read_hits(Path(tmp))
    assert [t for t, _ in ranked[0]] == [7, 8], ranked

    # Multiclass: test0 best-hits a class-"x" train seq, test1 has no hit.
    m = eval_multiclass({0: [(0, 50.0), (1, 10.0)]}, ["x", "y"], ["x", "y"])
    assert m["hit_coverage"] == 0.5, m
    assert m["Accuracy"] == 0.5, m  # test0 correct, test1 falls to class 0 ("x")

    # Regression: no-hit queries fall back to the training mean.
    g = eval_regression({0: [(1, 9.0)]}, [0.0, 10.0], [10.0, 5.0])
    assert g["hit_coverage"] == 0.5, g

    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
