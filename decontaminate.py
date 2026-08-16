"""Filter a corpus against benchmark test sets with MMseqs2.

Pretraining corpora routinely contain the very proteins a benchmark holds out.
CATH, SCOPe and remote-homology test sets are drawn from the PDB, and any
corpus built from UniRef or the PDB will overlap them, so a model can score well
by recall rather than generalisation.

This script searches each task's TEST sequences against your corpus and reports
the corpus entries that match above an identity and coverage threshold -- a stop
list to drop before pretraining.

Reuses ``mmseqs_baseline`` (binary resolution, FASTA writing) and the benchmark's
own ``prepare_data``, so the test sequences filtered against are byte-identical
to the ones the benchmark will later score on.

Usage:
    python decontaminate.py --selfcheck
    python decontaminate.py --corpus pretrain.fasta \\
        --tasks cath_eat remote_homology scope40_retrieval \\
        --min-seq-id 0.3 --coverage 0.8 -o stoplist.txt

    # then, before pretraining:
    python decontaminate.py --corpus pretrain.fasta --tasks cath_eat \\
        --write-filtered clean.fasta

Needs `mmseqs` on PATH or $MMSEQS_BIN (conda install -c bioconda mmseqs2).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

from mmseqs_baseline import MMSEQS, write_fasta

logger = logging.getLogger(__name__)

# 30% identity over 80% of the sequence is the conventional "same fold, probably
# the same protein" line used for redundancy reduction (the CD-HIT / PISCES
# convention). Both are knobs -- structural benchmarks often want them lower,
# since remote homologues still leak fold information.
DEFAULT_MIN_SEQ_ID = 0.3
DEFAULT_COVERAGE = 0.8


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    """[(id, sequence)] preserving the corpus's own identifiers.

    The id is the header up to the first whitespace, matching what MMseqs2 puts
    in its query/target columns.
    """
    records: List[Tuple[str, str]] = []
    header: str | None = None
    chunks: List[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].split()[0] if len(line) > 1 else ""
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def parse_hits(
    tsv: Path, min_seq_id: float, coverage: float
) -> Dict[str, Tuple[str, float]]:
    """corpus_id -> (test_id, identity) for hits clearing both thresholds.

    MMseqs2 already filters on these, but re-checking here means the numbers in
    the report come from the alignment rows themselves rather than from trusting
    a flag, and it keeps the thresholds meaningful if the flags ever drift.
    """
    contaminated: Dict[str, Tuple[str, float]] = {}
    with open(tsv) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            test_id, corpus_id = parts[0], parts[1]
            fident, qcov, tcov = float(parts[2]), float(parts[3]), float(parts[4])
            if fident < min_seq_id or min(qcov, tcov) < coverage:
                continue
            # Keep the strongest match, so the report names the worst offender.
            prev = contaminated.get(corpus_id)
            if prev is None or fident > prev[1]:
                contaminated[corpus_id] = (test_id, fident)
    return contaminated


def search(
    query_fasta: Path,
    target_fasta: Path,
    out_tsv: Path,
    *,
    min_seq_id: float,
    coverage: float,
    threads: int,
) -> None:
    """mmseqs easy-search tuned for contamination, not homology recall.

    Distinct from ``mmseqs_baseline.easy_search``, which deliberately runs a
    permissive `-e 10` search because recall is the point there. Here the
    thresholds do the work, so they are pushed into MMseqs2 rather than applied
    afterwards -- on a corpus of millions that difference is the whole runtime.
    """
    tmp = Path(tempfile.mkdtemp(prefix="mmseqs_decon_", dir=out_tsv.parent))
    cmd = [
        str(MMSEQS), "easy-search",
        str(query_fasta), str(target_fasta), str(out_tsv), str(tmp),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "0",
        "-s", "7.5",
        "--max-seqs", "1000",
        "--alignment-mode", "3",
        "--format-output", "query,target,fident,qcov,tcov,evalue",
        "--threads", str(threads),
        "--remove-tmp-files", "-v", "3",
    ]
    logger.info("running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=24 * 3600)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"mmseqs failed ({exc.returncode}): {' '.join(cmd)}") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sequences(task: str) -> List[str]:
    """The task's TEST split, exactly as the benchmark will score it."""
    from mmseqs_baseline import load_task

    _cfg, _train, (test_seqs, _test_labels) = load_task(task)
    return list(test_seqs)


def _iter_clean(
    records: List[Tuple[str, str]], stop: Set[str]
) -> Iterator[Tuple[str, str]]:
    for rid, seq in records:
        if rid not in stop:
            yield rid, seq


def _selfcheck() -> None:
    """Runnable check: python decontaminate.py --selfcheck

    Covers the parts that do not need the mmseqs binary: FASTA round-tripping
    and threshold logic. A hit that clears identity but not coverage must NOT be
    called contamination -- getting that backwards silently deletes good data.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        fasta = tmp / "corpus.fasta"
        fasta.write_text(
            ">sp|P1|FOO some description\nMKTA\nYIAK\n"
            ">sp|P2|BAR\nGGGGG\n"
            "\n"
            ">P3\nWWWW\n"
        )
        records = read_fasta(fasta)
        assert [r[0] for r in records] == ["sp|P1|FOO", "sp|P2|BAR", "P3"], records
        assert records[0][1] == "MKTAYIAK", "multi-line sequence not joined"
        assert records[2][1] == "WWWW", "blank line broke parsing"

        tsv = tmp / "hits.tsv"
        tsv.write_text(
            # test_id  corpus_id  fident  qcov  tcov  evalue
            "t1\tsp|P1|FOO\t0.95\t0.99\t0.99\t1e-40\n"   # clear contamination
            "t2\tsp|P2|BAR\t0.91\t0.20\t0.95\t1e-05\n"   # identical but 20% cov
            "t3\tP3\t0.10\t0.99\t0.99\t1e-02\n"          # full cov, 10% identity
            "t4\tsp|P1|FOO\t0.40\t0.99\t0.99\t1e-10\n"   # weaker dup of P1
        )
        hits = parse_hits(tsv, min_seq_id=0.3, coverage=0.8)
        assert set(hits) == {"sp|P1|FOO"}, hits
        assert hits["sp|P1|FOO"] == ("t1", 0.95), "did not keep the strongest match"

        # Lowering identity catches the remote homologue; lowering coverage
        # catches the partial match. Each threshold acts on its own.
        assert set(parse_hits(tsv, 0.05, 0.8)) == {"sp|P1|FOO", "P3"}
        assert set(parse_hits(tsv, 0.3, 0.1)) == {"sp|P1|FOO", "sp|P2|BAR"}

        clean = list(_iter_clean(records, set(hits)))
        assert [r[0] for r in clean] == ["sp|P2|BAR", "P3"]

    print("decontaminate selfcheck OK")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, help="FASTA to filter")
    parser.add_argument(
        "--tasks", nargs="+", default=[], help="benchmark task keys to filter against"
    )
    parser.add_argument("--min-seq-id", type=float, default=DEFAULT_MIN_SEQ_ID)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "-o", "--out", type=Path, help="write the stop list (one corpus id per line)"
    )
    parser.add_argument(
        "--write-filtered", type=Path, help="write the corpus minus the stop list"
    )
    parser.add_argument("--workdir", type=Path, default=Path("decontaminate_tmp"))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if args.selfcheck:
        _selfcheck()
        return 0
    if not args.corpus or not args.tasks:
        parser.error("--corpus and --tasks are required (or pass --selfcheck)")
    if not shutil.which(str(MMSEQS)) and not Path(MMSEQS).exists():
        parser.error(
            f"mmseqs not found (looked at {MMSEQS}). conda install -c bioconda "
            "mmseqs2, or set $MMSEQS_BIN."
        )

    args.workdir.mkdir(parents=True, exist_ok=True)
    records = read_fasta(args.corpus)
    logger.info("corpus: %d sequences from %s", len(records), args.corpus)

    stop: Set[str] = set()
    for task in args.tasks:
        seqs = test_sequences(task)
        query = args.workdir / f"{task}_test.fasta"
        write_fasta(seqs, query)
        tsv = args.workdir / f"{task}_hits.tsv"
        search(
            query,
            args.corpus,
            tsv,
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
            threads=args.threads,
        )
        hits = parse_hits(tsv, args.min_seq_id, args.coverage)
        logger.info(
            "%s: %d test sequences -> %d corpus entries contaminated (%.2f%%)",
            task,
            len(seqs),
            len(hits),
            100.0 * len(hits) / max(len(records), 1),
        )
        stop |= set(hits)

    logger.info(
        "TOTAL: %d of %d corpus entries hit a test sequence at >=%.0f%% identity "
        "over >=%.0f%% coverage (%.2f%%)",
        len(stop),
        len(records),
        100 * args.min_seq_id,
        100 * args.coverage,
        100.0 * len(stop) / max(len(records), 1),
    )

    if args.out:
        args.out.write_text("".join(f"{rid}\n" for rid in sorted(stop)))
        logger.info("stop list -> %s", args.out)
    if args.write_filtered:
        with open(args.write_filtered, "w") as fh:
            for rid, seq in _iter_clean(records, stop):
                fh.write(f">{rid}\n{seq}\n")
        logger.info(
            "filtered corpus -> %s (%d sequences)",
            args.write_filtered,
            len(records) - len(stop),
        )
    if not args.out and not args.write_filtered:
        for rid in sorted(stop):
            print(rid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
