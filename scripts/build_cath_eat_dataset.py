#!/usr/bin/env python
"""Build the CATH annotation-transfer dataset used by ProtTucker (Heinzinger 2022).

Reproduces the evaluation from "Contrastive learning on protein embeddings
enlightens midnight zone" (NAR Genomics and Bioinformatics 4:2, lqac043,
doi:10.1093/nargab/lqac043): transfer a CATH classification from a large
labelled lookup set to a small, strictly non-redundant query set, and score how
often the transferred label is right.

The splits are the authors' own, taken verbatim from github.com/Rostlab/EAT so
the numbers stay comparable to their Table 1:

  lookup69k  69k domains, the labelled set annotations are transferred FROM.
  test219    219 queries. Non-redundant to lookup69k at HVAL <= 0 -- roughly,
             no query has a lookup relative detectable by sequence alignment.
             This is what makes the benchmark a midnight-zone test rather than
             a lookup exercise.

Labels come from CATH v4.3.0's CathDomainList (CLF 2.0), whose columns 2-5 are
the Class, Architecture, Topology and Homologous-superfamily numbers. The four
nest, so `1.10.8.10` at H implies `1.10.8` at T and so on; we emit one column
per level rather than making every consumer re-split the string.

Sources (both CC-BY-4.0, cite the paper above if you use this):
  splits  https://github.com/Rostlab/EAT/tree/main/data/ProtTucker
  labels  https://zenodo.org/records/14675997  (cath-domain-list.txt)

Usage:
    python scripts/build_cath_eat_dataset.py \\
        --eat_dir  /path/to/EAT/data/ProtTucker \\
        --cath_list /path/to/cath-domain-list.txt \\
        --out_dir  ./cath_eat
    # then, to publish:
    #   huggingface-cli upload <user>/cath43-eat ./cath_eat --repo-type=dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The splits that make up the benchmark. train66k/val200 are the authors'
# training and early-stopping sets -- included so the dataset can also be used
# to retrain a ProtTucker-style model, not just to evaluate one.
SPLITS = ("test219", "lookup69k", "train66k", "val200", "test300")


def read_fasta(path: Path) -> dict[str, str]:
    """domain id -> sequence. Headers here are bare ids, e.g. '>2vwaA00'."""
    seqs: dict[str, str] = {}
    key = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if key is not None:
                seqs[key] = "".join(chunks)
            key = line[1:].strip().split()[0]
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if key is not None:
        seqs[key] = "".join(chunks)
    return seqs


def read_cath_labels(path: Path) -> dict[str, tuple[str, str, str, str]]:
    """domain id -> (C, A, T, H) from CathDomainList CLF 2.0 columns 2-5."""
    labels: dict[str, tuple[str, str, str, str]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 5:
            continue
        labels[f[0]] = (f[1], f[2], f[3], f[4])
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eat_dir", type=Path, required=True)
    ap.add_argument("--cath_list", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    labels = read_cath_labels(args.cath_list)
    print(f"CATH labels: {len(labels):,} domains")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for split in SPLITS:
        fasta = args.eat_dir / f"{split}.fasta"
        if not fasta.exists():
            print(f"  skip {split}: {fasta} not found")
            continue
        seqs = read_fasta(fasta)

        rows, missing = [], 0
        for dom, seq in seqs.items():
            hit = labels.get(dom)
            if hit is None:
                # A domain with no CATH entry has no label to transfer, so it
                # can be neither a query nor a lookup target. Dropping is the
                # only correct move; count them so the loss is visible.
                missing += 1
                continue
            c, a, t, h = hit
            rows.append(
                {
                    "id": dom,
                    "sequence": seq,
                    "cath_c": c,
                    "cath_a": f"{c}.{a}",
                    "cath_t": f"{c}.{a}.{t}",
                    "cath_h": f"{c}.{a}.{t}.{h}",
                }
            )

        out = args.out_dir / f"{split}.jsonl"
        with out.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        stats[split] = {
            "rows": len(rows),
            "dropped_unlabelled": missing,
            "superfamilies": len({r["cath_h"] for r in rows}),
        }
        print(
            f"  {split:10s} {len(rows):6,} rows  "
            f"{stats[split]['superfamilies']:5,} superfamilies  "
            f"({missing} dropped, no CATH entry)"
        )

    # A query whose superfamily is absent from the lookup set cannot be answered
    # correctly by ANY method, so the paper drops it from that level's score
    # rather than charging every method for an impossible case. The exclusion is
    # per level, because a query can be answerable at T and not at H.
    #
    # We emit the H-level survivors as their own split. H is the headline number
    # and the only level where the exclusion is large (69 of 219), so shipping it
    # pre-filtered means the default task is comparable to the paper without
    # anyone having to know this paragraph exists.
    if "test219" in stats and "lookup69k" in stats:
        queries = [json.loads(x) for x in (args.out_dir / "test219.jsonl").read_text().splitlines()]
        lookup = [json.loads(x) for x in (args.out_dir / "lookup69k.jsonl").read_text().splitlines()]
        print("\nqueries answerable per level (rest are excluded from that level):")
        for level in ("cath_c", "cath_a", "cath_t", "cath_h"):
            present = {r[level] for r in lookup}
            n = sum(1 for r in queries if r[level] in present)
            stats.setdefault("answerable", {})[level] = n
            print(f"  {level:8s} {n:4d} / {len(queries)}")

        present_h = {r["cath_h"] for r in lookup}
        keep = [r for r in queries if r["cath_h"] in present_h]
        with (args.out_dir / "test_h.jsonl").open("w") as fh:
            for r in keep:
                fh.write(json.dumps(r) + "\n")
        stats["test_h"] = {"rows": len(keep), "superfamilies": len({r["cath_h"] for r in keep})}
        print(f"\nwrote test_h.jsonl ({len(keep)} answerable queries) -- the default benchmark split")

    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
