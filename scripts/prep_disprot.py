#!/usr/bin/env python3
"""Download LiteFold/DisProt and build a residue-level disorder benchmark.

Run from the bench directory:
    python scripts/prep_disprot.py

Creates data/disprot/ as a local Arrow DatasetDict with train/validation/test.

DisProt ships no per-residue label column — only curated disordered region
spans (``region_starts`` / ``region_ends``, 1-based inclusive). We build the
standard per-residue binary disorder target (CAID-style): residue == 1 iff it
falls inside ANY curated region, else 0. The union of regions reproduces the
dataset's ``disorder_content`` fraction exactly (validated below).

Splits reuse DisProt's deterministic ``split_bucket`` (= sha256(disprot_id)%10):
  - test       = bucket 0   (DisProt's official held-out test, the HF `test` split)
  - validation = bucket 1   (carved from the HF `train` split)
  - train      = buckets 2-9

Each row: sequence (str), disorder_labels (list[int] 0/1), seq_length (int),
disprot_id (str), accession (str). disprot_id/accession are kept for the later
mmseqs pretraining-contamination filter (move test homologs out of train).

Problem type: token_classification (binary: 0=ordered, 1=disordered).
"""
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BENCH_DIR / "data" / "disprot"


def build_residue_labels(seq_len, starts, ends, terms=None):
    """Per-residue 0/1 disorder labels from 1-based inclusive region spans.

    Only regions whose ``region_term`` is exactly ``disorder`` count — DisProt
    also annotates functional aspects ('protein binding', 'flexible linker',
    'disorder to order', ...) that are NOT the disorder target. The union of
    disorder-term regions reproduces the dataset's ``disorder_content`` for
    ~99% of proteins; ``terms=None`` falls back to counting every region.
    """
    labels = [0] * seq_len
    starts = starts or []
    ends = ends or []
    terms = terms if terms is not None else ["disorder"] * len(starts)
    for s, e, t in zip(starts, ends, terms):
        if str(t).strip().lower() != "disorder":
            continue
        a = max(1, int(s)) - 1          # -> 0-based start
        b = min(int(e), seq_len)        # inclusive 1-based end == exclusive 0-based
        for i in range(a, b):
            labels[i] = 1
    return labels


def _build_rows(split_ds):
    rows, mismatches = [], 0
    for r in split_ds:
        seq = r["sequence"]
        if not seq:
            continue
        labels = build_residue_labels(
            len(seq), r["region_starts"], r["region_ends"], r.get("region_terms")
        )
        # Sanity: union of regions must reproduce disorder_content.
        dc = r.get("disorder_content")
        if dc is not None and abs(sum(labels) / len(seq) - dc) > 1e-3:
            mismatches += 1
        rows.append(
            {
                "sequence": seq,
                "disorder_labels": labels,
                "seq_length": len(seq),
                "disprot_id": r["disprot_id"],
                "accession": r.get("accession") or "",
            }
        )
    return rows, mismatches


def main():
    from datasets import Dataset, DatasetDict, load_dataset

    ds = load_dataset("LiteFold/DisProt")
    print(f"Loaded DisProt: { {k: len(v) for k, v in ds.items()} }")

    # test = HF test split (bucket 0). validation = bucket 1 carved from train.
    train_src = ds["train"]
    val_src = train_src.filter(lambda r: int(r["split_bucket"]) == 1)
    tr_src = train_src.filter(lambda r: int(r["split_bucket"]) != 1)

    split_rows, total_mismatch = {}, 0
    for name, src in (("train", tr_src), ("validation", val_src), ("test", ds["test"])):
        rows, mm = _build_rows(src)
        split_rows[name] = rows
        total_mismatch += mm
        n_dis = sum(sum(r["disorder_labels"]) for r in rows)
        n_res = sum(r["seq_length"] for r in rows)
        print(
            f"  {name}: {len(rows)} proteins, {n_res} residues, "
            f"{n_dis / max(n_res, 1):.3f} disordered fraction"
        )
    if total_mismatch:
        print(f"WARNING: {total_mismatch} proteins where region union != disorder_content")

    dd = DatasetDict({k: Dataset.from_list(v) for k, v in split_rows.items() if v})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(OUT_DIR))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
