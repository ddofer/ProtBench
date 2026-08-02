#!/usr/bin/env python3
"""Download and parse the FLIP Conservation dataset from bioembeddings.com.

Run from the bench directory:
    python scripts/prep_conservation.py

Creates data/conservation_flip/ as a local Arrow DatasetDict with:
  - train split: ~9,392 proteins
  - validation split: ~555 proteins
  - test split: ~519 proteins

Each row has columns:
  - sequence (str): amino acid sequence
  - conservation_labels (list[int]): per-residue conservation scores 1-9
  - seq_length (int)

Problem type: token_classification (9 classes: 1-9)
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BENCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BENCH_DIR / "data"
OUT_DIR = DATA_DIR / "conservation_flip"

AA_SEQS_URL = "http://data.bioembeddings.com/public/FLIP/fasta/conservation/sequences.fasta"
SAMPLED_URL = "http://data.bioembeddings.com/public/FLIP/fasta/conservation/sampled.fasta"


def _curl_download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  Using cached {dest}")
        return
    print(f"Downloading {url} ...")
    result = subprocess.run(
        ["curl", "--max-time", "300", "-#", url, "-o", str(dest)],
        check=True,
    )


def _parse_fasta(path):
    records = []
    current_header = None
    current_seq_parts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append((current_header, "".join(current_seq_parts)))
                current_header = line[1:].strip()
                current_seq_parts = []
            else:
                current_seq_parts.append(line)
    if current_header is not None:
        records.append((current_header, "".join(current_seq_parts)))
    return records


def main():
    from datasets import Dataset, DatasetDict

    tmp = Path("/tmp")
    seq_path = tmp / "conservation_sequences.fasta"
    sampled_path = tmp / "conservation_sampled.fasta"

    _curl_download(AA_SEQS_URL, seq_path)
    _curl_download(SAMPLED_URL, sampled_path)

    aa_records = _parse_fasta(seq_path)
    aa_by_id = {header.split()[0]: seq for header, seq in aa_records}
    print(f"Parsed {len(aa_by_id)} sequences from sequences.fasta")

    sampled_records = _parse_fasta(sampled_path)
    print(f"Parsed {len(sampled_records)} conservation records from sampled.fasta")

    split_rows = {"train": [], "validation": [], "test": []}
    n_missing_seq = 0
    n_bad_labels = 0

    for header, score_str in sampled_records:
        parts = header.split()
        seq_id = parts[0]
        meta = {k: v for k, v in (p.split("=") for p in parts[1:] if "=" in p)}

        set_val = meta.get("SET", "train").lower()
        is_val = meta.get("VALIDATION", "False").lower() == "true"

        if set_val == "test":
            split_key = "test"
        elif is_val:
            split_key = "validation"
        else:
            split_key = "train"

        aa_seq = aa_by_id.get(seq_id)
        if aa_seq is None:
            n_missing_seq += 1
            continue

        scores = [int(c) for c in score_str if c.isdigit()]
        if not scores:
            n_bad_labels += 1
            continue
        if len(scores) != len(aa_seq):
            # Trim to shorter to maintain alignment; log if severe mismatch
            min_len = min(len(scores), len(aa_seq))
            scores = scores[:min_len]
            aa_seq = aa_seq[:min_len]

        split_rows[split_key].append(
            {
                "sequence": aa_seq,
                "conservation_labels": scores,
                "seq_length": len(aa_seq),
            }
        )

    if n_missing_seq:
        print(f"WARNING: {n_missing_seq} records had no matching AA sequence")
    if n_bad_labels:
        print(f"WARNING: {n_bad_labels} records had empty label strings")

    for split_key, rows in split_rows.items():
        print(f"  {split_key}: {len(rows)} proteins")

    dd = DatasetDict(
        {k: Dataset.from_list(v) for k, v in split_rows.items() if v}
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(OUT_DIR))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
