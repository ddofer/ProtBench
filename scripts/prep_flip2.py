#!/usr/bin/env python3
"""Download LiteFold/FLIP2 and extract amylase + rhomax subtasks as local Arrow datasets.

Run from the bench directory:
    python scripts/prep_flip2.py

Creates:
    data/flip2_amylase/   (train=8921, test=4402 rows)
    data/flip2_rhomax/    (train=621, test=171 rows)

Each dataset has columns: sequence (str), score (float).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BENCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BENCH_DIR / "data"

SUBTASKS = {
    "amylase": "flip2_amylase",
    "rhomax": "flip2_rhomax",
}


def main():
    from datasets import Dataset, DatasetDict, load_dataset

    print("Downloading LiteFold/FLIP2 (may take a moment)...")
    ds = load_dataset("LiteFold/FLIP2")

    # Combine train + test into one iterable for filtering
    all_rows_by_split = {}
    for hf_split_name, hf_split_data in ds.items():
        for row in hf_split_data:
            task = row["task_name"]
            set_val = (row.get("set") or "train").lower()
            if task not in SUBTASKS:
                continue
            all_rows_by_split.setdefault(task, {"train": [], "test": []})
            bucket = "test" if set_val == "test" else "train"
            score = row.get("score_value")
            if score is None:
                try:
                    score = float(row.get("raw_target", 0.0))
                except (TypeError, ValueError):
                    continue
            if score is None or (isinstance(score, float) and np.isnan(score)):
                continue
            all_rows_by_split[task][bucket].append(
                {"sequence": str(row["sequence"]), "score": float(score)}
            )

    for task_name, out_name in SUBTASKS.items():
        if task_name not in all_rows_by_split:
            print(f"WARNING: no rows found for task '{task_name}', skipping")
            continue
        splits = all_rows_by_split[task_name]
        train_rows, test_rows = splits["train"], splits["test"]
        print(
            f"{task_name}: {len(train_rows)} train, {len(test_rows)} test rows"
        )
        dd = DatasetDict(
            {
                "train": Dataset.from_list(train_rows),
                "test": Dataset.from_list(test_rows),
            }
        )
        out_path = DATA_DIR / out_name
        out_path.mkdir(parents=True, exist_ok=True)
        dd.save_to_disk(str(out_path))
        print(f"  Saved to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
