"""Loading hub CSV repos whose per-split files have different column sets.

`proteinea/secondary_structure_prediction` is the real case: `CASP13.csv` carries
`xyz_coordinates`, `CASP14.csv` additionally carries `Unnamed: 0`, and
`training_hhblits.csv` carries `cb513_mask`. Handing all of them to one
`load_dataset` call makes the CSV builder try to unify the schemas and raise
"Please either edit the data files to have matching columns".

Offline: fixtures are temp CSVs read through the local `csv` builder.
"""

import csv

import pytest

from protein_benchmark_suite import load_dataset_splits


def _write_csv(path, columns, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


@pytest.fixture
def mismatched_csvs(tmp_path):
    """Two CSVs sharing the columns we read, differing in everything else."""
    train = _write_csv(
        tmp_path / "train.csv",
        ["input", "dssp8", "cb513_mask"],
        [{"input": "ACDEF", "dssp8": "HHEEC", "cb513_mask": "1.0"}],
    )
    test = _write_csv(
        tmp_path / "test.csv",
        ["Unnamed: 0", "input", "dssp8", "xyz_coordinates"],
        [{"Unnamed: 0": 0, "input": "GHIKL", "dssp8": "GGTTS", "xyz_coordinates": "[]"}],
    )
    return {"train": train, "test": test}


def test_splits_with_different_columns_load(mismatched_csvs):
    ds = load_dataset_splits("csv", data_files=mismatched_csvs)
    assert set(ds.keys()) == {"train", "test"}
    assert len(ds["train"]) == 1
    assert len(ds["test"]) == 1


def test_each_split_keeps_its_own_columns(mismatched_csvs):
    ds = load_dataset_splits("csv", data_files=mismatched_csvs)
    assert "cb513_mask" in ds["train"].column_names
    assert "xyz_coordinates" in ds["test"].column_names
    # The shared columns the tasks actually read are present in both.
    for split in ds.values():
        assert "input" in split.column_names
        assert "dssp8" in split.column_names


def test_values_land_in_the_right_split(mismatched_csvs):
    ds = load_dataset_splits("csv", data_files=mismatched_csvs)
    assert ds["train"][0]["input"] == "ACDEF"
    assert ds["test"][0]["dssp8"] == "GGTTS"


def test_without_data_files_it_is_a_plain_load(tmp_path):
    """The no-data_files path must stay the ordinary single load_dataset call."""
    path = _write_csv(
        tmp_path / "only.csv", ["input", "dssp3"], [{"input": "AC", "dssp3": "HH"}]
    )
    ds = load_dataset_splits("csv", data_files={"train": path})
    assert list(ds.keys()) == ["train"]
