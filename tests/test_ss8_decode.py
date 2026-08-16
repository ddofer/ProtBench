"""SS8 label decoding and the new task registry entries.

The headline case is a silent-wrong-answer guard: an 8-state task is also named
"Secondary Structure ...", so a naive name match sends it down the 3-state HEC
branch, which DROPS every G/I/B/S/T/D residue instead of raising. The residue
probe then truncates the embeddings to the shortened label list and reports a
plausible-looking number computed on the wrong residues.
"""

import pytest

from benchmark_tasks import (
    DEFAULT_TASKS,
    MULTILABEL_EXCLUDED_TASKS,
    SS_HELDOUT_TASKS,
    TASKS,
)
from protein_benchmark_suite import (
    _SS3_ALPHABET,
    _SS8_ALPHABET,
    _decode_residue_label,
)

# Every symbol the two SS8 sources emit: the 8 DSSP states plus GleghornLab's
# `D` for unassigned termini.
SS8_LABEL = "DDDCHHHEEGGITTSSBBC"
SS3_LABEL = "CCCHHHEEECCC"


def test_ss8_alphabet_is_the_eight_published_dssp_states():
    """Both sources' REAL states are covered; GleghornLab's extra `D` marker is
    deliberately not a class (see the unassigned-residue section below)."""
    proteinea = set("BCEGHIST")
    gleghorn_real = set("CBEGHIST")
    assert set(_SS8_ALPHABET) == proteinea == gleghorn_real | {"C"}
    assert len(_SS8_ALPHABET) == 8
    assert "D" not in _SS8_ALPHABET


def test_ss8_by_task_name_keeps_every_residue():
    decoded = _decode_residue_label("Secondary Structure 8 (DSSP8)", "labels", SS8_LABEL)
    assert len(decoded) == len(SS8_LABEL), "SS8 decoded against the 3-state alphabet"
    assert max(decoded) >= 3, "8-state labels collapsed into 3 classes"


def test_ss8_by_label_column_keeps_every_residue():
    """The held-out sets are named 'Secondary Structure 8 (CASP12)' with no
    literal 'ss8', so routing must also key off the dssp8 column."""
    decoded = _decode_residue_label(
        "Secondary Structure 8 (CASP12)", "dssp8", "BCEGHIST"
    )
    assert len(decoded) == 8
    assert len(set(decoded)) == 8


def test_ss3_still_routes_to_three_states():
    decoded = _decode_residue_label("Secondary Structure 3 (CASP12)", "dssp3", SS3_LABEL)
    assert len(decoded) == len(SS3_LABEL)
    assert max(decoded) < len(_SS3_ALPHABET)


def test_ss3_and_ss8_disagree_on_the_same_string():
    """Direct evidence of the routing trap. Both branches now preserve length,
    so the tell is SCORABILITY: the 3-state alphabet cannot represent G/I/B/S/T,
    so those residues come back as ignore, while the 8-state branch gives every
    one of them a real class. Routing an 8-state task to HEC would therefore
    silently discard most of the label set."""
    as_ss3 = _decode_residue_label("Secondary Structure 3 (CB513)", "dssp3", SS8_LABEL)
    as_ss8 = _decode_residue_label("Secondary Structure 8 (CB513)", "dssp8", SS8_LABEL)
    assert len(as_ss3) == len(as_ss8) == len(SS8_LABEL)
    scorable_ss3 = sum(1 for v in as_ss3 if v >= 0)
    scorable_ss8 = sum(1 for v in as_ss8 if v >= 0)
    # SS8_LABEL holds 3 unassigned `D`s, ignore under both alphabets. Of the 16
    # real states left, HEC can only represent the 7 that are H, E or C.
    assert (scorable_ss3, scorable_ss8) == (7, 16)


def test_disorder_task_name_containing_ss3_still_routes_to_disorder():
    """Regression: "Disorder (NetSurfP-SS3 mask)" contains "ss3", so it used to
    fall into the HEC branch and decode to [] -- every residue dropped, and the
    probe then trained on zero rows."""
    decoded = _decode_residue_label("Disorder (NetSurfP-SS3 mask)", "disorder", "0110")
    assert decoded == [0, 1, 1, 0]


def test_disorder_parses_netsurfp_stringified_float_list():
    """NetSurfP ships the disorder column as "['0.0', '1.0', ...]", not "01"."""
    raw = "['0.0', '0.0', '1.0', '1.0', '0.0']"
    assert _decode_residue_label("Disorder (NetSurfP-SS3 mask)", "disorder", raw) == [
        0,
        0,
        1,
        1,
        0,
    ]


def test_disorder_label_length_matches_sequence():
    """The failure mode was a SHORT label list, which the residue probe silently
    truncates against -- so assert the count, not just the values."""
    raw = str([f"{v}.0" for v in [0, 1, 0, 1, 1, 0, 0]])
    decoded = _decode_residue_label("Disorder (NetSurfP-SS3 mask)", "disorder", raw)
    assert len(decoded) == 7


def test_new_tasks_registered():
    for key in ("ss8", "contact_probe", "go_bp", "go_cc"):
        assert key in TASKS


def test_contact_task_shape():
    cfg = TASKS["contact_probe"]
    assert cfg.problem_type == "contact_prediction"
    assert cfg.main_metric == "P@L/5_long"
    assert cfg.input_map["seq"] == "seq"


@pytest.mark.parametrize("key", ["ss3_casp12", "ss8_cb513", "ss8_ts115"])
def test_ss_heldout_tasks_carry_data_files(key):
    cfg = TASKS[key]
    assert cfg.data_files is not None
    assert cfg.data_files["train"] == "training_hhblits.csv"
    assert cfg.data_files["test"].endswith(".csv")
    assert cfg.label_col in ("dssp3", "dssp8")


def test_heldout_and_go_tasks_are_opt_in():
    """Ten near-duplicate SS sets and two large GO sets must not join a sweep."""
    for key in SS_HELDOUT_TASKS:
        assert key not in DEFAULT_TASKS
    for key in ("go_bp", "go_cc"):
        assert key in MULTILABEL_EXCLUDED_TASKS
        assert key not in DEFAULT_TASKS


def test_contact_and_ss8_are_in_the_broad_sweep():
    assert "contact_probe" in DEFAULT_TASKS
    assert "ss8" in DEFAULT_TASKS


# ---------------------------------------------------------------------------
# Unassigned residues (`D`) must not become a 9th class
# ---------------------------------------------------------------------------
# GleghornLab/SS8 marks unassigned termini `D`: 6.4% of train and 11.5% of test
# residues, 91% of them in terminal runs and so trivially predictable from
# position alone. Scoring them as a real class inflates Accuracy and makes the
# task's 9-class F1_Macro incomparable to the 8-class published Q8 that
# `ss8_cb513` / `ss8_casp12` report. Standard Q8 evaluation masks them out.


def test_ss8_real_states_use_the_published_eight_classes():
    decoded = _decode_residue_label(
        "Secondary Structure 8 (DSSP8)", "labels", "GHIBESTC"
    )
    assert sorted(decoded) == list(range(8))


def test_ss8_class_ids_agree_across_both_sources():
    """`ss8` (GleghornLab) and `ss8_cb513` (proteinea) must assign the same id to
    the same DSSP state, or their F1_Macro values are not comparable."""
    states = "GHIBESTC"
    gleghorn = _decode_residue_label("Secondary Structure 8 (DSSP8)", "labels", states)
    proteinea = _decode_residue_label("Secondary Structure 8 (CB513)", "dssp8", states)
    assert gleghorn == proteinea


def test_ss8_unassigned_residues_become_the_ignore_sentinel():
    decoded = _decode_residue_label("Secondary Structure 8 (DSSP8)", "labels", "DDCHH")
    assert decoded == [-1, -1, 7, 1, 1]


def test_ss8_decode_preserves_length_so_labels_stay_aligned():
    """Dropping unassigned positions instead of marking them would SHIFT every
    later label against its residue embedding."""
    label = "DDDCHHHEEGGITTSSBBCDD"
    decoded = _decode_residue_label("Secondary Structure 8 (DSSP8)", "labels", label)
    assert len(decoded) == len(label)


def test_ignored_residues_are_dropped_without_breaking_alignment():
    """The residue probe must drop sentinel rows from BOTH the embeddings and
    the labels, together."""
    import numpy as np

    from token_classification_probe import drop_ignored_residues

    X = np.arange(12, dtype="float32").reshape(6, 2)
    y = np.array([0, -1, 3, -1, 5, 2], dtype="int64")
    X_kept, y_kept = drop_ignored_residues(X, y)
    assert y_kept.tolist() == [0, 3, 5, 2]
    assert X_kept.tolist() == [[0, 1], [4, 5], [8, 9], [10, 11]]


def test_ss3_unknown_symbol_is_ignored_not_dropped():
    """Same landmine the SS8 branch was fixed for: dropping an unrecognised
    symbol shortens the label list and shifts every later label against its
    residue embedding. No shipped dataset emits one today -- this keeps the two
    alphabet branches from disagreeing about what "unrecognised" means."""
    decoded = _decode_residue_label("Secondary Structure 3 (CB513)", "dssp3", "HHXEC")
    assert len(decoded) == 5
    assert decoded[2] == -1
    assert decoded == [0, 0, -1, 1, 2]
