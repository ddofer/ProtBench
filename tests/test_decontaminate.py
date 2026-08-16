"""Corpus decontamination against benchmark test sets.

Offline: MMseqs2 is a system binary and is not required here — these cover the
FASTA parsing and threshold logic, which is where a mistake silently deletes
good data or silently keeps leaked data.
"""

from pathlib import Path

import pytest

from decontaminate import _iter_clean, parse_hits, read_fasta

# test_id, corpus_id, fident, qcov, tcov, evalue
HITS = (
    "t1\tCLEAR\t0.95\t0.99\t0.99\t1e-40\n"  # near-identical, full length
    "t2\tPARTIAL\t0.91\t0.20\t0.95\t1e-05\n"  # identical over 20% of the query
    "t3\tREMOTE\t0.10\t0.99\t0.99\t1e-02\n"  # full length, 10% identity
    "t4\tCLEAR\t0.40\t0.99\t0.99\t1e-10\n"  # weaker second hit on CLEAR
)


@pytest.fixture
def hits_tsv(tmp_path) -> Path:
    path = tmp_path / "hits.tsv"
    path.write_text(HITS)
    return path


def test_reads_multiline_and_blank_separated_fasta(tmp_path):
    fasta = tmp_path / "c.fasta"
    fasta.write_text(">sp|P1|FOO description here\nMKTA\nYIAK\n\n>P2\nWWWW\n")
    records = read_fasta(fasta)
    assert records == [("sp|P1|FOO", "MKTAYIAK"), ("P2", "WWWW")]


def test_identifier_is_the_header_up_to_whitespace(tmp_path):
    """MMseqs2 reports that prefix in its query/target columns, so the stop list
    has to key on the same thing or nothing will match."""
    fasta = tmp_path / "c.fasta"
    fasta.write_text(">A1 some long free text\nMK\n")
    assert read_fasta(fasta)[0][0] == "A1"


def test_default_thresholds_flag_only_the_real_contamination(hits_tsv):
    assert set(parse_hits(hits_tsv, min_seq_id=0.3, coverage=0.8)) == {"CLEAR"}


def test_high_identity_over_short_alignment_is_not_contamination(hits_tsv):
    """A shared 20% window is a domain match, not the same protein. Calling it
    contamination would delete unrelated sequences from the corpus."""
    assert "PARTIAL" not in parse_hits(hits_tsv, min_seq_id=0.3, coverage=0.8)


def test_lowering_identity_catches_remote_homologues(hits_tsv):
    """Structural benchmarks leak through remote homology, so the threshold has
    to be able to go below the usual 30%."""
    assert set(parse_hits(hits_tsv, min_seq_id=0.05, coverage=0.8)) == {
        "CLEAR",
        "REMOTE",
    }


def test_lowering_coverage_catches_partial_matches(hits_tsv):
    assert set(parse_hits(hits_tsv, min_seq_id=0.3, coverage=0.1)) == {
        "CLEAR",
        "PARTIAL",
    }


def test_strongest_match_is_reported_per_corpus_entry(hits_tsv):
    """CLEAR is hit twice; the report should name the worst offender."""
    assert parse_hits(hits_tsv, 0.3, 0.8)["CLEAR"] == ("t1", 0.95)


def test_coverage_uses_the_weaker_of_query_and_target(tmp_path):
    """A short test protein fully contained in a long corpus protein has qcov 1.0
    and tcov near 0. Taking only qcov would flag every long sequence that happens
    to contain a common domain."""
    path = tmp_path / "h.tsv"
    path.write_text("t1\tLONG\t0.99\t1.00\t0.05\t1e-30\n")
    assert parse_hits(path, 0.3, 0.8) == {}


def test_filtering_keeps_everything_not_on_the_stop_list():
    records = [("A", "MK"), ("B", "WW"), ("C", "GG")]
    assert list(_iter_clean(records, {"B"})) == [("A", "MK"), ("C", "GG")]


def test_empty_stop_list_keeps_the_whole_corpus():
    records = [("A", "MK"), ("B", "WW")]
    assert list(_iter_clean(records, set())) == records


def test_malformed_rows_are_skipped_not_crashed(tmp_path):
    path = tmp_path / "h.tsv"
    path.write_text("garbage\nt1\tOK\t0.95\t0.99\t0.99\t1e-40\n\n")
    assert set(parse_hits(path, 0.3, 0.8)) == {"OK"}


def test_selfcheck_runs():
    from decontaminate import _selfcheck

    _selfcheck()
