"""cath_levels must build the same cache key the suite writes, or it never gets a hit."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def _suite_cfg_key(probe_embed_mode, l2_normalize_embeddings, max_length, amp_dtype):
    """The suite's _cfg_key, restated. Kept in sync by test_suite_cfg_key_format_is_unchanged."""
    return (f"{probe_embed_mode}|l2={int(bool(l2_normalize_embeddings))}"
            f"|ml={max_length}|dt={amp_dtype}")


def test_suite_cfg_key_format_is_unchanged():
    """Pin the suite's key expression, since the agreement test below restates it."""
    src = (ROOT / "protein_benchmark_suite.py").read_text()
    assert '''f"{probe_embed_mode}|l2={int(bool(l2_normalize_embeddings))}"''' in src
    assert '''f"|ml={max_length}|dt={amp_dtype}"''' in src


def test_cath_levels_key_matches_the_suite_at_default_precision():
    """--amp_dtype fp32 is the default and leaves amp_dtype as None, not a torch dtype.

    The suite interpolates that variable straight into the key, so it writes "dt=None". cath_levels
    hardcoded "dt=fp32", which reads as the same thing to a person and as a different key to the
    cache: a guaranteed miss on every run, silently re-embedding all 69,605 CATH lookup sequences
    while the comment above it claimed the keys matched.
    """
    cath_src = (ROOT / "cath_levels.py").read_text()
    literal = re.search(r'cfg_key = f"([^"]+)"', cath_src).group(1)
    cath_key = literal.replace("{max_length}", "1024")

    # amp_dtype is None at the fp32 default -- that is the whole point of the bug.
    assert cath_key == _suite_cfg_key("trunk", False, 1024, None)
