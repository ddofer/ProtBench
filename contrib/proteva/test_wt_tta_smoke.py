"""Slow CPU smoke: WT-TTT on the real AMPLIFY_120M + real ProteinGym BLAT data.

Marked ``slow`` (skipped by default). Needs the cached AMPLIFY_120M model and
ProteinGym_v1 dataset. Run with::

    CUDA_VISIBLE_DEVICES="" pytest tests/test_wt_tta_smoke.py -m slow
"""

import math

import pytest


@pytest.mark.slow
def test_amplify_blat_tta_pipeline_runs_and_is_finite():
    from wt_tta_smoke import run_blat_smoke

    res = run_blat_smoke(n_variants=8, iters=3, layers=2, seed=0, device="cpu")

    assert res["wt_len"] > 200          # TEM-1 β-lactamase is ~286 aa
    assert res["n_variants"] == 8
    # Pipeline runs on the real model and yields valid Spearman correlations.
    # (That adaptation actually MOVES the embeddings is asserted deterministically
    # in test_wt_tta.py::test_adapt_moves_the_embedding_readout; here, with few
    # variants + few steps, the cosine *ranking* may legitimately tie -> delta 0.)
    assert math.isfinite(res["baseline"]) and -1.0 <= res["baseline"] <= 1.0
    assert math.isfinite(res["tta"]) and -1.0 <= res["tta"] <= 1.0
