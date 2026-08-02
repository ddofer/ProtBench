"""Percentile bootstrap confidence intervals over per-item scores.

Bootstrapping is exact only where the metric is a mean over independent
per-item values -- then resampling items with replacement gives the sampling
distribution directly, with no refitting. Retrieval is the clean case:
``per_query_metrics`` reduces a ranking to one value per query, and
``boot_ci`` resamples those.

For probe metrics that are not simple means (F1, AUC), see ``_boot_ci`` in
``protein_benchmark_suite.py``, which resamples predictions and recomputes the
metric on each draw.

A note on reading the output: overlapping *marginal* intervals for two methods
do NOT mean the difference is unresolved, because the same queries are scored
by both. The interval on the per-query *difference* is the one that settles a
comparison, and it is much tighter.

ponytail: trimmed from ProtSent's bootstrap_ci.py to the two functions
hmmer_baseline actually calls. The dropped half was a SCOPe-only CLI with
hardcoded model paths.
"""

from __future__ import annotations

import numpy as np

N_BOOT = 10_000
SEED = 0


def per_query_metrics(ranking: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """hit@1, hit@10, hit@30 and average precision for each query.

    `ranking[q]` is the gallery order for query q, self already removed.
    Queries with no achievable positive get 0 and are marked ineligible.
    """
    n = len(labels)
    uniq, cnt = np.unique(labels, return_counts=True)
    fam = dict(zip(uniq.tolist(), cnt.tolist()))

    out = {k: np.zeros(n) for k in ("hit1", "hit10", "hit30", "ap")}
    eligible = np.zeros(n, dtype=bool)
    for q in range(n):
        n_rel = fam[labels[q]] - 1
        if n_rel <= 0:
            continue
        eligible[q] = True
        rel = labels[ranking[q]] == labels[q]
        out["hit1"][q] = float(rel[:1].any())
        out["hit10"][q] = float(rel[:10].any())
        out["hit30"][q] = float(rel[:30].any())
        hr = np.flatnonzero(rel) + 1
        if hr.size:
            out["ap"][q] = float(np.sum(np.arange(1, hr.size + 1) / hr) / n_rel)
    out["eligible"] = eligible
    return out


def boot_ci(
    values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap over the query axis."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )
