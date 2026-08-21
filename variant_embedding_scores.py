"""Embedding readouts for variant effect: RED and edit-span pooling.

Why these exist next to the masked-marginal scorer in ``proteingym_mlm_zeroshot``:
the two task shapes have opposite cost structures.

* **Substitutions amortize.** One masked forward at position *i* yields log-probs
  for every amino acid there, so all ~19 variants at that position are free ->
  ~86k forwards for ProteinGym's 2.47M DMS substitutions. Nothing here can beat
  that, and it is also the more accurate scorer. Use masked-marginal.
* **Indels do not amortize.** Every variant is a different sequence, so the only
  question is forwards *per variant*: strided masked PLL needs ``k``
  (``--indel_pll_passes``, default 32); an embedding readout needs **1**. That is
  ~287k forwards instead of ~9.2M for ProteinGym's DMS indels, with no assay
  skipping.

``red`` is ported from diploid_glm's covariance_pooling branch
(``embedding/covariance_pooling/auxfeat.py``), whose own upstream is a protein
library (flair-bio/proseqo). Numbers there are from DNA tasks and do not transfer;
treat protein accuracy as unmeasured until benchmarked.
"""

from __future__ import annotations

import numpy as np


def red(X: np.ndarray) -> float:
    """Residue Embedding Diversity = 1 - mean pairwise cosine of a sequence's residues.

    ``X`` is one sequence's per-residue embeddings, ``(T, d)``. Range [0, 2]:
    0 = every residue identical, 1 = mutually orthogonal, 2 = antipodal pair.

    O(T*d), not the naive O(T^2*d), via: for unit-normalised rows u_t,
    ``sum_{i,j} <u_i,u_j> == ||sum_t u_t||^2`` and the diagonal contributes
    ``sum_t ||u_t||^2``, so the unit-normalised copy is never materialised.
    ``||S||^2`` is formed in float64 -- RED for two sequences differing at one
    residue is a difference of near-equal O(1) numbers.
    """
    X = np.asarray(X)
    T = X.shape[0]
    if T < 2:
        return 0.0
    n2 = np.einsum("td,td->t", X, X, dtype=np.float64)
    inv = 1.0 / np.maximum(np.sqrt(n2), 1e-30)
    S = inv @ X.astype(np.float64)  # (d,) == sum_t u_t
    amp = (n2 * inv * inv).sum()  # ~T, accumulated rather than assumed
    return float(1.0 - ((S**2).sum() - amp) / (T * (T - 1)))


def edit_span(wt: str, mut: str, pad: int = 0) -> tuple[int, int]:
    """Half-open span of ``mut`` that differs from ``wt``, in MUTANT coordinates.

    ProteinGym stores ``mutant = "N/A"`` for indels, so there is no position to
    pool around until one is derived. Multiple edit regions collapse to the
    enclosing span: pooling wants one contiguous window, and indel assays edit
    one locus in practice.

    A pure deletion leaves no differing residue in the mutant, so the span is
    widened to the single residue at the join -- pooling over an empty span
    would return nothing to compare. Identical sequences return ``(0, 0)``.
    """
    import difflib

    if wt == mut:
        return (0, 0)
    ops = [op for op in difflib.SequenceMatcher(None, wt, mut, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    if not ops:
        return (0, 0)
    start = min(op[3] for op in ops)
    end = max(op[4] for op in ops)
    end = max(end, start + 1)  # deletions have j1 == j2
    return (max(0, start - pad), min(len(mut), end + pad))


def span_pooled_score(
    wt_X: np.ndarray, mut_X: np.ndarray, span: tuple[int, int], metric: str = "cosine"
) -> float:
    """Distance between WT and mutant, pooling only over ``span`` (mutant coordinates).

    Whole-sequence mean pooling dilutes a handful of edited residues across
    hundreds of unchanged ones; the per-residue vectors already carry
    surrounding context through attention, so pooling over the edit alone reads
    the change without the dilution. The WT side pools the same coordinate range
    clipped to its own length, which keeps the two windows aligned for
    substitutions and adjacent for indels.

    ``metric``: ``cosine`` (1 - cos, in [0, 2]) or ``l2``. An empty span means
    "no located edit" and falls back to whole-sequence pooling. Higher = more
    disrupted.
    """
    start, end = span
    if end <= start:
        start, end = 0, max(len(wt_X), len(mut_X))
    mut_win = mut_X[start:end]
    wt_win = wt_X[min(start, len(wt_X)) : min(end, len(wt_X))]
    if len(mut_win) == 0 or len(wt_win) == 0:
        return 0.0

    a = wt_win.astype(np.float64).mean(axis=0)
    b = mut_win.astype(np.float64).mean(axis=0)
    if metric == "l2":
        return float(np.linalg.norm(a - b))
    if metric != "cosine":
        raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-30:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def score_indel_variants(
    wt: str,
    variants: list[str],
    embedder,
    *,
    arm: str = "span",
    metric: str = "cosine",
    model_window: int | None = None,
) -> list[float | None]:
    """Score indel variants with ONE forward per sequence. Higher = more disrupted.

    ``embedder(seqs) -> list of (T_i, d) arrays`` is injected; in the benchmark it
    is ``token_classification_probe.iter_residue_embeddings``, in tests a fake.
    The WT is embedded once alongside the variants, not once per variant.

    Arms:
      * ``span`` -- pool over the derived edit span only (default), see
        ``span_pooled_score``;
      * ``red``  -- ``RED(mut) - RED(wt)``, whole-sequence residue diversity.

    Returns one score per variant, aligned with ``variants``; ``None`` where a
    sequence exceeds ``model_window``, matching the skip contract of
    ``proteingym_mlm_zeroshot.strided_masked_pll_table`` so the two are drop-in
    swappable.
    """
    if arm not in ("span", "red"):
        raise ValueError(f"arm must be 'span' or 'red', got {arm!r}")

    fits = [model_window is None or len(v) <= model_window for v in variants]
    if model_window is not None and len(wt) > model_window:
        return [None] * len(variants)

    to_embed = [wt] + [v for v, ok in zip(variants, fits) if ok]
    embedded = list(embedder(to_embed))
    wt_X, rest = embedded[0], embedded[1:]

    wt_red = red(wt_X) if arm == "red" else None
    scored = iter(rest)
    out: list[float | None] = []
    for variant, ok in zip(variants, fits):
        if not ok:
            out.append(None)
            continue
        mut_X = next(scored)
        if arm == "red":
            out.append(float(red(mut_X) - wt_red))
        else:
            out.append(span_pooled_score(wt_X, mut_X, edit_span(wt, variant), metric=metric))
    return out
