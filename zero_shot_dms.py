"""Zero-shot variant-effect (deep-mutational-scan) scoring for protein MLMs.

Implements the ESM-1v / AMPLIFY *masked-marginal* pseudo-likelihood protocol: for
a single-mutant variant (wild-type residue ``r`` -> mutant ``m`` at position
``i``) the predicted effect is the log-likelihood ratio

    score = log p(m | x_masked_at_i) - log p(r | x_masked_at_i)

where ``x`` is the wild-type sequence with position ``i`` replaced by ``[MASK]``
and the probabilities come from the MLM head softmax over the 20 amino-acid token
ids at position ``i``. Multi-mutant variants sum the per-position LLRs (each
mutated position masked independently over the wild-type background — the
standard masked-marginal approximation). The per-dataset metric is
Spearman(predicted_LLR, measured_fitness).

This is the *unsupervised* counterpart to the supervised frozen-embedding probes
(``variant_effect`` / ``beta_lactamase_peer`` in ``benchmark_tasks.py``): it asks
whether continued pretraining sharpens the model's own sequence likelihood on
GB1 / beta-lactamase even where the supervised probe regresses.

Scoring is decoupled from the architecture through :class:`MaskedLMScorer`, which
carries the tokenizer-derived amino-acid <-> token-id map, the special-token ids,
and a masked forward. Two adapters are provided so the Proteva continued-pretrain
checkpoints and the AMPLIFY-120M reference are scored apples-to-apples with one
implementation of the protocol. The amino-acid <-> token-id map is taken from the
project tokenizer (via :meth:`ProteinPackedCollator.canon_token_ids`), never the
hardcoded ``CANON_TOKEN_IDS`` (which disagrees with the model vocab for 15/20
AAs); AMPLIFY shares that exact map.

CLI::

    python -m plm.bench.zero_shot_dms --ckpt DIR --task {gb1,beta_lactamase} \
        [--device cuda] [--batch-size 64] [--max-variants N]
    python -m plm.bench.zero_shot_dms --amplify --task beta_lactamase [...]

Both downstream datasets store *full variant sequences* of fixed length (no
explicit wt+mutation list), so the wild-type is derived per dataset as the
per-position consensus residue and each variant's mutation list is obtained by
diffing against it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# Canonical 20 amino-acid letters (matches AA_ORDER used by the aux losses).
_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"

# Task aliases -> (benchmark_tasks key, sequence column, fitness column).
_TASK_SPECS: dict[str, tuple[str, str, str]] = {
    "gb1": ("variant_effect", "seq", "label"),
    "variant_effect": ("variant_effect", "seq", "label"),
    "beta_lactamase": ("beta_lactamase_peer", "protein", "label"),
    "beta_lactamase_peer": ("beta_lactamase_peer", "protein", "label"),
}


@dataclass(frozen=True)
class DMSVariant:
    """A single DMS row: wild-type background, mutation list, measured fitness.

    Attributes:
        wt_sequence: The wild-type amino-acid sequence (no special tokens).
        mutations: List of ``(position, wt_aa, mut_aa)`` tuples; ``position`` is
            0-based into ``wt_sequence``. An empty list is the wild type itself
            (predicted LLR 0).
        fitness: The measured fitness / activity label (the Spearman target).
    """

    wt_sequence: str
    mutations: list[tuple[int, str, str]]
    fitness: float


@dataclass
class DMSResult:
    """Outcome of scoring a DMS dataset.

    Attributes:
        spearman: Spearman rank correlation between predicted LLR and measured
            fitness (``nan`` when fewer than two scorable variants).
        n: Number of variants scored.
        predictions: Per-variant predicted LLR (populated only when
            ``return_predictions=True``).
    """

    spearman: float
    n: int
    predictions: list[float] = field(default_factory=list)


class MaskedLMScorer(Protocol):
    """Architecture-agnostic interface the masked-marginal scorer relies on.

    An adapter exposes the special-token ids, an amino-acid -> token-id map, a
    tokenizer (sequence -> bracketed token ids), and a batched masked forward
    returning per-token logits over the model vocab.
    """

    mask_id: int
    pad_id: int
    aa_to_id: dict[str, int]
    device: torch.device

    def tokenize(self, sequence: str) -> np.ndarray:
        """Return the bracketed (BOS + residues + EOS) int token ids."""
        ...

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return ``(B, T, vocab)`` logits for a padded ``(B, T)`` id batch."""
        ...


def aa_token_id_lookup(collator) -> dict[str, int]:
    """Build an amino-acid-letter -> model-token-id map from the collator.

    Uses :meth:`ProteinPackedCollator.canon_token_ids` (AA_ORDER -> tokenizer id),
    the model's authoritative vocab, rather than a hardcoded table.

    Args:
        collator: A :class:`~plm.hf.collator.ProteinPackedCollator` (loaded or
            lazily loadable).

    Returns:
        Mapping from each of the 20 canonical AA letters to its token id.
    """
    canon = collator.canon_token_ids()
    return {aa: int(tid) for aa, tid in zip(_AA_LETTERS, canon)}


def derive_mutations(
    wt_sequence: str, variant_sequence: str
) -> list[tuple[int, str, str]]:
    """Diff a variant sequence against the wild type into a mutation list.

    Args:
        wt_sequence: The wild-type sequence.
        variant_sequence: A variant sequence of the same length.

    Returns:
        List of ``(position, wt_aa, mut_aa)`` for every differing position
        (0-based, ascending).

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(wt_sequence) != len(variant_sequence):
        raise ValueError(
            "derive_mutations requires equal-length sequences; got "
            f"{len(wt_sequence)} vs {len(variant_sequence)}"
        )
    return [
        (i, w, m)
        for i, (w, m) in enumerate(zip(wt_sequence, variant_sequence))
        if w != m
    ]


def consensus_wild_type(sequences: list[str]) -> str:
    """Derive a wild-type sequence as the per-position consensus residue.

    Both downstream DMS datasets store fixed-length variant sequences without an
    explicit reference; the most common residue at each position recovers the
    wild type (mutational libraries leave most positions untouched).

    Args:
        sequences: Equal-length variant sequences.

    Returns:
        The consensus sequence.

    Raises:
        ValueError: If ``sequences`` is empty or not all equal length.
    """
    if not sequences:
        raise ValueError("consensus_wild_type received no sequences")
    length = len(sequences[0])
    if any(len(s) != length for s in sequences):
        raise ValueError("consensus_wild_type requires equal-length sequences")
    return "".join(Counter(col).most_common(1)[0][0] for col in zip(*sequences))


@torch.no_grad()
def _masked_aa_logprobs(
    scorer: MaskedLMScorer,
    wt_token_ids: np.ndarray,
    positions: list[int],
    aa_ids: torch.Tensor,
    *,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Run masked forwards and return per-position AA log-probabilities.

    For each residue position ``p`` (0-based) a copy of the wild-type token
    sequence has token index ``p + 1`` (BOS at index 0) set to ``[MASK]``; the
    softmax over the 20 AA token ids at that index is returned as log-probs.

    Args:
        scorer: The model adapter (masked forward + special ids).
        wt_token_ids: 1D ``(T,)`` int array of the bracketed wild-type tokens.
        positions: Residue positions (0-based) to mask.
        aa_ids: ``(20,)`` long tensor of AA token ids (column order = AA_LETTERS).
        batch_size: Number of masked positions per forward.

    Returns:
        Mapping ``position -> (20,) float tensor`` of AA log-probs on CPU.
    """
    base = torch.as_tensor(wt_token_ids, dtype=torch.long)
    aa_ids = aa_ids.to(scorer.device)
    out: dict[int, torch.Tensor] = {}

    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        token_idx = torch.tensor([p + 1 for p in chunk], dtype=torch.long)
        ids = base.unsqueeze(0).repeat(len(chunk), 1).clone()
        ids[torch.arange(len(chunk)), token_idx] = scorer.mask_id
        logits = scorer.logits(ids.to(scorer.device))
        rows = torch.arange(len(chunk), device=logits.device)
        masked_logits = logits[rows, token_idx.to(logits.device)].float()
        logp = F.log_softmax(masked_logits, dim=-1)[:, aa_ids].cpu()
        for j, p in enumerate(chunk):
            out[p] = logp[j]
    return out


def score_variants(
    scorer: MaskedLMScorer,
    variants: list[DMSVariant],
    *,
    batch_size: int = 64,
    return_predictions: bool = False,
) -> DMSResult:
    """Score DMS variants by summed masked-marginal LLR; report Spearman.

    Masked forwards are deduplicated across variants: one forward per unique
    ``(wt_sequence, position)`` pair (the 20-way AA log-prob vector at that
    masked position is reused by every variant touching it).

    Args:
        scorer: A :class:`MaskedLMScorer` adapter wrapping the model + tokenizer.
        variants: The DMS rows to score.
        batch_size: Masked positions per forward pass.
        return_predictions: If True, populate ``DMSResult.predictions``.

    Returns:
        A :class:`DMSResult` with the Spearman correlation and variant count.

    Raises:
        ValueError: If ``variants`` is empty.
    """
    if not variants:
        raise ValueError("score_variants received no variants")

    aa_col = {aa: c for c, aa in enumerate(_AA_LETTERS)}
    aa_ids = torch.tensor(
        [scorer.aa_to_id[aa] for aa in _AA_LETTERS], dtype=torch.long
    )

    # Group required (wt, position) lookups so each distinct wild-type is
    # tokenized once and each masked position is forwarded once.
    wt_to_positions: dict[str, set[int]] = {}
    for v in variants:
        wt_to_positions.setdefault(v.wt_sequence, set()).update(
            pos for pos, _, _ in v.mutations
        )

    logprob_cache: dict[str, dict[int, torch.Tensor]] = {
        wt_seq: _masked_aa_logprobs(
            scorer,
            scorer.tokenize(wt_seq),
            sorted(positions),
            aa_ids,
            batch_size=batch_size,
        )
        for wt_seq, positions in wt_to_positions.items()
    }

    predictions: list[float] = []
    fitness: list[float] = []
    for v in variants:
        cache = logprob_cache[v.wt_sequence]
        llr = sum(
            float(cache[pos][aa_col[mut_aa]] - cache[pos][aa_col[wt_aa]])
            for pos, wt_aa, mut_aa in v.mutations
        )
        predictions.append(llr)
        fitness.append(float(v.fitness))

    n = len(predictions)
    rho = float("nan") if n < 2 else float(spearmanr(predictions, fitness).statistic)
    return DMSResult(
        spearman=rho, n=n, predictions=predictions if return_predictions else []
    )


# --------------------------------------------------------------------------- #
# Model adapters.
# --------------------------------------------------------------------------- #
@dataclass
class _CallableScorer:
    """Concrete :class:`MaskedLMScorer` built from primitive parts.

    Args:
        mask_id: ``[MASK]`` token id.
        pad_id: Pad token id.
        aa_to_id: Amino-acid-letter -> token-id map.
        device: Torch device the model lives on.
        tokenize_fn: Sequence -> bracketed int token-id array.
        logits_fn: Padded ``(B, T)`` id batch -> ``(B, T, vocab)`` logits.
    """

    mask_id: int
    pad_id: int
    aa_to_id: dict[str, int]
    device: torch.device
    tokenize_fn: Callable[[str], np.ndarray]
    logits_fn: Callable[[torch.Tensor], torch.Tensor]

    def tokenize(self, sequence: str) -> np.ndarray:
        ids = self.tokenize_fn(sequence)
        if ids is None:
            raise ValueError(f"could not tokenize sequence of length {len(sequence)}")
        return np.asarray(ids, dtype=np.int64)

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logits_fn(input_ids)


def build_proteva_scorer(ckpt_dir: str, device: str | torch.device) -> MaskedLMScorer:
    """Build a scorer for a Proteva continued-pretrain checkpoint.

    Args:
        ckpt_dir: HF checkpoint directory (``model.safetensors`` + ``config.json``).
        device: Torch device for the forward pass.

    Returns:
        A :class:`MaskedLMScorer` over the loaded model + project tokenizer.
    """
    from plm.hf.collator import ProteinPackedCollator
    from plm.hf.modeling import ProtevaForPretraining

    device = torch.device(device)
    model = ProtevaForPretraining.from_pretrained(ckpt_dir).to(device).eval()
    collator = ProteinPackedCollator()
    collator._ensure_ready()

    @torch.no_grad()
    def _logits(input_ids: torch.Tensor) -> torch.Tensor:
        attn = (input_ids != collator._pad_id).long()
        try:
            out = model(input_ids, attention_mask=attn)
        except TypeError:
            out = model(input_ids)
        logits = getattr(out, "logits", None)
        return logits if logits is not None else out[0]

    return _CallableScorer(
        mask_id=int(collator._mask_id),
        pad_id=int(collator._pad_id),
        aa_to_id=aa_token_id_lookup(collator),
        device=device,
        tokenize_fn=lambda seq: collator._row_to_ids({"sequence": seq}),
        logits_fn=_logits,
    )


def build_amplify_scorer(device: str | torch.device) -> MaskedLMScorer:
    """Build a scorer for the AMPLIFY-120M reference MLM.

    AMPLIFY shares the project's amino-acid <-> token-id map (the Proteva
    tokenizer was derived from it) but uses an *additive* attention mask and its
    own remote-code forward, so it needs a dedicated adapter.

    Args:
        device: Torch device for the forward pass.

    Returns:
        A :class:`MaskedLMScorer` over AMPLIFY-120M + its tokenizer.
    """
    from plm.amplify_loader import load_amplify_120m

    device = torch.device(device)
    loaded = load_amplify_120m(device=device, dtype=torch.float32)
    tok = loaded.tokenizer
    vocab = tok.get_vocab()
    aa_to_id = {aa: int(vocab[aa]) for aa in _AA_LETTERS}
    pad_id = int(tok.pad_token_id or 0)
    mask_id = int(tok.mask_token_id)

    @torch.no_grad()
    def _logits(input_ids: torch.Tensor) -> torch.Tensor:
        # AMPLIFY expects an ADDITIVE mask (0.0 valid / -inf pad), not 0/1 ints.
        add_mask = torch.where(
            input_ids != pad_id, 0.0, float("-inf")
        ).to(input_ids.device)
        out = loaded.model(input_ids, attention_mask=add_mask)
        logits = getattr(out, "logits", None)
        return logits if logits is not None else out[0]

    def _tokenize(seq: str) -> np.ndarray:
        return np.asarray(tok(seq)["input_ids"], dtype=np.int64)

    return _CallableScorer(
        mask_id=mask_id,
        pad_id=pad_id,
        aa_to_id=aa_to_id,
        device=device,
        tokenize_fn=_tokenize,
        logits_fn=_logits,
    )


# --------------------------------------------------------------------------- #
# Dataset loading.
# --------------------------------------------------------------------------- #
def load_task_variants(task: str, max_variants: int | None = None) -> list[DMSVariant]:
    """Load GB1 / beta-lactamase variant+fitness rows as :class:`DMSVariant`.

    Both datasets store full fixed-length variant sequences; the wild type is
    derived as the per-position consensus and each variant is diffed against it.

    Args:
        task: One of ``gb1`` / ``variant_effect`` / ``beta_lactamase`` /
            ``beta_lactamase_peer``.
        max_variants: Optional cap (first ``N`` rows after concatenating splits).

    Returns:
        The DMS variants for the task.

    Raises:
        ValueError: If ``task`` is unknown or no rows load.
    """
    if task not in _TASK_SPECS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(_TASK_SPECS)}")
    from datasets import load_dataset

    from plm.bench.benchmark_tasks import TASKS

    task_key, seq_col, label_col = _TASK_SPECS[task]
    cfg = TASKS[task_key]
    ds = load_dataset(cfg.dataset)

    sequences: list[str] = []
    labels: list[float] = []
    for split in ds.keys():
        split_ds = ds[split]
        cols = split_ds.column_names
        if seq_col not in cols or label_col not in cols:
            continue
        sequences.extend(str(s) for s in split_ds[seq_col])
        labels.extend(float(x) for x in split_ds[label_col])

    if not sequences:
        raise ValueError(f"no rows loaded for task {task!r} (dataset {cfg.dataset})")

    wt = consensus_wild_type(sequences)
    variants = [
        DMSVariant(wt_sequence=wt, mutations=derive_mutations(wt, seq), fitness=label)
        for seq, label in zip(sequences, labels)
    ]
    return variants if max_variants is None else variants[:max_variants]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: score one task with a checkpoint and print Spearman."""
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="Proteva HF checkpoint directory.")
    src.add_argument(
        "--amplify",
        action="store_true",
        help="Score the AMPLIFY-120M reference instead of a Proteva checkpoint.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(_TASK_SPECS),
        help="DMS task (gb1 / beta_lactamase).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Optional cap on the number of variants scored.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    logger.info("Loading variants for task %s", args.task)
    variants = load_task_variants(args.task, max_variants=args.max_variants)
    n_mut = sum(len(v.mutations) for v in variants)
    logger.info(
        "Loaded %d variants (avg %.2f mutations/variant)",
        len(variants),
        n_mut / max(len(variants), 1),
    )

    if args.amplify:
        logger.info("Loading AMPLIFY-120M reference")
        scorer = build_amplify_scorer(args.device)
        model_tag = "amplify_120m"
    else:
        logger.info("Loading Proteva checkpoint from %s", args.ckpt)
        scorer = build_proteva_scorer(args.ckpt, args.device)
        model_tag = args.ckpt

    result = score_variants(scorer, variants, batch_size=args.batch_size)
    # Single clean stdout line, parseable by callers.
    print(
        f"{args.task}\tmodel={model_tag}\tspearman={result.spearman:.6f}\tn={result.n}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
