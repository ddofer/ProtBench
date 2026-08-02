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

    python -m zero_shot_dms --ckpt DIR --task {gb1,beta_lactamase} \
        [--device cuda] [--batch-size 64] [--max-variants N]
    python -m zero_shot_dms --amplify --task beta_lactamase [...]

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

    from benchmark_tasks import TASKS

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


# --------------------------------------------------------------------------- #
# ProteinGym multi-assay loading (the locally cached DMS panel).
# --------------------------------------------------------------------------- #
# ProteinGym v1 ships ~217 substitution DMS assays as a sharded parquet dataset
# (one row per variant) with an *explicit* wild-type ``target_seq`` and a
# ``mutant`` field of the form ``W<pos>M`` (1-based position), colon-separated
# for multi-mutant variants. This is richer than the GB1 / beta-lactamase TAPE
# datasets (which carry only full variant sequences and need a consensus WT), so
# the loader parses the mutation list directly rather than diffing.
_PROTEINGYM_REPO = "OATML-Markslab/ProteinGym_v1"
_PROTEINGYM_SUBSTITUTIONS_DIR = "DMS_substitutions"
_PROTEINGYM_PREFIX = "proteingym:"

# AMPLIFY-120M masked-marginal Spearman references (fixed external facts; only
# the assays we have a measured AMPLIFY number for). beta_lactamase is the
# audited n=5198 figure carried in project memory.
AMPLIFY_REFERENCES: dict[str, float] = {
    "beta_lactamase": 0.347,
}


def _proteingym_snapshot_dir() -> str | None:
    """Return the locally cached ProteinGym snapshot dir, or ``None`` if absent.

    Resolves the cache *offline* (``local_files_only=True``) so no network call
    or download is ever made — the panel only scores data already on disk.
    """
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            _PROTEINGYM_REPO, repo_type="dataset", local_files_only=True
        )
    except Exception:  # not cached / hub_hub unavailable
        return None


def _proteingym_shards() -> list[str]:
    """Return the cached ProteinGym DMS_substitutions parquet shard paths."""
    import glob
    import os

    snap = _proteingym_snapshot_dir()
    if snap is None:
        return []
    sub = os.path.join(snap, _PROTEINGYM_SUBSTITUTIONS_DIR)
    return sorted(glob.glob(os.path.join(sub, "*.parquet")))


def proteingym_assay_ids() -> list[str]:
    """Enumerate the DMS_id of every locally cached ProteinGym substitution assay.

    Returns:
        Sorted ``DMS_id`` strings, or an empty list when ProteinGym is not
        cached. Cheap: reads only the ``DMS_id`` column across shards.
    """
    shards = _proteingym_shards()
    if not shards:
        return []
    import pyarrow.parquet as pq

    ids: set[str] = set()
    for shard in shards:
        table = pq.read_table(shard, columns=["DMS_id"])
        ids.update(str(x) for x in table.column("DMS_id").to_pylist())
    return sorted(ids)


def parse_proteingym_mutant(
    mutant: str, target_seq: str
) -> list[tuple[int, str, str]]:
    """Parse a ProteinGym ``mutant`` field into a 0-based mutation list.

    Args:
        mutant: ``W<pos>M`` (1-based ``pos``), colon-separated for multi-mutants
            (e.g. ``"D6A:E32F"``). The synonymous wild-type sentinel ``"WT"`` (and
            an empty string) parse to an empty list.
        target_seq: The wild-type sequence, used to validate the listed WT residue.

    Returns:
        List of ``(position, wt_aa, mut_aa)`` (0-based ``position``, ascending).

    Raises:
        ValueError: If a token is malformed or its WT residue disagrees with
            ``target_seq`` at the given position.
    """
    mutant = mutant.strip()
    if not mutant or mutant.upper() == "WT":
        return []
    muts: list[tuple[int, str, str]] = []
    for token in mutant.split(":"):
        token = token.strip()
        if len(token) < 3:
            raise ValueError(f"malformed ProteinGym mutant token {token!r}")
        wt_aa, mut_aa, pos_str = token[0], token[-1], token[1:-1]
        try:
            pos1 = int(pos_str)
        except ValueError as exc:
            raise ValueError(
                f"malformed ProteinGym mutant token {token!r}"
            ) from exc
        pos0 = pos1 - 1  # ProteinGym positions are 1-based.
        if not (0 <= pos0 < len(target_seq)):
            raise ValueError(
                f"mutant {token!r} position {pos1} out of range for "
                f"target_seq of length {len(target_seq)}"
            )
        if target_seq[pos0] != wt_aa:
            raise ValueError(
                f"mutant {token!r} WT residue {wt_aa!r} disagrees with "
                f"target_seq[{pos0}]={target_seq[pos0]!r}"
            )
        muts.append((pos0, wt_aa, mut_aa))
    muts.sort()
    return muts


def load_proteingym_assay(
    dms_id: str, max_variants: int | None = None
) -> list[DMSVariant]:
    """Load one cached ProteinGym substitution assay as :class:`DMSVariant` rows.

    Args:
        dms_id: The assay ``DMS_id`` (e.g. ``"BLAT_ECOLX_Stiffler_2015"``).
        max_variants: Optional cap on the number of variants returned.

    Returns:
        The DMS variants for the assay (explicit WT from ``target_seq``).

    Raises:
        ValueError: If ProteinGym is not cached or ``dms_id`` is not present.
    """
    shards = _proteingym_shards()
    if not shards:
        raise ValueError(
            "ProteinGym is not cached locally; cannot load assay "
            f"{dms_id!r} (run the benchmark suite once to populate the cache)"
        )
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    variants: list[DMSVariant] = []
    for shard in shards:
        table = pq.read_table(
            shard, columns=["DMS_id", "target_seq", "mutant", "DMS_score"]
        )
        mask = pc.equal(table.column("DMS_id"), dms_id)
        sub = table.filter(mask)
        if sub.num_rows == 0:
            continue
        data = sub.to_pydict()
        for target_seq, mutant, score in zip(
            data["target_seq"], data["mutant"], data["DMS_score"]
        ):
            variants.append(
                DMSVariant(
                    wt_sequence=str(target_seq),
                    mutations=parse_proteingym_mutant(str(mutant), str(target_seq)),
                    fitness=float(score),
                )
            )
            if max_variants is not None and len(variants) >= max_variants:
                return variants
    if not variants:
        raise ValueError(f"no ProteinGym rows for DMS_id {dms_id!r}")
    return variants


# --------------------------------------------------------------------------- #
# Task registry + panel.
# --------------------------------------------------------------------------- #
def available_tasks(include_proteingym: bool = True) -> list[str]:
    """List every DMS task that can be scored from *locally cached* data.

    Always includes the curated single-protein tasks (``gb1``,
    ``beta_lactamase``); appends one ``proteingym:<DMS_id>`` entry per cached
    ProteinGym substitution assay when present.

    Args:
        include_proteingym: When False, skip the ProteinGym panel (the curated
            two-protein set only).

    Returns:
        Sorted, de-duplicated task names.
    """
    tasks = ["beta_lactamase", "gb1"]
    if include_proteingym:
        tasks.extend(f"{_PROTEINGYM_PREFIX}{a}" for a in proteingym_assay_ids())
    return tasks


def load_variants_for_task(
    task: str, max_variants: int | None = None
) -> list[DMSVariant]:
    """Dispatch a task name to its variant loader (curated or ProteinGym).

    Args:
        task: ``gb1`` / ``beta_lactamase`` (curated) or ``proteingym:<DMS_id>``.
        max_variants: Optional cap.

    Returns:
        The DMS variants for the task.
    """
    if task.startswith(_PROTEINGYM_PREFIX):
        return load_proteingym_assay(
            task[len(_PROTEINGYM_PREFIX) :], max_variants=max_variants
        )
    return load_task_variants(task, max_variants=max_variants)


def amplify_reference_for(task: str) -> float | None:
    """Return the AMPLIFY-120M Spearman reference for a task, if known."""
    if task.startswith(_PROTEINGYM_PREFIX):
        return AMPLIFY_REFERENCES.get(task[len(_PROTEINGYM_PREFIX) :])
    return AMPLIFY_REFERENCES.get(task)


def score_panel(
    scorer: MaskedLMScorer,
    tasks: list[str],
    *,
    batch_size: int = 64,
    max_variants: int | None = None,
    on_result: Callable[[str, DMSResult], None] | None = None,
) -> dict[str, DMSResult]:
    """Score every task in ``tasks`` with one scorer; return per-task results.

    A failing assay (load or scoring error) is logged and skipped rather than
    aborting the whole panel, so one malformed assay never sinks a run.

    Args:
        scorer: The model adapter.
        tasks: Task names (see :func:`available_tasks`).
        batch_size: Masked positions per forward.
        max_variants: Optional per-assay cap (handy for smoke runs).
        on_result: Optional callback invoked as ``(task, result)`` after each
            assay (e.g. to stream progress).

    Returns:
        Mapping ``task -> DMSResult`` for every assay that scored successfully.
    """
    results: dict[str, DMSResult] = {}
    for task in tasks:
        try:
            variants = load_variants_for_task(task, max_variants=max_variants)
            result = score_variants(scorer, variants, batch_size=batch_size)
        except Exception:  # noqa: BLE001 - one bad assay must not sink the panel
            logger.exception("skipping task %s (load/score failed)", task)
            continue
        results[task] = result
        if on_result is not None:
            on_result(task, result)
    return results


def build_panel_artifact(
    model_id: str,
    results: dict[str, DMSResult],
    *,
    is_amplify: bool = False,
) -> dict:
    """Assemble the durable JSON-serialisable panel artifact.

    Schema (the audit's missing artifact)::

        {
          "model_id": str,           # ckpt dir or "amplify_120m"
          "is_amplify": bool,
          "n_assays": int,
          "mean_spearman": float,    # unweighted mean over scored assays
          "assays": [str, ...],      # the task names scored (sorted)
          "per_assay": {             # one entry per assay
             task: {"spearman": float, "n": int,
                    "amplify_reference": float | None}
          }
        }

    Args:
        model_id: Checkpoint dir (or ``amplify_120m``).
        results: ``task -> DMSResult`` (typically from :func:`score_panel`).
        is_amplify: Mark the artifact as the AMPLIFY reference panel.

    Returns:
        A plain ``dict`` ready for :func:`json.dump`.
    """
    assays = sorted(results)
    finite = [
        results[t].spearman
        for t in assays
        if results[t].spearman == results[t].spearman  # not NaN
    ]
    mean_spearman = float(np.mean(finite)) if finite else float("nan")
    per_assay = {
        task: {
            "spearman": results[task].spearman,
            "n": results[task].n,
            "amplify_reference": amplify_reference_for(task),
        }
        for task in assays
    }
    return {
        "model_id": model_id,
        "is_amplify": is_amplify,
        "n_assays": len(assays),
        "mean_spearman": mean_spearman,
        "assays": assays,
        "per_assay": per_assay,
    }


def ckpt_tag(model_id: str) -> str:
    """Derive a filesystem-safe artifact tag from a ckpt dir or model id.

    Args:
        model_id: Checkpoint directory path or ``amplify_120m``.

    Returns:
        A slug usable in ``zeroshot_<tag>.json`` (basename, non-alnum -> ``_``).
    """
    import os
    import re

    base = os.path.basename(os.path.normpath(model_id)) or model_id
    return re.sub(r"[^0-9A-Za-z._-]+", "_", base).strip("_") or "model"


def write_panel_artifact(
    artifact: dict, results_dir: str | None = None
) -> str:
    """Write a panel artifact to ``<results_dir>/zeroshot_<tag>.json``.

    Args:
        artifact: The dict from :func:`build_panel_artifact`.
        results_dir: Output directory (default ``plm/results`` next to the
            package). Created if absent.

    Returns:
        The path written.
    """
    import json
    import os

    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
        )
    os.makedirs(results_dir, exist_ok=True)
    tag = ckpt_tag(str(artifact["model_id"]))
    path = os.path.join(results_dir, f"zeroshot_{tag}.json")
    with open(path, "w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


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
        help=(
            "DMS task. A curated task (gb1 / beta_lactamase), a single ProteinGym "
            "assay (proteingym:<DMS_id>), or 'all' to score every locally cached "
            "assay and write a JSON panel artifact."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Optional cap on the number of variants scored (per assay).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Where to write the panel artifact for --task all (default plm/results).",
    )
    parser.add_argument(
        "--no-proteingym",
        action="store_true",
        help="With --task all, score only the curated gb1/beta_lactamase set.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    # Build the scorer once (shared across the whole panel).
    if args.amplify:
        logger.info("Loading AMPLIFY-120M reference")
        scorer = build_amplify_scorer(args.device)
        model_tag = "amplify_120m"
    else:
        logger.info("Loading Proteva checkpoint from %s", args.ckpt)
        scorer = build_proteva_scorer(args.ckpt, args.device)
        model_tag = args.ckpt

    if args.task == "all":
        tasks = available_tasks(include_proteingym=not args.no_proteingym)
        logger.info("Scoring panel of %d locally cached assays", len(tasks))

        def _progress(task: str, result: DMSResult) -> None:
            print(
                f"{task}\tmodel={model_tag}"
                f"\tspearman={result.spearman:.6f}\tn={result.n}",
                flush=True,
            )

        results = score_panel(
            scorer,
            tasks,
            batch_size=args.batch_size,
            max_variants=args.max_variants,
            on_result=_progress,
        )
        artifact = build_panel_artifact(
            model_tag, results, is_amplify=args.amplify
        )
        path = write_panel_artifact(artifact, results_dir=args.results_dir)
        logger.info(
            "Wrote panel artifact %s (%d assays, mean Spearman %.4f)",
            path,
            artifact["n_assays"],
            artifact["mean_spearman"],
        )
        print(f"PANEL\tmodel={model_tag}\tartifact={path}", flush=True)
        return 0

    logger.info("Loading variants for task %s", args.task)
    variants = load_variants_for_task(args.task, max_variants=args.max_variants)
    n_mut = sum(len(v.mutations) for v in variants)
    logger.info(
        "Loaded %d variants (avg %.2f mutations/variant)",
        len(variants),
        n_mut / max(len(variants), 1),
    )

    result = score_variants(scorer, variants, batch_size=args.batch_size)
    # Single clean stdout line, parseable by callers.
    print(
        f"{args.task}\tmodel={model_tag}\tspearman={result.spearman:.6f}\tn={result.n}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
