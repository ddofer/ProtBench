"""Zero-shot contact prediction by categorical Jacobian.

The unsupervised half of ProtBench's contact prediction, and the model-agnostic
alternative to attention-map contact heads: no MSA, no ``.a3m`` file, no
training, no structure input -- one wild-type sequence in, an L x L coupling
matrix out.

For every position ``i``, substitute each of the 20 canonical amino acids and
record how the model's output distribution shifts at every other position ``j``.
A model that has learned a residue-residue coupling will change its prediction
at ``j`` when ``i`` changes; one that has only learned amino-acid statistics
will not. Double-centring removes the single-site terms, the Frobenius norm
collapses the 20x20 block at each ``(i, j)`` to one number, the self-coupling
diagonal is zeroed, and APC then removes the per-position background. The
diagonal must go first: it is the largest entry by far, and leaving it in makes
APC's background track self-coupling instead of the off-diagonal noise.

This needs an MLM head, which is why it is a separate script rather than a task
in the suite -- ``contact_probe`` is the path that runs against every model.
Both report the same precision-at-L metrics, so the numbers are comparable.

Usage:
    python contact_catjac.py --selfcheck
    python contact_catjac.py -m facebook/esm2_t33_650M_UR50D
    python contact_catjac.py -m facebook/esm2_t33_650M_UR50D --max_proteins 5

References:
    Zhang et al., "Scaling unlocks broader generation and deeper functional
    understanding of proteins" / Ovchinnikov's categorical-Jacobian notebooks.
    Dunn et al. 2008 for APC.
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from contact_metrics import (
    MIN_SEPARATION,
    apc,
    average_contact_metrics,
    contact_metrics,
    contacts_from_tertiary,
    separation_matrix,
)

logger = logging.getLogger(__name__)

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_DATASET = "heya5/protein_contact_map"
DEFAULT_SPLIT = "test"


def _aa_token_ids(tokenizer) -> np.ndarray:
    """Vocabulary ids for the 20 canonical amino acids, in ``CANONICAL_AA`` order."""
    ids = []
    for aa in CANONICAL_AA:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        unk = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or (unk is not None and token_id == unk):
            raise ValueError(
                f"tokenizer has no single-token entry for amino acid {aa!r}; "
                "the categorical Jacobian needs one token per residue"
            )
        ids.append(int(token_id))
    return np.asarray(ids, dtype="int64")


def categorical_jacobian(
    refs,
    tokenizer,
    sequence: str,
    *,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """L x L coupling matrix for one sequence. Costs ``L * 20 + 1`` forwards.

    ``refs`` is a ``wt_test_time_training.MLMHeadRefs``; its ``forward_logits``
    already handles the per-family input quirks (AMPLIFY additive masks, Proteva
    packing, plain HF).
    """
    import torch

    encoded = tokenizer(
        sequence,
        return_tensors="pt",
        return_special_tokens_mask=True,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    special = encoded["special_tokens_mask"][0].numpy().astype(bool)
    residue_positions = np.where(~special)[0]
    length = residue_positions.size
    if length != len(sequence):
        raise ValueError(
            f"tokenizer produced {length} residue tokens for a {len(sequence)}-residue "
            "sequence; the categorical Jacobian assumes one token per residue"
        )

    aa_ids = torch.as_tensor(_aa_token_ids(tokenizer), device=device)
    res_pos = torch.as_tensor(residue_positions, device=device)

    def _logits(ids: "torch.Tensor") -> "torch.Tensor":
        mask = None
        if attention_mask is not None:
            mask = attention_mask.expand(ids.shape[0], -1)
        with torch.inference_mode():
            out = refs.forward_logits(ids, mask)
        # (B, T, V) -> (B, L, 20): residue positions, canonical amino acids only.
        return out.float()[:, res_pos, :][:, :, aa_ids]

    baseline = _logits(input_ids)[0]  # (L, 20)

    # One position's variants is 20 sequences; pack as many whole positions into
    # a forward as batch_size allows so short proteins do not run 20 at a time.
    positions_per_batch = max(batch_size // len(CANONICAL_AA), 1)
    scores = np.zeros((length, length), dtype="float64")

    for start in range(0, length, positions_per_batch):
        chunk = list(range(start, min(start + positions_per_batch, length)))
        variants = input_ids.repeat(len(chunk) * len(CANONICAL_AA), 1)
        for slot, pos in enumerate(chunk):
            row = slot * len(CANONICAL_AA)
            variants[row : row + len(CANONICAL_AA), res_pos[pos]] = aa_ids
        out = _logits(variants).reshape(
            len(chunk), len(CANONICAL_AA), length, len(CANONICAL_AA)
        )
        # jac axes: (position i, mutant aa a, position j, predicted aa b)
        jac = out - baseline
        # Double-centre: strip the single-site terms, leaving the coupling.
        jac = jac - jac.mean(dim=1, keepdim=True)
        jac = jac - jac.mean(dim=3, keepdim=True)
        block = torch.linalg.vector_norm(jac, dim=(1, 3))  # (len(chunk), L)
        scores[chunk] = block.cpu().numpy()

    scores = (scores + scores.T) / 2.0
    # Zero the self-coupling BEFORE APC. Mutating position i dominates the
    # logits at i, so the diagonal is by far the largest entry; leaving it in
    # inflates the row/column sums that APC's rank-1 background is built from,
    # and the background then tracks self-coupling magnitude instead of the
    # off-diagonal noise it is meant to remove. Every off-diagonal score shifts.
    np.fill_diagonal(scores, 0.0)
    scores = apc(scores)
    # Trivially-predictable near-diagonal pairs are never scored, so zero them
    # rather than letting them win top-L slots after the APC shift.
    scores[separation_matrix(length) < MIN_SEPARATION] = 0.0
    return scores


def load_mlm_model(model_name: str, device: Optional[str] = None, bf16: bool = False):
    """Return ``(refs, tokenizer, device)`` for a model with a reachable MLM head.

    Uses the benchmark suite's loader so the custom families (AMPLIFY, ESM++,
    Proteva, ...) get their compatibility patches. That loader prefers
    SentenceTransformer, which discards the MLM head, so fall back to a direct
    ``AutoModelForMaskedLM`` load in that case -- plain ESM2 checkpoints take
    exactly this path.
    """
    import torch
    from transformers import AutoTokenizer

    from model_utils import from_pretrained_with_flash, patch_unknown_residue_tokens
    from protein_benchmark_suite import load_model
    from wt_test_time_training import resolve_mlm_head

    dtype = torch.bfloat16 if bf16 else None
    model_obj, is_sbert, resolved_device = load_model(
        model_name, device=device, torch_dtype=dtype
    )
    if is_sbert:
        import gc

        from transformers import AutoModelForMaskedLM

        logger.info(
            "Loaded as SentenceTransformer (no MLM head); reloading %s as "
            "AutoModelForMaskedLM",
            model_name,
        )
        # Release the SentenceTransformer copy first -- holding both on device
        # doubles VRAM, and this is the path every stock ESM2 checkpoint takes.
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = from_pretrained_with_flash(
            AutoModelForMaskedLM, model_name, **({"dtype": dtype} if dtype else {})
        )
        model.to(resolved_device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        patch_unknown_residue_tokens(tokenizer)
    else:
        tokenizer, model = model_obj
        model.eval()

    # Raises with an actionable message when no MLM/decoder head is reachable.
    refs = resolve_mlm_head(model, tokenizer)
    return refs, tokenizer, resolved_device


def score_dataset(
    refs,
    tokenizer,
    records: Sequence[Dict],
    *,
    device: str,
    batch_size: int,
    max_len: int,
) -> Dict[str, float]:
    """Score every record short enough to run, averaging metrics per protein."""
    per_protein: List[Dict[str, float]] = []
    skipped: List[str] = []
    for record in records:
        sequence = str(record["seq"])
        if len(sequence) > max_len:
            skipped.append(f"{record.get('name', '?')}({len(sequence)})")
            continue
        scores = categorical_jacobian(
            refs, tokenizer, sequence, device=device, batch_size=batch_size
        )
        contacts, valid_pair = contacts_from_tertiary(
            np.asarray(record["tertiary"], dtype="float64"),
            record.get("valid_mask"),
        )
        per_protein.append(contact_metrics(scores, contacts, valid_pair))
        logger.info(
            "  %s (L=%d): P@L/5_long=%.4f",
            record.get("name", "?"),
            len(sequence),
            per_protein[-1]["P@L/5_long"],
        )

    if skipped:
        # Never let a coverage cut pass silently -- a mean over the short
        # proteins only is not the number the header claims it is.
        logger.warning(
            "SKIPPED %d/%d proteins longer than --max_len %d: %s",
            len(skipped),
            len(records),
            max_len,
            ", ".join(skipped),
        )
    if not per_protein:
        raise RuntimeError(
            f"no protein was short enough to score at --max_len {max_len}"
        )
    metrics = average_contact_metrics(per_protein)
    metrics["Proteins_Scored"] = float(len(per_protein))
    return metrics


def _selfcheck() -> None:
    """Runnable check: python contact_catjac.py --selfcheck

    Runs the real Jacobian against a tiny stub model whose logits depend on a
    planted pair of positions, so a correct implementation must recover exactly
    that pair. A stub keeps the check offline and instant; the shape invariants
    (symmetry, zeroed band) are asserted on the same output.
    """
    import torch

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    class _StubTokenizer:
        """One token per residue, ids 0..19, plus a CLS/SEP pair."""

        vocab = {aa: i for i, aa in enumerate(CANONICAL_AA)}
        unk_token_id = None

        def convert_tokens_to_ids(self, token):
            return self.vocab[token]

        def __call__(self, seq, **kwargs):
            ids = [20] + [self.vocab[c] for c in seq] + [21]
            special = [1] + [0] * len(seq) + [1]
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "special_tokens_mask": torch.tensor([special]),
            }

    length = 24
    coupled = (3, 19)  # separation 16, comfortably past MIN_SEPARATION

    class _StubRefs:
        """Logits at `coupled[1]` depend on the residue at `coupled[0]`, and
        every other position depends only on itself."""

        def forward_logits(self, input_ids, attention_mask=None):
            batch, tokens = input_ids.shape
            out = torch.zeros(batch, tokens, 22)
            residues = input_ids[:, 1:-1]
            for pos in range(length):
                out[:, pos + 1, :20] = torch.nn.functional.one_hot(
                    residues[:, pos], 22
                )[:, :20].float()
            src = residues[:, coupled[0]]
            out[:, coupled[1] + 1, :20] += (
                torch.nn.functional.one_hot(src, 22)[:, :20].float() * 5.0
            )
            return out

    rng = np.random.RandomState(0)
    seq = "".join(CANONICAL_AA[i] for i in rng.randint(0, 20, size=length))
    scores = categorical_jacobian(_StubRefs(), _StubTokenizer(), seq, batch_size=40)

    assert scores.shape == (length, length)
    assert np.allclose(scores, scores.T), "coupling matrix must be symmetric"
    band = separation_matrix(length) < MIN_SEPARATION
    assert not scores[band].any(), "near-diagonal band must be zeroed"

    # The planted pair must be the single highest-scoring pair.
    upper = np.triu(np.ones((length, length), dtype=bool), k=MIN_SEPARATION)
    best = np.unravel_index(np.argmax(np.where(upper, scores, -np.inf)), scores.shape)
    assert tuple(sorted(best)) == coupled, f"recovered {best}, planted {coupled}"

    print("contact_catjac selfcheck OK")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-m", "--model_name", help="HF id or local checkpoint path")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--device", default=None, help="auto/cuda/cpu")
    parser.add_argument("--bf16", action="store_true", help="load weights in bfloat16")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="sequences per forward; rounded down to whole positions (20 each)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="skip proteins longer than this; skips are logged, never silent",
    )
    parser.add_argument(
        "--max_proteins", type=int, default=0, help="0 = the whole split"
    )
    parser.add_argument("--output_dir", "-o", default="results/benchmarks")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    if args.selfcheck:
        _selfcheck()
        return 0
    if not args.model_name:
        parser.error("-m/--model_name is required (or pass --selfcheck)")

    from datasets import load_dataset

    from protein_benchmark_suite import ResultTracker

    split = load_dataset(args.dataset, split=args.split)
    # Truncate BEFORE materialising: the train split is 25,299 proteins whose
    # coordinate arrays cost far more as Python objects than as Arrow.
    if args.max_proteins:
        split = split.select(range(min(args.max_proteins, len(split))))
    records = list(split)
    logger.info("Scoring %d proteins from %s[%s]", len(records), args.dataset, args.split)

    refs, tokenizer, device = load_mlm_model(args.model_name, args.device, args.bf16)
    metrics = score_dataset(
        refs,
        tokenizer,
        records,
        device=device,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )

    tracker = ResultTracker(args.model_name)
    tracker.add(
        "Contact Prediction (Categorical Jacobian)",
        metrics,
        samples=len(records),
        probe="catjac",
        eval_mode="zeroshot",
        eval_split=args.split,
        eval_strategy="test_split",
    )
    tracker.display()
    tracker.save(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
