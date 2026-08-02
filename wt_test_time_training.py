"""Wild-type test-time training (TTT) for zero-shot variant-effect prediction.

Implements the "ProteinTTT" trick (arXiv 2411.02109, ICLR 2026): before scoring
an assay's variants, run a few extra rounds of masked-language-model (MLM)
fine-tuning on that assay's *single wild-type (WT) sequence*, then read out from
the lightly-adapted model. Here the readout is the benchmark suite's existing
zero-shot embedding-cosine score, so the adaptation must move the *encoder*
representations (last-N transformer blocks), not just the MLM head.

Design + evidence: docs/superpowers/specs/2026-06-05-wt-test-time-training-design.md

Scope: AMPLIFY and Proteva HF models loaded by ``protein_benchmark_suite.load_model``
(``model_obj == (tokenizer, model)``). The engine itself is architecture-agnostic
— it operates only on an :class:`MLMHeadRefs` bundle produced by
:func:`resolve_mlm_head`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TTAConfig:
    """Hyperparameters for one WT adaptation. Defaults follow the ProteinTTT recipe.

    SGD with momentum=0 / weight_decay=0 is the published catastrophic-forgetting
    guard (not Adam). ``train_head`` defaults True per the user's choice; set
    False to match ProteinTTT exactly (adapt the backbone ``f``, not the head ``h``).
    """

    iters: int = 20
    lr: float = 4e-4
    n_layers: int = 2
    mask_rate: float = 0.15
    train_head: bool = True
    seed: int = 0


@dataclass
class MLMHeadRefs:
    """References into a model needed to run MLM TTT + the embedding readout.

    ``forward_logits(input_ids, attention_mask) -> logits [B, L, V]`` runs a
    grad-enabled forward producing vocabulary logits sliced back to the input
    length. ``encoder_blocks`` is the transformer-block ``ModuleList`` whose top
    ``n_layers`` we unfreeze; ``final_norm`` (may be None) and ``head_module`` are
    unfrozen too only when relevant.
    """

    forward_logits: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    encoder_blocks: nn.ModuleList
    final_norm: Optional[nn.Module]
    head_module: nn.Module
    mask_token_id: int
    special_ids: List[int]


def _model_type(model) -> str:
    return str(getattr(getattr(model, "config", None), "model_type", "") or "")


def resolve_mlm_head(model, tokenizer) -> MLMHeadRefs:
    """Locate the MLM head + encoder blocks for a benchmark-loaded model.

    Raises a clear ``ValueError`` if no MLM/decoder head is reachable (e.g. an
    encoder-only ``AutoModel`` export), so the caller can tell the user to load
    the ``*ForMaskedLM`` / pretraining checkpoint instead.
    """
    mtype = _model_type(model)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError(
            "Tokenizer has no mask_token_id; cannot run MLM test-time training."
        )
    special_ids = list(getattr(tokenizer, "all_special_ids", []) or [])

    if mtype == "AMPLIFY":
        if not (hasattr(model, "transformer_encoder") and hasattr(model, "decoder")):
            raise ValueError(
                "AMPLIFY model is missing transformer_encoder/decoder; cannot run TTT."
            )
        from model_utils import _prepare_amplify_inputs

        def forward_logits(input_ids, attention_mask):
            param = next(model.parameters(), None)
            ids, additive_mask, orig_len, _ = _prepare_amplify_inputs(
                input_ids,
                attention_mask,
                dtype=param.dtype if param is not None else None,
            )
            out = model(
                input_ids=ids,
                attention_mask=additive_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            return out.logits[:, :orig_len, :]

        return MLMHeadRefs(
            forward_logits=forward_logits,
            encoder_blocks=model.transformer_encoder,
            final_norm=getattr(model, "layer_norm_2", None),
            head_module=model.decoder,
            mask_token_id=int(mask_id),
            special_ids=special_ids,
        )

    if mtype == "proteva":
        # ProtevaForPretraining.forward(input_ids, attention_mask) returns
        # ProtevaOutput.logits = full-length [B, T, vocab] MLM logits via the
        # encoder's own decoder head. In fp32 (flash_attn_mode="off", SDPA dense)
        # the standard 1/0 attention_mask is consumed directly — no additive-mask
        # or varlen prep needed (unlike AMPLIFY).
        encoder = getattr(model, "encoder", None)
        decoder = getattr(encoder, "decoder", None)
        if encoder is None or decoder is None:
            raise ValueError(
                "Proteva model is missing encoder.decoder; cannot run MLM zero-shot."
            )

        def forward_logits(input_ids, attention_mask):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits

        return MLMHeadRefs(
            forward_logits=forward_logits,
            encoder_blocks=encoder.blocks,
            final_norm=getattr(encoder, "final_norm", None),
            head_module=decoder,
            mask_token_id=int(mask_id),
            special_ids=special_ids,
        )

    if mtype in ("ESMplusplus", "esmplusplus", "esm_plusplus"):
        # Vanilla ESM-C (Synthyra ESMplusplusForMaskedLM): a standard HF MaskedLM
        # whose forward returns .logits [B, T, vocab]. Plain MLM zero-shot needs
        # only forward_logits + mask_token_id + special_ids; the encoder_blocks /
        # head_module handles below are best-effort so optional TTT does not KeyError
        # — but TTT is NOT validated for this arch, so run vanilla ESM-C zero-shot
        # only (no --ttt). Without this branch every ProteinGym MLM zero-shot shard
        # for vanilla ESM-C died with the ValueError below.
        def forward_logits(input_ids, attention_mask):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits

        inner = getattr(model, "transformer", None) or getattr(model, "esm", None) or model
        blocks = (
            getattr(getattr(inner, "transformer", inner), "blocks", None)
            or getattr(inner, "blocks", None)
            or getattr(inner, "layers", None)
            or nn.ModuleList()
        )
        head = (
            getattr(model, "sequence_head", None)
            or getattr(model, "lm_head", None)
            or getattr(model, "decoder", None)
            or model
        )
        return MLMHeadRefs(
            forward_logits=forward_logits,
            encoder_blocks=blocks,
            final_norm=None,
            head_module=head,
            mask_token_id=int(mask_id),
            special_ids=special_ids,
        )

    raise ValueError(
        f"WT test-time training is not supported for model_type={mtype!r}. "
        "Supported: AMPLIFY, proteva, ESMplusplus. Load a model exposing an MLM/decoder head."
    )


def _trainable_params(refs: MLMHeadRefs, cfg: TTAConfig) -> List[nn.Parameter]:
    """Parameters TTT updates: top ``n_layers`` blocks (+ final norm, + head).

    De-duplicated by tensor identity so a head that is weight-tied to the
    embedding does not appear twice.
    """
    blocks = list(refs.encoder_blocks)
    n = max(1, min(cfg.n_layers, len(blocks)))
    chosen: List[nn.Parameter] = []
    for blk in blocks[-n:]:
        chosen.extend(blk.parameters())
    if refs.final_norm is not None:
        chosen.extend(refs.final_norm.parameters())
    if cfg.train_head:
        chosen.extend(refs.head_module.parameters())
    seen, unique = set(), []
    for p in chosen:
        if id(p) not in seen:
            seen.add(id(p))
            unique.append(p)
    return unique


def snapshot_trainable(refs: MLMHeadRefs, cfg: TTAConfig):
    """Snapshot the pristine values of the to-be-trained params, taken ONCE.

    Returns a list of ``(param, cloned_value)``; pass it to every
    :func:`adapt_to_wt` call so each assay restores to the same HEAD weights.
    """
    return [(p, p.detach().clone()) for p in _trainable_params(refs, cfg)]


class WTAdaptation:
    """Handle for one WT adaptation. ``restore()`` reverts to the pristine snapshot."""

    def __init__(self, pristine_snapshot, losses: List[float]):
        self._snapshot = pristine_snapshot
        self.losses = losses

    def restore(self) -> None:
        with torch.no_grad():
            for p, saved in self._snapshot:
                p.data.copy_(saved)


def adapt_to_wt(
    model,
    refs: MLMHeadRefs,
    wt_seq: str,
    tokenizer,
    cfg: TTAConfig,
    device: str = "cpu",
    pristine_snapshot=None,
    max_length: int = 1024,
) -> WTAdaptation:
    """Run ``cfg.iters`` SGD-MLM steps on the single WT sequence, in place.

    Freezes the whole model, unfreezes the top ``n_layers`` blocks (+ final norm,
    + head iff ``cfg.train_head``), and minimises masked-token cross-entropy with
    SGD(momentum=0, weight_decay=0). The model is left in ``eval()`` mode with the
    adapted weights live; call ``WTAdaptation.restore()`` to revert.
    """
    if pristine_snapshot is None:
        pristine_snapshot = snapshot_trainable(refs, cfg)

    params = _trainable_params(refs, cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)

    enc = tokenizer(
        wt_seq, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    special = torch.tensor(refs.special_ids, device=device)
    maskable = (attention_mask == 1) & (~torch.isin(input_ids, special))
    if not bool(maskable.any()):
        raise ValueError(
            "WT sequence has no maskable (non-special) tokens; cannot run TTT."
        )

    gen = torch.Generator(device=device).manual_seed(int(cfg.seed))
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.0, weight_decay=0.0)

    model.train()
    losses: List[float] = []
    for _ in range(cfg.iters):
        probs = torch.full(input_ids.shape, cfg.mask_rate, device=device)
        draw = (torch.bernoulli(probs, generator=gen).bool()) & maskable
        if not bool(draw.any()):
            # Guarantee at least one masked token for a usable gradient.
            first = maskable.nonzero(as_tuple=False)[0]
            draw[tuple(first.tolist())] = True
        masked = input_ids.clone()
        masked[draw] = refs.mask_token_id

        logits = refs.forward_logits(masked, attention_mask)
        loss = F.cross_entropy(logits[draw], input_ids[draw])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    return WTAdaptation(pristine_snapshot, losses)


def run_tta_zeroshot(
    model,
    refs: MLMHeadRefs,
    tokenizer,
    mutants: List[str],
    wts: List[str],
    groups,
    labels,
    problem_type: str,
    embed_fn: Callable[[List[str]], "object"],
    tta_cfg: TTAConfig,
    device: str = "cpu",
    max_length: int = 1024,
    min_group: int = 2,
) -> List[float]:
    """Per-assay WT-TTT zero-shot scoring with the embedding-cosine readout.

    For each assay group: adapt the model to that group's WT, embed the group's
    mutants + the WT with the adapted weights (via ``embed_fn``), score
    ``cosine(mutant, WT)`` against the labels (Spearman for regression, AUC for
    binary), then restore the model to its pristine weights before the next group.

    ``embed_fn(seqs) -> array[n, H]`` is injected (the suite passes a closure over
    ``embed_sequences``) so the orchestration is testable without the full suite.
    Returns the list of finite per-group metric values.
    """
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    groups = np.asarray(groups)
    labels = np.asarray(labels)
    pristine = snapshot_trainable(refs, tta_cfg)

    metrics: List[float] = []
    for g in np.unique(groups):
        mask = groups == g
        idx = np.nonzero(mask)[0]
        if idx.size < min_group:
            continue
        wt = wts[idx[0]]
        if not wt:
            continue
        handle = adapt_to_wt(
            model, refs, wt, tokenizer, tta_cfg, device=device,
            pristine_snapshot=pristine, max_length=max_length,
        )
        try:
            mut_embs = np.asarray(embed_fn([mutants[i] for i in idx]), dtype=np.float64)
            wt_emb = np.asarray(embed_fn([wt]), dtype=np.float64)
            sims = F.cosine_similarity(
                torch.as_tensor(mut_embs), torch.as_tensor(wt_emb)
            ).numpy()
        finally:
            handle.restore()

        y = labels[idx].astype(float)
        if problem_type == "regression":
            corr, _ = spearmanr(y, sims)
            if np.isfinite(corr):
                metrics.append(float(corr))
        else:
            # Pathogenic = lower cosine to WT → negate (same sign convention as
            # the cosine + MLM zero-shot paths).
            try:
                metrics.append(float(roc_auc_score(y, -sims)))
            except ValueError:
                pass
    return metrics
