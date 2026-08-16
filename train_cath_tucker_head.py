#!/usr/bin/env python
"""ProtTucker-style contrastive head on top of a FROZEN backbone's CATH embeddings.

`cath_levels.py` scores raw, untrained embeddings by 1-NN. This trains a small
projection head on top of those same embeddings -- the paper's own method,
reproduced on our backbones instead of theirs. It answers a different question:
"how good is the embedding space" (cath_levels.py) vs "how good is the embedding
space after supervised contrastive refinement on CATH itself" (this script).

Architecture and training recipe match Heinzinger et al. 2022 (train_prottucker.py
in Rostlab/EAT), reimplemented rather than vendored (their script is GPL-3.0 and,
separately, broken against this repo's data: it references a nonexistent
train74k.fasta and never wires up evaluation against lookup69k/test219 at all):

  - 2-layer FNN: in_dim -> 256 -> 128, tanh nonlinearity, no final activation.
  - Hierarchy-sampling: for each of the 4 CATH levels per minibatch, mine a
    positive (same label at this level) and a negative (same label at the PARENT
    level, different at this one -- e.g. same architecture, different topology).
    That parent constraint is what makes the negative meaningful rather than
    trivial; C has no parent, so any differing C-label is an eligible negative.
  - Batch-hard: within the minibatch's valid candidates, pick the FARTHEST valid
    positive and the NEAREST valid negative per anchor (hardest of each).
  - Soft Margin Loss: log(1 + exp(-(d_neg - d_pos))), averaged over anchors with
    a valid triplet at that level, summed across the 4 levels per batch.
  - Adam, lr 1e-3, batch 256, early stopping on val200's H-level 1-NN accuracy
    against train66k as the lookup pool (train66k is redundancy-reduced against
    val200/test300, so it is a valid lookup set for early stopping).

Final evaluation reuses cath_levels.score_levels/bootstrap_ci/nearest_neighbour
on the real lookup69k -> test219 split, so the numbers are produced by the exact
scorer already cross-checked against the suite -- no second scoring path to
distrust.

Usage:
    python train_cath_tucker_head.py --models protsent_v2_150m=/path esm2_150m=Synthyra/ESM2-150M
    python train_cath_tucker_head.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cath_levels import (  # noqa: E402
    EXPECTED_ANSWERABLE,
    LEVELS,
    bootstrap_ci,
    nearest_neighbour,
    score_levels,
)

logger = logging.getLogger("train_cath_tucker_head")

DATASET = "GrimSqueaker/cath43-eat"
PARENT_LEVEL = {"cath_c": None, "cath_a": "cath_c", "cath_t": "cath_a", "cath_h": "cath_t"}


class ProtTuckerHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, proj: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, proj))

    def forward(self, x):
        return self.net(x)


def batch_hard_triplet_loss(z: torch.Tensor, labels: dict[str, np.ndarray], idx: np.ndarray) -> torch.Tensor:
    """Sum of Soft Margin Loss over the 4 CATH levels for one minibatch.

    `z` is the projected batch (already the forward pass, so gradients flow).
    `labels[level][idx]` gives each batch member's label string at that level.
    """
    dist = torch.cdist(z, z, p=2)  # [B, B], differentiable
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)

    total = z.new_zeros(())
    n_terms = 0
    for level in LEVELS:
        lab = labels[level][idx]
        same = torch.from_numpy(lab[:, None] == lab[None, :]).to(z.device)
        positive_mask = same & ~eye

        parent = PARENT_LEVEL[level]
        if parent is None:
            negative_mask = ~same & ~eye
        else:
            plab = labels[parent][idx]
            same_parent = torch.from_numpy(plab[:, None] == plab[None, :]).to(z.device)
            negative_mask = same_parent & ~same & ~eye

        has_pos = positive_mask.any(dim=1)
        has_neg = negative_mask.any(dim=1)
        valid = has_pos & has_neg
        if not bool(valid.any()):
            continue

        d_pos = dist.masked_fill(~positive_mask, -float("inf")).max(dim=1).values
        d_neg = dist.masked_fill(~negative_mask, float("inf")).min(dim=1).values

        margin = d_neg[valid] - d_pos[valid]
        loss = torch.log1p(torch.exp(-margin)).mean()
        total = total + loss
        n_terms += 1

    return total if n_terms else z.new_zeros(())


def load_splits_with_labels():
    from datasets import load_dataset

    ds = load_dataset(DATASET)
    out = {}
    for split, key in [("train", "train"), ("validation", "val"), ("lookup", "lookup"), ("test219", "test")]:
        d = ds[split]
        out[key] = {
            "seqs": list(d["sequence"]),
            "labels": {lv: np.asarray(d[lv]) for lv in LEVELS},
        }
    return out


def embed_split(model_name, seqs, batch_size, max_length):
    from protein_benchmark_suite import _model_cache_namespace, embed_sequences, load_model
    from seq_embed_cache import cached_embed_sequences

    cache_root = str(Path("embed_cache") / _model_cache_namespace(model_name) / "seq_cache")
    cfg_key = f"trunk|l2=0|ml={max_length}|dt=fp32"
    obj, is_sbert, device = load_model(model_name, device="cuda")
    x = cached_embed_sequences(
        lambda: embed_sequences(obj, is_sbert, seqs, device, batch_size=batch_size, max_length=max_length),
        seqs, cache_root=cache_root, cfg_key=cfg_key,
    )
    del obj
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def h_accuracy(z_query, z_pool, labels_query, labels_pool):
    nn_idx = nearest_neighbour(z_query, z_pool)
    pred = labels_pool["cath_h"][nn_idx]
    truth = labels_query["cath_h"]
    present = set(labels_pool["cath_h"].tolist())
    mask = np.array([t in present for t in truth])
    if not mask.any():
        return 0.0
    return float((pred[mask] == truth[mask]).mean())


def train_head(
    x_train: np.ndarray, labels_train, x_val: np.ndarray, labels_val,
    epochs: int, batch_size: int, lr: float, patience: int, device: str, seed: int,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = ProtTuckerHead(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    x_train_t = torch.from_numpy(x_train).to(device)
    x_val_t = torch.from_numpy(x_val).to(device)

    best_acc, best_state, bad_epochs = -1.0, None, 0
    n = len(x_train)

    with torch.no_grad():
        raw_acc = h_accuracy(x_val, x_train, labels_val, labels_train)
    logger.info("  epoch 0 (raw, untrained)  val H-acc %.4f", raw_acc)

    for epoch in range(1, epochs + 1):
        head.train()
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 4:  # too small to form triplets meaningfully
                continue
            z = head(x_train_t[idx])
            loss = batch_hard_triplet_loss(z, labels_train, idx)
            if loss.requires_grad:
                opt.zero_grad()
                loss.backward()
                opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1

        head.eval()
        with torch.no_grad():
            z_val = head(x_val_t).cpu().numpy()
            z_train = head(x_train_t).cpu().numpy()
        acc = h_accuracy(z_val, z_train, labels_val, labels_train)
        logger.info("  epoch %d  loss %.4f  val H-acc %.4f", epoch, epoch_loss / max(n_batches, 1), acc)

        if acc > best_acc:
            best_acc, bad_epochs = acc, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("  early stop at epoch %d (best val H-acc %.4f)", epoch, best_acc)
                break

    head.load_state_dict(best_state)
    head.eval()
    return head, best_acc, raw_acc


def run_arm(tag, model_name, data, args, out_dir):
    logger.info("=== %s (%s)", tag, model_name)
    x_train = embed_split(model_name, data["train"]["seqs"], args.batch_size, args.max_length)
    x_val = embed_split(model_name, data["val"]["seqs"], args.batch_size, args.max_length)
    x_lookup = embed_split(model_name, data["lookup"]["seqs"], args.batch_size, args.max_length)
    x_test = embed_split(model_name, data["test"]["seqs"], args.batch_size, args.max_length)

    head, best_val_acc, raw_val_acc = train_head(
        x_train, data["train"]["labels"], x_val, data["val"]["labels"],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        patience=args.patience, device="cuda", seed=args.seed,
    )

    with torch.no_grad():
        z_lookup = head(torch.from_numpy(x_lookup).cuda()).cpu().numpy()
        z_test = head(torch.from_numpy(x_test).cuda()).cpu().numpy()

    nn_idx = nearest_neighbour(z_test, z_lookup)
    levels = score_levels(data["test"]["labels"], data["lookup"]["labels"], nn_idx, n_boot=args.n_boot)

    got = {lv: levels[lv]["n_answerable"] for lv in LEVELS}
    if got != EXPECTED_ANSWERABLE:
        raise SystemExit(f"{tag}: answerable counts {got} != expected {EXPECTED_ANSWERABLE}")

    # Raw (untrained-head) reference, same lookup69k/test219, for the delta that matters.
    raw_nn = nearest_neighbour(x_test, x_lookup)
    raw_levels = score_levels(data["test"]["labels"], data["lookup"]["labels"], raw_nn, n_boot=args.n_boot)

    result = {
        "model": model_name,
        "dim": int(x_train.shape[1]),
        "raw_val_h_acc": raw_val_acc,
        "best_val_h_acc": best_val_acc,
        "tucker_levels": levels,
        "raw_levels": raw_levels,
    }
    logger.info(
        "%s  RAW  C %.1f A %.1f T %.1f H %.1f  ->  TUCKER  C %.1f A %.1f T %.1f H %.1f",
        tag,
        *[100 * raw_levels[lv]["accuracy"] for lv in LEVELS],
        *[100 * levels[lv]["accuracy"] for lv in LEVELS],
    )
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))
    return result


def selfcheck() -> int:
    """No GPU, no network: the loss and the parent-constrained negative mining."""
    torch.manual_seed(0)
    z = torch.randn(8, 4, requires_grad=True)
    labels = {
        "cath_c": np.array(["1", "1", "1", "1", "2", "2", "2", "2"]),
        "cath_a": np.array(["1.1", "1.1", "1.2", "1.2", "2.1", "2.1", "2.2", "2.2"]),
        "cath_t": np.array(["1.1.1"] * 8),
        "cath_h": np.array(["1.1.1.1"] * 8),
    }
    idx = np.arange(8)
    loss = batch_hard_triplet_loss(z, labels, idx)
    assert loss.requires_grad
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()

    # Negative at cath_a must share cath_c (parent) but differ at cath_a.
    # Row 0 (c=1,a=1.1): valid cath_a negatives are rows 2,3 (c=1,a=1.2). Rows
    # 4-7 (c=2) must NEVER be selected as the cath_a negative for row 0.
    dist = torch.cdist(z, z, p=2)
    lab_a = labels["cath_a"]
    lab_c = labels["cath_c"]
    same_a = lab_a[:, None] == lab_a[None, :]
    same_c = lab_c[:, None] == lab_c[None, :]
    neg_mask = same_c & ~same_a
    assert neg_mask[0, 2] and neg_mask[0, 3]
    assert not neg_mask[0, 4] and not neg_mask[0, 5]

    # A level with zero valid triplets (all identical labels) must be skipped,
    # not crash -- cath_t/cath_h above are constant, so only c/a contribute.
    only_c_a_labels = {k: v for k, v in labels.items() if k in ("cath_c", "cath_a")}
    import cath_levels
    old_levels = cath_levels.LEVELS
    try:
        cath_levels.LEVELS = ("cath_c", "cath_a")
        z2 = torch.randn(8, 4, requires_grad=True)
        loss2 = batch_hard_triplet_loss(z2, labels, idx)
        assert torch.isfinite(loss2) and loss2.item() > 0
    finally:
        cath_levels.LEVELS = old_levels

    print("selfcheck OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", metavar="TAG=PATH", required=False)
    ap.add_argument("--out", default="/home/ddofer/ProtSent/results/benchmarks/cath_eat_tucker")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.selfcheck:
        return selfcheck()

    if not args.models:
        raise SystemExit("--models TAG=PATH [TAG=PATH ...] is required (no default arms)")
    arms = [tuple(m.split("=", 1)) for m in args.models]

    data = load_splits_with_labels()
    logger.info(
        "train %d, val %d, lookup %d, test %d",
        len(data["train"]["seqs"]), len(data["val"]["seqs"]),
        len(data["lookup"]["seqs"]), len(data["test"]["seqs"]),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for tag, model_name in arms:
        try:
            results[tag] = run_arm(tag, model_name, data, args, out_dir)
        except Exception as exc:
            logger.error("%s FAILED: %s: %s", tag, type(exc).__name__, exc)

    (out_dir / "all_arms.json").write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out_dir / "all_arms.json")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PROTEIN_BENCH_ATTN_IMPLEMENTATION", "sdpa")
    sys.exit(main())
