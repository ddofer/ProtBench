"""Measure dynconv-qkv tok/s vs a no-conv baseline.

CUDA records real overhead for the 0.90 vendored throughput gate; CPU is a
plumbing/correctness check only (run CPU with CUDA_VISIBLE_DEVICES='' per the
smoke-isolation rule — never co-tenant the manifest GPU). Prints the tok/s
ratio so the gate can be checked before queueing the Stage-2 slate.
"""
import argparse
import time

import torch

from plm.model import EncoderConfig, ProteinEncoder


def _mk(layers, mode):
    return EncoderConfig(num_layers=24, num_heads=10, hidden_size=640,
                         ffn_size=2560, vocab_size=33, max_position=2048,
                         dynconv_qkv_layers=layers, dynconv_qkv_mode=mode)


def _tps(cfg, device, steps=20, B=2, L=512):
    m = ProteinEncoder(cfg).to(device)
    ids = torch.randint(0, 33, (B, L), device=device)
    for _ in range(3):
        m(ids)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        m(ids)
    if device == "cuda":
        torch.cuda.synchronize()
    return steps * B * L / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="headwise", choices=["headwise", "lowrank"])
    ap.add_argument("--layers", default="0,1,2,3,4,5")
    a = ap.parse_args()
    layers = tuple(int(x) for x in a.layers.split(",") if x.strip())
    base = _tps(_mk((), a.mode), a.device)
    conv = _tps(_mk(layers, a.mode), a.device)
    ratio = conv / base
    print(f"baseline tok/s={base:,.0f}  dynconv tok/s={conv:,.0f}  ratio={ratio:.3f}")
    print("PASS gate" if ratio >= 0.90 else "BELOW 0.90 gate — flag for parity/cross-bucket")


if __name__ == "__main__":
    main()
