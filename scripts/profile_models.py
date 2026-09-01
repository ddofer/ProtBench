#!/usr/bin/env python3
"""Profile frozen encoder load, parameter count, memory, and inference cost."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parameter_counts(model: Any) -> dict[str, int]:
    counts = {"total_parameters": 0, "encoder_parameters": 0, "aux_parameters": 0}
    aux_markers = (
        "aux",
        "pssm_head",
        "cons_head",
        "di3_",
        "tax_",
        "plddt_head",
        "pfam_",
        "ptm_any_head",
        "ss3_head",
    )
    for name, parameter in model.named_parameters():
        size = int(parameter.numel())
        counts["total_parameters"] += size
        if name.startswith("encoder."):
            counts["encoder_parameters"] += size
        if any(marker in name for marker in aux_markers):
            counts["aux_parameters"] += size
    return counts


def reset_peak_memory(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    files = (
        [file for file in path.rglob("*") if file.is_file()]
        if path.is_dir()
        else [path]
    )
    config = path / "config.json" if path.is_dir() else None
    weights = [file for file in files if file.name.endswith((".safetensors", ".bin"))]
    return {
        "checkpoint": str(path.resolve()),
        "file_size_bytes": sum(file.stat().st_size for file in files),
        "config_sha256": _sha256(config) if config and config.is_file() else None,
        "weight_sha256": {
            str(file.relative_to(path)) if path.is_dir() else file.name: _sha256(file)
            for file in weights
        },
    }


def _forward(
    encoder,
    tokenizer,
    sequences: list[str],
    *,
    device: str,
    batch_size: int,
    max_length: int,
) -> None:
    from token_classification_probe import iter_residue_embeddings

    list(
        iter_residue_embeddings(
            encoder=encoder,
            tokenizer=tokenizer,
            sequences=sequences,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    )


def profile_model(
    name: str,
    checkpoint: Path,
    *,
    device: str,
    batch_size: int,
    max_length: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    import torch

    from protein_benchmark_suite import _unwrap_encoder_tokenizer, load_model

    load_started = time.perf_counter()
    dtype = (
        torch.bfloat16
        if device.startswith("cuda") and torch.cuda.is_bf16_supported()
        else None
    )
    model_obj, is_sbert, resolved_device = load_model(
        str(checkpoint), device=device, torch_dtype=dtype
    )
    load_seconds = time.perf_counter() - load_started
    tokenizer, encoder = _unwrap_encoder_tokenizer(model_obj, is_sbert)
    model = model_obj[1] if isinstance(model_obj, tuple) else encoder
    sequences = [
        "M" + "ACDEFGHIKLMNPQRSTVWY" * ((max_length - 2) // 20)
        for _ in range(batch_size)
    ]
    for _ in range(warmup_iterations):
        _forward(
            encoder,
            tokenizer,
            sequences,
            device=resolved_device,
            batch_size=batch_size,
            max_length=max_length,
        )
    if resolved_device.startswith("cuda"):
        torch.cuda.synchronize(resolved_device)
    reset_peak_memory(torch, resolved_device)
    seconds: list[float] = []
    for _ in range(measured_iterations):
        started = time.perf_counter()
        _forward(
            encoder,
            tokenizer,
            sequences,
            device=resolved_device,
            batch_size=batch_size,
            max_length=max_length,
        )
        if resolved_device.startswith("cuda"):
            torch.cuda.synchronize(resolved_device)
        seconds.append(time.perf_counter() - started)
    p95_index = max(0, round(0.95 * (len(seconds) - 1)))
    tokens = sum(min(len(sequence), max_length - 2) for sequence in sequences)
    peak = (
        torch.cuda.max_memory_allocated(resolved_device)
        if resolved_device.startswith("cuda")
        else 0
    )
    return {
        "name": name,
        **checkpoint_fingerprint(checkpoint),
        **parameter_counts(model),
        "load_seconds": load_seconds,
        "median_latency_seconds": statistics.median(seconds),
        "p95_latency_seconds": sorted(seconds)[p95_index],
        "tokens_per_second": tokens / statistics.median(seconds),
        "peak_vram_bytes": int(peak),
        "precision": "bf16" if dtype is not None else "fp32",
        "device": resolved_device,
        "attention_backend": "suite-default",
        "batch_size": batch_size,
        "max_length": max_length,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }


def write_reports(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "profile.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    fields = sorted({field for row in rows for field in row})
    with (output / "profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    columns = [
        "name",
        "total_parameters",
        "encoder_parameters",
        "aux_parameters",
        "median_latency_seconds",
        "p95_latency_seconds",
        "tokens_per_second",
        "peak_vram_bytes",
    ]
    lines = [
        "# Frozen-model runtime profile",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    )
    (output / "profile.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measured-iterations", type=int, default=10)
    args = parser.parse_args(argv)
    rows = []
    for spec in args.model:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name or not Path(raw_path).exists():
            raise ValueError(f"--model needs an existing NAME=PATH: {spec}")
        rows.append(
            profile_model(
                name,
                Path(raw_path),
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                warmup_iterations=args.warmup_iterations,
                measured_iterations=args.measured_iterations,
            )
        )
    write_reports(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
