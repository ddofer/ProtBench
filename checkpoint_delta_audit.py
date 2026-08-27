"""Audit post-hoc tensor interventions between local model checkpoints.

This is designed for inference-time component sensitivity checks: it proves
which tensors changed relative to a trained checkpoint and whether the changed
tensors were zeroed. It does not turn such interventions into independently
trained or causal architecture ablations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def audit_variant(baseline: Path, variant: Path) -> dict[str, Any]:
    from safetensors import safe_open

    changed = []
    with safe_open(baseline, framework="np") as baseline_file, safe_open(
        variant, framework="np"
    ) as variant_file:
        baseline_keys = list(baseline_file.keys())
        variant_keys = list(variant_file.keys())
        if baseline_keys != variant_keys:
            missing = sorted(set(baseline_keys) - set(variant_keys))
            extra = sorted(set(variant_keys) - set(baseline_keys))
            raise ValueError(
                f"checkpoint tensor keys differ: missing={missing[:3]}, extra={extra[:3]}"
            )
        for name in baseline_keys:
            baseline_tensor = baseline_file.get_tensor(name)
            variant_tensor = variant_file.get_tensor(name)
            if np.array_equal(baseline_tensor, variant_tensor):
                continue
            delta = variant_tensor.astype(np.float64) - baseline_tensor.astype(np.float64)
            changed.append(
                {
                    "name": name,
                    "category": name.rsplit(".", 1)[-1],
                    "shape": list(baseline_tensor.shape),
                    "dtype": str(baseline_tensor.dtype),
                    "baseline_nonzero": int(np.count_nonzero(baseline_tensor)),
                    "variant_nonzero": int(np.count_nonzero(variant_tensor)),
                    "variant_all_zero": not bool(np.any(variant_tensor)),
                    "max_abs_delta": float(np.max(np.abs(delta))),
                }
            )
    categories = Counter(item["category"] for item in changed)
    return {
        "variant": str(variant),
        "variant_sha256": file_sha256(variant),
        "tensor_count": len(baseline_keys),
        "changed_tensor_count": len(changed),
        "unchanged_tensor_count": len(baseline_keys) - len(changed),
        "changed_categories": dict(sorted(categories.items())),
        "all_changed_tensors_zeroed": all(
            item["variant_all_zero"] for item in changed
        ),
        "changed_tensors": changed,
    }


def audit_checkpoints(
    baseline: Path, variants: list[tuple[str, Path]]
) -> dict[str, Any]:
    if not variants:
        raise ValueError("at least one variant is required")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "baseline": str(baseline),
        "baseline_sha256": file_sha256(baseline),
        "interpretation": (
            "post-hoc inference-time tensor intervention; not a separately trained "
            "causal architecture ablation"
        ),
        "variants": {},
    }
    seen = set()
    for tag, path in variants:
        if tag in seen:
            raise ValueError(f"duplicate variant tag: {tag}")
        seen.add(tag)
        payload["variants"][tag] = audit_variant(baseline, path)
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint tensor-delta audit",
        "",
        f"Baseline: `{payload['baseline']}`",
        "",
        "These are post-hoc inference-time tensor interventions, not separately",
        "trained causal architecture ablations.",
        "",
        "| Variant | Changed / total tensors | Changed categories | All changed zeroed |",
        "|---|---:|---|:---:|",
    ]
    for tag, item in sorted(payload["variants"].items()):
        categories = ", ".join(
            f"{name}={count}" for name, count in item["changed_categories"].items()
        )
        lines.append(
            f"| {tag} | {item['changed_tensor_count']} / {item['tensor_count']} | "
            f"{categories} | {'yes' if item['all_changed_tensors_zeroed'] else 'NO'} |"
        )
    return "\n".join(lines) + "\n"


def parse_variant(value: str) -> tuple[str, Path]:
    tag, path = value.split("=", 1)
    if not tag or not path:
        raise ValueError(f"invalid variant specification: {value!r}")
    return tag, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", action="append", default=[], metavar="TAG=PATH")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit_checkpoints(
        args.baseline, [parse_variant(value) for value in args.variant]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoint_delta_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (args.out / "CHECKPOINT_DELTA_AUDIT.md").write_text(render_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
