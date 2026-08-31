"""Compare two persisted PTM-site panels with a paired protein bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ptm_benchmark import read_ptm_predictions
from ptm_compare import paired_site_auprc_bootstrap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    candidate_metadata, candidate_sites, _ = read_ptm_predictions(args.candidate)
    baseline_metadata, baseline_sites, _ = read_ptm_predictions(args.baseline)
    report = paired_site_auprc_bootstrap(
        candidate_sites,
        baseline_sites,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    report.update(
        {
            "schema_version": 1,
            "candidate": str(args.candidate.resolve()),
            "candidate_metadata": candidate_metadata,
            "baseline": str(args.baseline.resolve()),
            "baseline_metadata": baseline_metadata,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
