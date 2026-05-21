"""Vendored protein-benchmark suite (sibling repo ProteinSentenceTransformers @ 7aa984e).

See README.md for usage, runtime requirements, and the rationale for keeping
this directory as a thin self-contained copy that reuses the existing HF cache
and local data (via symlinks under ``./data``).

This package is intentionally not imported by the proteva training/search loop.
It is an on-demand evaluation tool invoked directly via
``protein_benchmark_suite.py --tasks ...``.
"""
