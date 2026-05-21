# `plm/bench` — vendored protein-benchmark suite

## Origin

Vendored from the sibling repo `/data/prot/ProteinSentenceTransformers` at
git SHA **`7aa984e`** (same user owns both repos; no LICENSE in source).

Files copied verbatim (no edits):

| File | Role |
|---|---|
| `protein_benchmark_suite.py` | Main CLI entry point |
| `benchmark_tasks.py` | `TASKS` registry (~35 protein tasks) |
| `benchmark_utils.py` | Shared utilities + result-IO helpers |
| `benchmark_comparison.py` | Pairwise model comparison helpers |
| `model_utils.py` | Model load + AMPLIFY/ESMplusplus detection |
| `attention_pooling.py` | Pooling module (unused by the main CLI but kept for completeness) |

Plus `data/chezod` → symlink into the sibling repo's CheZoD corpus.

## Why vendored

Proteva can now run the full benchmark suite on its own without depending on
a separate working tree. The vendored code reuses the **existing** HF dataset
cache (`~/.cache/huggingface/hub/` — already populated with all the
`biomap-research/*` corpora) and the sibling repo's local CheZoD dataset via
a symlink. **No re-downloads required.**

## Runtime — must use the sibling venv

The suite depends on heavy ML libs (`sentence-transformers`,
`transformers>=5`, `torch>=2.10`, `scikit-learn`, `scipy`, `datasets`).
**Proteva's own venv is Python 3.11 and pins different versions.** Always
invoke the suite with the sibling repo's Python 3.13 venv:

```bash
/data/prot/ProteinSentenceTransformers/.venv/bin/python \
    /data/proteva/plm/bench/protein_benchmark_suite.py [args...]
```

Do **not** install these deps into proteva's `pyproject.toml`.

## GPU safety (standing CLAUDE.md rule)

Per the project's GPU-safety rule, never co-tenant a GPU listed in
`/data/proteva/plm/manifest.lock`. Choose a free GPU explicitly, or run on
CPU for smokes:

```bash
# Smoke (CPU only):
CUDA_VISIBLE_DEVICES="" /data/prot/ProteinSentenceTransformers/.venv/bin/python \
    /data/proteva/plm/bench/protein_benchmark_suite.py --help

# Real eval on a free GPU (replace 4 with a GPU NOT in manifest.lock):
CUDA_VISIBLE_DEVICES=4 /data/prot/ProteinSentenceTransformers/.venv/bin/python \
    /data/proteva/plm/bench/protein_benchmark_suite.py --model_name <ckpt> --tasks stability
```

## Quick examples (Stage-2 search: fast subsets, clean signal)

```bash
PY=/data/prot/ProteinSentenceTransformers/.venv/bin/python
BENCH=/data/proteva/plm/bench/protein_benchmark_suite.py

# One task, fast mode, sample cap (default --fast already caps at FAST_MAX_SAMPLES=100k):
CUDA_VISIBLE_DEVICES=4 $PY $BENCH --model_name <ckpt> --tasks stability --max_samples 5000

# Three correlated tasks (stability + solubility + beta_lactamase_peer):
CUDA_VISIBLE_DEVICES=4 $PY $BENCH --model_name <ckpt> --tasks stability solubility beta_lactamase_peer --max_samples 5000

# Disorder-only (CheZoD via the local symlinked dataset):
CUDA_VISIBLE_DEVICES=4 $PY $BENCH --model_name <ckpt> --tasks chezod_disorder

# Compare two CSVs from prior runs:
$PY $BENCH --compare --compare_model1 results/benchmarks/bench_<a>.csv \
                     --compare_model2 results/benchmarks/bench_<b>.csv
```

The task key for the CheZoD benchmark is **`chezod_disorder`** (mean-Z-score
regression — the std-Z variant is commented out in the upstream registry).

Pass `--help` for the full list of flags (probe type, eval-split,
`--max_samples`, embedding cache, `--seed_list`, …).

## HF cache

Datasets are loaded via `datasets.load_dataset(...)` with the **default**
cache (`~/.cache/huggingface/hub/`), which already contains every
`biomap-research/*` corpus this suite uses. No env-var setup needed. We
verified no explicit `cache_dir=` override exists in the vendored code — the
only `cache_dir` references in the files are for the *embedding-output* cache
controlled by `--embed_cache_dir`, which is unrelated to dataset downloads.

## Not in the default training/search loop

This is an **on-demand** evaluation tool. It is intentionally not wired into
any autoresearch path, `train_proxy.py`, or the queue runner. Call it
manually during Stage-2 search to spot-check pretraining objectives against
downstream signal.

## Updating

To pull a newer revision from the sibling repo, re-run the same `cp`
operations from `/data/prot/ProteinSentenceTransformers/` (no patches were
applied to the copied files). If the sibling adds new local-data
dependencies, mirror them under `./data/` as symlinks first.
