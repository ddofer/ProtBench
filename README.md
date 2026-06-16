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

## Unified harness + delta-vs-vanilla comparison

For a full model report (the way the AMPLIFY-vanilla / step-0 / trained-epoch
comparison is run), use the one-shot harness instead of the scripts below:

```bash
# probe (linear) + ProteinGym (cosine + MLM masked-marginal) + LoRA (val+test),
# all collected into ONE long CSV. GPU arg must NOT be in manifest.lock.
bash plm/scripts/run_full_bench.sh <ckpt-or-hf-id> "<notes>" <gpu>
```

Pipeline: `run_full_bench.sh` → `collect_bench_results.py` → the unified
`results/bench_results_all.csv` (one row per model × task × probe × split ×
metric; the dedup key is `(model, notes, task, probe, split, metric)` so a
corrected re-run **overwrites** the stale row). Read it as a delta-vs-vanilla
table with:

```bash
# pivot every (probe, task) to vanilla AMPLIFY + each model + Δ-vs-vanilla
python plm/bench/compare_to_vanilla.py --split test            # all probes
python plm/bench/compare_to_vanilla.py --split test --probe lora
```

The harness runs on proteva's own venv (`plm/.venv`, Python 3.12, `peft`
already installed) — NOT the sibling venv described below, which applies only
to calling `protein_benchmark_suite.py` directly.

## Fine-tuning scripts (residue + sequence; LoRA)

Two on-demand HF-Trainer wrappers live alongside the linear-probe path:

- `finetune_residue.py` — token classification (SS3, intrinsic disorder,
  signal peptides). Modes: `probe` (frozen, default), `full`, `lora`, `last_n`.
- `finetune_sequence.py` — sequence-level fine-tuning for any task in
  `TASKS` whose `problem_type` is `binary / multiclass / regression`.
  Modes: `probe` (default), `full`, `lora` (PEFT), `last_n`.

**LoRA config (modern PEFT best practice, set in `run_full_bench.sh`):**
`--lora_r 32 --lora_alpha 64` (α = 2r), 1 epoch / keep-last (regression tasks
overfit past 1 epoch), `lr 1e-4`. `target_modules` are resolved per model
family in `_hf_finetune_common.py` to **all body Linears** — Proteva
`wq/wk/wv/wo/attn_gate/w12/w3`, AMPLIFY `q/k/v/wo/w12/w3` (attention + SwiGLU
FFN); the MLM decoder + pretraining aux heads stay frozen, the task classifier
trains via `modules_to_save`. NOT `all-linear` (that would wrap the frozen
heads).

```bash
PY=/data/prot/ProteinSentenceTransformers/.venv/bin/python

# Residue probe on SS3:
CUDA_VISIBLE_DEVICES=4 $PY /data/proteva/plm/bench/finetune_residue.py \
    --model_name chandar-lab/AMPLIFY_120M \
    --task ss3 --mode probe --max_length 512 \
    --output_dir /data/proteva/plm/results/bench/

# All three residue tasks sequentially:
CUDA_VISIBLE_DEVICES=4 $PY /data/proteva/plm/bench/finetune_residue.py \
    --model_name chandar-lab/AMPLIFY_120M --task all

# Sequence LoRA on stability (best-practice r=32/alpha=64, 1 epoch):
CUDA_VISIBLE_DEVICES=4 $PY /data/proteva/plm/bench/finetune_sequence.py \
    --model_name chandar-lab/AMPLIFY_120M \
    --task stability --mode lora --lora_r 32 --lora_alpha 64 --num_train_epochs 1
```

**One-time setup** — `peft` and `seqeval` are not in the sibling venv.
Install them once before first use:

```bash
/data/prot/ProteinSentenceTransformers/.venv/bin/pip install \
    "peft>=0.13" "seqeval>=1.2"
```

(`evaluate` is optional; the scripts use `sklearn` / `scipy` directly.)

Output: one JSONL line per `(checkpoint, task)` appended to
`<output_dir>/finetune_<script>_<safe_ckpt>_<task>.jsonl`. LoRA mode
also saves adapter weights under `.../lora_adapter/`.

### Dataset provenance (verified 2026-05-21)

| Task | Dataset | Source paper | Caveats |
|---|---|---|---|
| `ss3`, `disorder` | [`agemagician/NetSurfP-SS3`](https://huggingface.co/datasets/agemagician/NetSurfP-SS3) | Klausen et al., *NetSurfP-2.0*, Proteins 2019 ([doi:10.1002/prot.25674](https://doi.org/10.1002/prot.25674)) | Third-party rehost by Ahmed Elnaggar (ProtTrans author). Train/val reshuffled from paper's 10,337/500 to 10,792/646. **CB513 = 511 chains** (vs 513 in paper); **CASP12 = 20** (vs 21). Disorder = PDB-missing-coordinate mask, NOT DisProt / CAID2 — do not cross-compare. No license on HF card. |
| `signal_peptide` | [`SaProtHub/Dataset-Signal-Peptides`](https://huggingface.co/datasets/SaProtHub/Dataset-Signal-Peptides) | Teufel et al., *SignalP 6.0*, Nat Biotechnol 2022 ([doi:10.1038/s41587-021-01156-3](https://doi.org/10.1038/s41587-021-01156-3)) | All 25,693 rows packed into HF `train` split; partition is in the `stage` column (20,490 / 2,569 / 2,634). Third-party rehost by SaProtHub. License: MIT. |

## Tests

Unit tests live in [`plm/bench/tests/`](tests/). Fast tests cover the
`TaskConfig` validator extension, label decoders, and label alignment.
Two CPU smoke tests (model + dataset download) are marked
`@pytest.mark.slow` and skipped by default.

```bash
PY=/data/prot/ProteinSentenceTransformers/.venv/bin/python

# Fast unit tests only:
$PY -m pytest /data/proteva/plm/bench/tests/ -m "not slow"

# Including the CPU smoke tests (needs peft + seqeval installed):
CUDA_VISIBLE_DEVICES="" $PY -m pytest /data/proteva/plm/bench/tests/ -m slow
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

## Structural test sets are also leakage sources

The structural tasks in this suite (`ss3`, `disorder`, `stability`,
`chezod_disorder`) are first-class **leakage protection targets** when
proteva pretrains with 3Di / fold-derived aux objectives. Their test
sequences feed the train-side filtering pipeline documented at
[`plm/docs/LEAKAGE_FILTERING_RUNBOOK.md`](../docs/LEAKAGE_FILTERING_RUNBOOK.md);
they are included by default in `--task-group critical` and selectable
in isolation via the new `--task-group structural`. PSSM-derived tasks
are intentionally not treated as a structural leak.

## Updating

To pull a newer revision from the sibling repo, re-run the same `cp`
operations from `/data/prot/ProteinSentenceTransformers/` (no patches were
applied to the copied files). If the sibling adds new local-data
dependencies, mirror them under `./data/` as symlinks first.
