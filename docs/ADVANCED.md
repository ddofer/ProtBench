# Advanced usage

Fine-tuning, ProteinGym zero-shot and test-time training. Assumes you have
already run a plain benchmark; start with the [README](../README.md).

## Unified harness + delta-vs-vanilla comparison

To get probe, ProteinGym and LoRA results into one table, run
`protein_benchmark_suite.py`, `proteingym_mlm_zeroshot.py` and
`finetune_sequence.py` separately, then collect their outputs.

Pipeline: those runs → `collect_bench_results.py` → the unified
`results/bench_results_all.csv` (one row per model × task × probe × split ×
metric; the dedup key is `(model, notes, task, probe, split, metric)` so a
corrected re-run **overwrites** the stale row). Read it as a delta-vs-vanilla
table with:

```bash
# pivot every (probe, task) to vanilla AMPLIFY + each model + Δ-vs-vanilla
python contrib/proteva/compare_to_vanilla.py --split test            # all probes
python contrib/proteva/compare_to_vanilla.py --split test --probe lora
```

**Humanized report (start here).** `contrib/proteva/report.py` is the one-command, DS-friendly
view — run it any time to refresh two artifacts in `results/`:

```bash
python contrib/proteva/report.py            # writes results/BENCH_REPORT.md + bench_pivot.csv
```

- **`results/BENCH_REPORT.md`** — markdown a human reads: linear / LoRA / MLM
  trajectory tables (vanilla → step0 → epoch1 → epoch3) with ↑/↓ vs vanilla,
  plus per-model win/loss tallies, top lifts and top regressions.
- **`results/bench_pivot.csv`** — wide task×model table for interactive work:
  `pandas.read_csv("results/bench_pivot.csv")` and slice.

`report.py` reuses `compare_to_vanilla.build_comparison`, so the baseline/Δ
logic is single-source. `--probe`/`--split` restrict to one view.

### ProteinGym zero-shot — 4 benchmarks, MLM-family scoring

`proteingym_mlm_zeroshot.py` scores all **4** ProteinGym benchmarks (default
`--tasks` = all 4), so AUC is available for every one:

| Benchmark | Scorer | Metric | Sign |
|---|---|---|---|
| DMS substitutions | masked-marginal Σ per-pos logP delta | Spearman (+ median-binarized AUC) | higher = fitter |
| DMS indels | pseudo-log-likelihood `PLL(mut)−PLL(WT)` | AUC (median-binarized) | higher = fitter |
| Clinical substitutions | masked-marginal | AUC | **negated** (pathogenic = lower logP) |
| Clinical indels | pseudo-log-likelihood | AUC | **negated** |

PLL = Σᵢ log P(seqᵢ \| seq with i masked) over the whole sequence — the
encoder/masked-LM analogue of a sequence likelihood (ESM-1v/ESM2 style); it
needs no position alignment, so indels are scorable. `--max_variants_per_assay`
(default 200) bounds the per-mutant PLL forward passes on large DMS-indel
assays (substitutions share the WT table and are unaffected). The cosine
zero-shot path is a weak proxy (clinical AUC ~0.68 after the sign fix vs MLM
~0.90) and is **default-off** (`PLM_BENCH_PGYM_COSINE=1` to restore — it is the
only path that *also* re-scores indels via embedding cosine).

## Fine-tuning scripts (residue + sequence; LoRA)

Two on-demand HF-Trainer wrappers live alongside the linear-probe path:

- `finetune_residue.py` — token classification (SS3, SS8, intrinsic disorder,
  signal peptides, conservation). Modes: `probe` (frozen, default), `full`, `lora`, `last_n`.
- `finetune_sequence.py` — sequence-level fine-tuning for any task in
  `TASKS` whose `problem_type` is `binary / multiclass / regression`.
  Modes: `probe` (default), `full`, `lora` (PEFT), `last_n`.

**LoRA defaults:**
Default `r=64 α=64 lr=2e-4 patience=1`. Per-task-type overrides:
- **Regression** (beta_lactamase, stability, fluorescence, meltome, …): `r=16 α=32 lr=1e-4 --fp32` — small sets overfit quickly; fp32 for RoPE precision.
- **Many-class classification** (remote_homology=1195 classes, ec_classification): `lr=1e-4 patience=3` — aggressive default causes loss to stay at ln(N) for the full 1-epoch grace period and patience=1 restores the near-random checkpoint.
- **Other classification** (binary, low-cardinality multiclass): default.


```bash
PY=python

# Residue probe on SS3:
CUDA_VISIBLE_DEVICES=0 $PY finetune_residue.py \
    --model_name chandar-lab/AMPLIFY_120M \
    --task ss3 --mode probe --max_length 512 \
    --output_dir results/

# All residue tasks sequentially:
CUDA_VISIBLE_DEVICES=0 $PY finetune_residue.py \
    --model_name chandar-lab/AMPLIFY_120M --task all

# Sequence LoRA on stability (best-practice r=32/alpha=64, 1 epoch):
CUDA_VISIBLE_DEVICES=0 $PY finetune_sequence.py \
    --model_name chandar-lab/AMPLIFY_120M \
    --task stability --mode lora --lora_r 32 --lora_alpha 64 --num_train_epochs 1
```

**One-time setup** — install the fine-tuning extra:
```bash
uv sync --extra finetune
```

Output: one JSONL line per `(checkpoint, task)` appended to
`<output_dir>/finetune_<script>_<safe_ckpt>_<task>.jsonl`. LoRA mode
also saves adapter weights under `.../lora_adapter/`.

### Dataset provenance

Moved to [DATASETS.md](DATASETS.md).

## Tests

Unit tests live in [`tests/`](../tests/). Fast tests cover the
`TaskConfig` validator extension, label decoders, and label alignment.

```bash
PY=python

# Fast unit tests only:
$PY -m pytest tests/ -m "not slow"

# Including the CPU smoke tests (needs the finetune extra):
CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/ -m slow
```

## Quick examples (Stage-2 search: fast subsets, clean signal)

```bash
PY=python
BENCH=protein_benchmark_suite.py

# One task, fast mode, sample cap (default --fast already caps at FAST_MAX_SAMPLES=100k):
CUDA_VISIBLE_DEVICES=0 $PY $BENCH --model_name <ckpt> --tasks stability --max_samples 5000

# Three correlated tasks (stability + solubility + beta_lactamase_peer):
CUDA_VISIBLE_DEVICES=0 $PY $BENCH --model_name <ckpt> --tasks stability solubility beta_lactamase_peer --max_samples 5000

# Residue-level disorder (see the caveat below -- two different tasks are
# both called "disorder"):
CUDA_VISIBLE_DEVICES=0 $PY $BENCH --model_name <ckpt> --tasks disprot

# Compare two CSVs from prior runs:
$PY $BENCH --compare --compare_model1 results/benchmarks/bench_<a>.csv \
                     --compare_model2 results/benchmarks/bench_<b>.csv
```

**`chezod_disorder` is no longer registered** -- it was disabled in favour of
the residue-level `disprot` task and is commented out in `benchmark_tasks.py`.
Passing it now fails with `invalid choice`. Two live tasks are called
"disorder": `disorder` (a PDB missing-coordinate mask, from NetSurfP) and
`disprot` (manually curated, CAID-style). They are not comparable to each
other; see [DATASETS.md](DATASETS.md).

Pass `--help` for the full list of flags (probe type, eval-split,
`--max_samples`, embedding cache, `--seed_list`, …).

## HF cache

Datasets are loaded via `datasets.load_dataset(...)` with the **default**
cache (`~/.cache/huggingface/hub/`), which already contains every
`biomap-research/*` corpus this suite uses. No env-var setup needed. We
verified no explicit `cache_dir=` override exists in the vendored code — the
only `cache_dir` references in the files are for the *embedding-output* cache
controlled by `--embed_cache_dir`, which is unrelated to dataset downloads.

