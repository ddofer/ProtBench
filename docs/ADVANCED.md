# Advanced usage

Inherited from the proteva benchmark directory this repo was seeded from.
Accurate, but aimed at someone already running pretraining sweeps. Start
with the [README](../README.md).

## ⚠ Which checkpoint dir to bench (`_orig_mod.` prefix)

**A `_orig_mod.`-prefixed checkpoint benchmarks as a RANDOM model — it does not
crash.** `torch.compile` prefixes every state-dict key, `from_pretrained` then
matches nothing, and you get plausible garbage: AUC pinned **0.5000**, Spearman
**0.0000**, Recall@10 ~0.01. That reads as "the model is broken"; it is not —
the *checkpoint path you picked* is.

| Path | Prefixed? | Bench it? |
|---|---|---|
| `<out>/` (final root save) | **no** — `run_stage2.py` strips it | ✅ **prefer this** |
| `<out>/checkpoint-N/` (periodic save) | **yes** — written from the compiled module | ⚠ only if you must |

Check any checkpoint in one line (no project imports, always works):

```bash
python -c "from safetensors import safe_open; f=safe_open('<ckpt>/model.safetensors','pt'); \
print(sum(k.startswith('_orig_mod.') for k in f.keys()), 'prefixed keys')"
```

Guards, in order (all reuse `plm/hf/checkpoint_utils.strip_orig_mod_prefix`):

- **Auto-strip on load** — `model_utils.from_pretrained_with_flash` *and* the
  Proteva branch of `protein_benchmark_suite.load_model`. Best-effort: it is a
  no-op on **sharded** checkpoints and its failures are logged, not raised.
- **Hard gate** — `_WRAPPER_PREFIXES` / `_validate_local_checkpoint_integrity`
  **raise** on any surviving `_orig_mod.` key or <50% key overlap. **Do not
  remove the plain `_orig_mod.` entry.**
- Tests: `tests/test_orig_mod_guard.py` (the gate),
  `plm/tests/bench/test_orig_mod_prefix_load.py` (strip + end-to-end loader).

Calling the strip helper standalone needs the repo root *and* `plm/` on the path
(`plm/uc30_aux_loader.py` imports `scripts.data.*`, which lives at
`plm/scripts/`), else it dies with `ModuleNotFoundError: No module named
'scripts'`:

```bash
# Run this from a proteva checkout -- strip_orig_mod_prefix lives there, not here.
PYTHONPATH=plm python -c \
  "from plm.hf.checkpoint_utils import strip_orig_mod_prefix as s; print(s('<ckpt>'))"
```

Incidents: 2026-07-21 v6-rtd re-bench (~3.5 GPU-h of chance-level results, caused
by benching `checkpoint-105000` after a swallowed strip failure), plus earlier
repeats.

## Unified harness + delta-vs-vanilla comparison

The full model report (probe + ProteinGym + LoRA, all collected into ONE long
CSV) was driven by `run_full_bench.sh`, which lived in proteva and did not
move here. Drive `protein_benchmark_suite.py`, `proteingym_mlm_zeroshot.py`
and `finetune_sequence.py` individually, then run the collector below over
their outputs.

Pipeline: those runs → `collect_bench_results.py` → the unified
`results/bench_results_all.csv` (one row per model × task × probe × split ×
metric; the dedup key is `(model, notes, task, probe, split, metric)` so a
corrected re-run **overwrites** the stale row). Read it as a delta-vs-vanilla
table with:

```bash
# pivot every (probe, task) to vanilla AMPLIFY + each model + Δ-vs-vanilla
python compare_to_vanilla.py --split test            # all probes
python compare_to_vanilla.py --split test --probe lora
```

**Humanized report (start here).** `report.py` is the one-command, DS-friendly
view — run it any time to refresh two artifacts in `results/`:

```bash
python report.py            # writes results/BENCH_REPORT.md + bench_pivot.csv
```

- **`results/BENCH_REPORT.md`** — markdown a human reads: linear / LoRA / MLM
  trajectory tables (vanilla → step0 → epoch1 → epoch3) with ↑/↓ vs vanilla,
  plus per-model win/loss tallies, top lifts and top regressions.
- **`results/bench_pivot.csv`** — wide task×model table for interactive work:
  `pandas.read_csv("results/bench_pivot.csv")` and slice.

`report.py` reuses `compare_to_vanilla.build_comparison`, so the baseline/Δ
logic is single-source. `--probe`/`--split` restrict to one view.

The harness runs on proteva's own venv (`plm/.venv`, Python 3.12, `peft`
already installed) — NOT the sibling venv described below, which applies only
to calling `protein_benchmark_suite.py` directly.

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

- `finetune_residue.py` — token classification (SS3, intrinsic disorder,
  signal peptides). Modes: `probe` (frozen, default), `full`, `lora`, `last_n`.
- `finetune_sequence.py` — sequence-level fine-tuning for any task in
  `TASKS` whose `problem_type` is `binary / multiclass / regression`.
  Modes: `probe` (default), `full`, `lora` (PEFT), `last_n`.

**LoRA config (set in `run_full_bench.sh`):**
Default `r=64 α=64 lr=2e-4 patience=1`. Per-task-type overrides:
- **Regression** (beta_lactamase, stability, fluorescence, meltome, …): `r=16 α=32 lr=1e-4 --fp32` — small sets overfit quickly; fp32 for RoPE precision.
- **Many-class classification** (remote_homology=1195 classes, ec_classification): `lr=1e-4 patience=3` — aggressive default causes loss to stay at ln(N) for the full 1-epoch grace period and patience=1 restores the near-random checkpoint (F1≈0.0003 verified 2026-06-16).
- **Other classification** (binary, low-cardinality multiclass): default.


```bash
PY=python

# Residue probe on SS3:
CUDA_VISIBLE_DEVICES=4 $PY finetune_residue.py \
    --model_name chandar-lab/AMPLIFY_120M \
    --task ss3 --mode probe --max_length 512 \
    --output_dir results/

# All three residue tasks sequentially:
CUDA_VISIBLE_DEVICES=4 $PY finetune_residue.py \
    --model_name chandar-lab/AMPLIFY_120M --task all

# Sequence LoRA on stability (best-practice r=32/alpha=64, 1 epoch):
CUDA_VISIBLE_DEVICES=4 $PY finetune_sequence.py \
    --model_name chandar-lab/AMPLIFY_120M \
    --task stability --mode lora --lora_r 32 --lora_alpha 64 --num_train_epochs 1
```

**One-time setup** — `peft` and `seqeval` are not in the sibling venv.
Install them once before first use:

```bash
pip install \
    "peft>=0.13" "seqeval>=1.2"
```

(`evaluate` is optional; the scripts use `sklearn` / `scipy` directly.)

Output: one JSONL line per `(checkpoint, task)` appended to
`<output_dir>/finetune_<script>_<safe_ckpt>_<task>.jsonl`. LoRA mode
also saves adapter weights under `.../lora_adapter/`.

### Dataset provenance

Moved to [DATASETS.md](DATASETS.md).

## Tests

Unit tests live in [`tests/`](tests/). Fast tests cover the
`TaskConfig` validator extension, label decoders, and label alignment.

```bash
PY=python

# Fast unit tests only:
$PY -m pytest tests/ -m "not slow"

# Including the CPU smoke tests (needs peft + seqeval installed):
CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/ -m slow
```

## Quick examples (Stage-2 search: fast subsets, clean signal)

```bash
PY=python
BENCH=protein_benchmark_suite.py

# One task, fast mode, sample cap (default --fast already caps at FAST_MAX_SAMPLES=100k):
CUDA_VISIBLE_DEVICES=4 $PY $BENCH --model_name <ckpt> --tasks stability --max_samples 5000

# Three correlated tasks (stability + solubility + beta_lactamase_peer):
CUDA_VISIBLE_DEVICES=4 $PY $BENCH --model_name <ckpt> --tasks stability solubility beta_lactamase_peer --max_samples 5000

# Residue-level disorder (see the caveat below -- two different tasks are
# both called "disorder"):
CUDA_VISIBLE_DEVICES=4 $PY $BENCH --model_name <ckpt> --tasks disprot

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

## Structural test sets are also leakage sources

The structural tasks in this suite (`ss3`, `disorder`, `stability`,
`chezod_disorder`) are first-class **leakage protection targets** when
proteva pretrains with 3Di / fold-derived aux objectives. Their test
sequences feed the train-side filtering pipeline documented at
proteva's `docs/LEAKAGE_FILTERING_RUNBOOK.md` (not part of this repo);
they are included by default in `--task-group critical` and selectable
in isolation via the new `--task-group structural`. PSSM-derived tasks
are intentionally not treated as a structural leak.

## Updating

This repo is the canonical copy — edit it here. The older checkouts
(`ProteinSentenceTransformers`, `protein`, `proteva/plm/bench`) are frozen
history, kept only so in-flight runs that still point at them keep working.
Changes do not flow back to them, and a fix landed there will not reach here.
