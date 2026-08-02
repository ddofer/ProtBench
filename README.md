# ProtBench

One benchmark harness for protein models: frozen probes, fine-tuning, and
non-model baselines, over 42 sequence-level and residue-level tasks.

Everything runs through one CLI, so a PLM, a LoRA fine-tune, an alignment
search and a k-mer count are scored on identical splits with identical
metrics and land in the same CSV.

```bash
# Frozen linear probe on one task
python protein_benchmark_suite.py -m facebook/esm2_t6_8M_UR50D \
    --tasks signalp_binary -p linear --eval_split test

# The no-learning floor, same splits and metrics
python protein_benchmark_suite.py -m kmer --tasks signalp_binary -p linear

# With 95% bootstrap CIs on every label-only metric
python protein_benchmark_suite.py -m <model> --tasks stability --bootstrap 1000
```

## What is here

| Component | Entry point |
|---|---|
| Frozen probes (linear / histgb / knn) | `protein_benchmark_suite.py` |
| Residue-level probes | `token_classification_probe.py` (via `--tasks ss3 disorder ...`) |
| Sequence fine-tuning (full / last-N / LoRA) | `finetune_sequence.py` |
| Residue fine-tuning | `finetune_residue.py` |
| ProteinGym zero-shot | `proteingym_mlm_zeroshot.py`, `zero_shot_dms.py` |
| Test-time training | `wt_test_time_training.py` |
| Task registry (42 tasks) | `benchmark_tasks.py` |
| mmseqs2 baseline | `mmseqs_baseline.py` |
| phmmer baseline | `hmmer_baseline.py` |
| k-mer baseline | `kmer_baseline.py`, or `-m kmer` / `-m kmer4` |
| Bootstrap CIs | `--bootstrap N`; retrieval CIs in `bootstrap_ci.py` |

## Origin, and the other copies

Seeded from proteva's `plm/bench` (72 commits of history preserved via
`git subtree split`), which was the most complete of five divergent copies of
this suite that had accumulated across machines. Grafted on top: the
cross-process embedding-cache fix from
`ProteinSentenceTransformers@33feae0`, and the mmseqs2 / phmmer baselines
from `oriel9p/ProtSent@rebuttal`.

**The older copies still exist and disagree with this one.** They carry 28
tasks against this repo's 42, and `/home/ddofer/protein` still treats
`ec_classification` as multiclass/`F1_Macro` where this repo has it as
multilabel/`F1_Micro`. Concatenating result CSVs across that boundary mixes
metric definitions under a single column name. The suite logs its own
resolved path and task count at startup for exactly this reason — check that
line before trusting a results directory.

`protJepa/run_pretrained_benchmarks.py` resolves this repo automatically;
`$PROTBENCH_HOME` overrides the search.

## Defaults worth knowing

- **Embedding caching is off.** Opt in with `--cache_embeddings`. The cache
  key for a hub model id is just the model name, so it does *not* invalidate
  when upstream weights change; local paths are keyed on size and mtime and
  do invalidate correctly.
- **`seed_all()` seeds stdlib `random`, numpy and torch** on top of the
  `BENCHMARK_SEED` that is threaded into sklearn and `datasets.shuffle`.
  `torch.use_deterministic_algorithms` is deliberately *not* set — it breaks
  or badly slows several embedding kernels, and embedding is inference-only.
- **`--bootstrap N` covers label-only metrics.** AUC and AP are computed from
  `predict_proba` after the fact and get no interval. Do not combine it with
  the cross-validation path: fold aggregation averages every numeric column,
  which would turn the CIs into a mean of intervals.
- **`METRIC_PRIORITY` is frozen.** MCC and balanced accuracy are reported but
  deliberately excluded from it, so adding them did not re-rank any
  historical comparison.

## Install

```bash
uv sync                      # probes only
uv sync --extra finetune     # adds peft, for the LoRA paths
```

Python >= 3.10. `mmseqs` is a system binary (`$MMSEQS_BIN` or `PATH`);
phmmer comes from the `pyhmmer` package. Neither is required for the probe
paths.

## Tests

```bash
pytest -m "not slow"    # 152 tests, no network
pytest                  # adds tests that download models
```

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

### Dataset provenance (verified 2026-05-21)

| Task | Dataset | Source paper | Caveats |
|---|---|---|---|
| `ss3`, `disorder` | [`agemagician/NetSurfP-SS3`](https://huggingface.co/datasets/agemagician/NetSurfP-SS3) | Klausen et al., *NetSurfP-2.0*, Proteins 2019 ([doi:10.1002/prot.25674](https://doi.org/10.1002/prot.25674)) |  Train/val reshuffled from paper's 10,337/500 to 10,792/646. **CB513 = 511 chains** (vs 513 in paper); **CASP12 = 20** (vs 21). Disorder = PDB-missing-coordinate mask, NOT DisProt / CAID2 — do not cross-compare. |
| `signal_peptide` | [`SaProtHub/Dataset-Signal-Peptides`](https://huggingface.co/datasets/SaProtHub/Dataset-Signal-Peptides) | Teufel et al., *SignalP 6.0*, Nat Biotechnol 2022 ([doi:10.1038/s41587-021-01156-3](https://doi.org/10.1038/s41587-021-01156-3)) | All 25,693 rows packed into HF `train` split; partition is in the `stage` column (20,490 / 2,569 / 2,634).|
| `disprot` | [`LiteFold/DisProt`](https://huggingface.co/datasets/LiteFold/DisProt) | Aspromonte et al., *DisProt 2024*, NAR ([doi:10.1093/nar/gkad928](https://doi.org/10.1093/nar/gkad928)) | Built to local Arrow `data/disprot/` by `scripts/prep_disprot.py`. Per-residue 0/1 = union of curated `region_terms == 'disorder'` spans. Split via DisProt's deterministic `split_bucket` (sha256(id)%10): test=bucket0 (324), val=bucket1 (340), train=buckets2-9 (2,535). **This is the manually-curated CAID-style target — distinct from the NetSurfP `disorder` mask above.** Headline metric MCC (imbalanced, ~17% disordered). |


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
