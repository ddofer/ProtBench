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
`--tasks` = all 4):

| Benchmark | Scorer | Metric | Sign |
|---|---|---|---|
| DMS substitutions | masked-marginal Σ per-pos logP delta | Spearman (hierarchical: UniProt → functional category) | higher = fitter |
| DMS indels | strided masked PLL, `PLL(mut)−PLL(WT)` | Spearman | higher = fitter |
| Clinical substitutions | masked-marginal | AUC (per-gene mean) | **negated** (pathogenic = lower logP) |
| Clinical indels | strided masked PLL | AUC (pooled across genes) | **negated** |

AUC is reported for the DMS benchmarks only where the official `DMS_score_bin`
column exists; there is no median-split fallback.

**Cost, and why the two task shapes differ.** Substitutions amortize: one masked
forward at a position yields log-probs for every amino acid there, so all
variants at that position are scored from it. The scorer takes the union of
mutated positions per assay, so DMS substitutions (217 assays, 2.47M variants)
cost **~86k forwards**, and clinical substitutions (2,525 genes, 62.7k variants)
fewer still. Both run in wall-clock minutes. `--max_variants_per_assay` (default
`None` = all variants, leaderboard-faithful) does not change substitution cost.

Indels cannot amortize — every variant is a distinct sequence — so cost is
`forwards per variant × variants`:

| `--indel_score_mode` | forwards/variant | DMS indels (287k variants) | note |
|---|---|---|---|
| `strided` (default) | `--indel_pll_passes` (32) | ~9.2M, ~7 h | leakage-free few-pass masked PLL |
| `masked_pll` | L | ~10⁸ | exact, unusable at this scale |
| `single_pass` | 1 | ~287k | **broken by leakage** (Spearman 0.50→0.31); reference only |
| `embedding_span`, `embedding_red` | 1 | ~287k | embedding readouts, see below |

Three ways to make indels affordable, best first: lower `--indel_pll_passes`
(cost is exactly linear in it); use an embedding arm; or `--skip_huge_assays
10000`, which drops the two `CAPSD_AAV2S` assays (87% of indel compute) at the
cost of leaderboard comparability.

**Calibration (ESM-C 300M, 2 mid-size DMS-indel assays, Spearman vs experimental
fitness).** Every variant in both assays is a single-residue deletion, so
sequence length is constant and cannot confound the comparison:

| arm | S22A1 (490 variants) | PTEN (314 variants) | forwards/variant | seconds (S22A1) |
|---|---|---|---|---|
| strided PLL k=32 | 0.343 | 0.756 | 32 | 202 |
| strided PLL **k=16** (default) | 0.331 | 0.755 | 16 | 96 |
| strided PLL k=8 | 0.271 | 0.729 | 8 | 48 |
| strided PLL k=4 | 0.222 | 0.692 | 4 | 25 |
| `embedding_red` | **0.471** | 0.710 | 1 | 7 |
| `embedding_span` | 0.194 | 0.569 | 1 | 12 |
| *no-model control: deletion position* | *0.083* | *0.540* | *0* | *0* |

k=16 is within 0.012 of k=32 on both assays for half the compute, so it is the
default; k=8 costs up to 0.07 and is too low. `embedding_red` is competitive
with 32-pass PLL at ~28x less compute — better on one assay, slightly worse on
the other — and clears the position-only control on both, so its signal is not
just "where the deletion is". This **contradicts** the DNA-domain numbers the
readout was ported with (~0.53 AUROC there), and its direction is the opposite
of the obvious assumption: residue diversity *rises* with fitness, so
`variant_embedding_scores` negates the raw delta to keep every arm on one
"higher = more disrupted" convention. Two assays are a calibration, not a
benchmark — treat the embedding arms as promising, not settled.

**Embedding arms** (`--indel_score_mode embedding_span|embedding_red`,
`variant_embedding_scores.py`) need one forward per sequence instead of 32.
`embedding_span` pools per-residue embeddings over the derived edit span only —
whole-sequence pooling dilutes a few edited residues across hundreds of
unchanged ones — and `embedding_red` uses residue-diversity delta. They are also
the arms to reach for on **multi-mutation** variants, where masked-marginal sums
independent per-position log-ratios and is blind to epistasis. Accuracy on
proteins is **not yet benchmarked**; the numbers behind these readouts come from
DNA tasks and do not transfer.

**Measured (ESM-C 300M / `Synthyra/ESMplusplus_small`, substitutions, all assays,
no capping):** DMS Spearman **0.407** (hierarchical; flat 0.433), AUC 0.727 ·
clinical AUC **0.869** (per-gene mean, 2,525 genes, 0 variants skipped).

The suite's own `proteingym_*_zeroshot` tasks use embedding cosine instead, which
costs one forward per mutant (2.47M on DMS substitutions) and scores ~0.68
clinical AUC against masked-marginal's ~0.87. It is **default-off**; set
`PLM_BENCH_PGYM_COSINE=1` to restore — it is the
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

