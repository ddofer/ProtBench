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

**Humanized report (start here).** `report.py` is the one-command, DS-friendly
view — run it any time to refresh two artifacts in `results/`:

```bash
python plm/bench/report.py            # writes results/BENCH_REPORT.md + bench_pivot.csv
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

`target_modules` resolved per model family in `_hf_finetune_common.py` to **all body Linears** — Proteva `wq/wk/wv/wo/attn_gate/w12/w3`, AMPLIFY `q/k/v/wo/w12/w3`; MLM decoder + aux heads stay frozen, task head via `modules_to_save`. NOT `all-linear`.

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

## Applied fixes (2026-06-16/17)

| Bug | Fix | Commit |
|---|---|---|
| Sequence probe used `OneVsRestClassifier(liblinear)` — OvR wrapper means pseudo-multinomial, not true multinomial; `liblinear` stalls at 100 iters on 100k+ sample tasks | Switched to `LogisticRegression(solver="saga")` — true multinomial, handles large n | `c70b743` |
| Residue probe (`token_classification_probe.py`) used `lbfgs` which stalls past 1000 iters on ~600k SS3 residues | Switched to `solver="saga"` | `ef9a1c6` |
| ProteinGym clinical pathogenicity AUC inverted (~0.10 instead of ~0.90) | Negate MLM scores before `roc_auc_score` — pathogenic = deleterious = lower logP | `c151261` |
| MLM JSONL not collected by `collect_bench_results.py` glob | Use subdir pattern `FT_OUT/mlm_zs_{ckpt}/` so `write_jsonl_record(.parent)` lands in `FT_OUT` | `7dc61f1` |
| `resolve_mlm_head` only knew AMPLIFY — Proteva models raised ValueError | Added Proteva branch using `model(…).logits` / `encoder.blocks` / `encoder.decoder` | `6844ae0` |
| Proteva MLM crashed with RoPE size mismatch at max_length=2048 | Clamp `max_length` to `encoder.config.max_position` (=1024) before scoring | `78ba78f` |
| `remote_homology` LoRA collapsed (F1≈0.0003) — 1195-class head can't descend from ln(1195) in 1 epoch with patience=1 | Many-class carve-out: `lr=1e-4, patience=3` for `remote_homology` and `ec_classification` | `23dd06e` |

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
