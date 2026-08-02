# ProtBench

Benchmark protein models — language models, alignment search, or a k-mer count —
on 43 tasks, scored identically and written to one CSV.

The point is comparability. A protein language model, a LoRA fine-tune, an
MMseqs2 search and a bag of amino-acid triplets all go through the same splits,
the same probes and the same metrics, so the numbers can be put in one table
without caveats about who evaluated what.

```bash
git clone https://github.com/ddofer/ProtBench && cd ProtBench
uv sync

python protein_benchmark_suite.py -m facebook/esm2_t6_8M_UR50D \
    --tasks solubility -p linear --eval_split test
```

That downloads the model and dataset, embeds the sequences, fits a linear probe,
and writes `bench_facebook_esm2_t6_8M_UR50D.csv`. On a GPU it takes a minute.

## How it works

Most tasks use a **frozen probe**: run the model once to get one vector per
protein, then fit a small classifier on those vectors. The model is never
updated. This measures what the pretrained representation already knows, it is
cheap, and it is hard to get wrong — which is exactly what you want from a
benchmark.

```
sequences ──▶ model (frozen) ──▶ one vector per protein ──▶ probe ──▶ metrics
```

Three probes are available via `-p`:

| Probe | What it is | Use it when |
|---|---|---|
| `linear` | Logistic/ridge regression on standardised embeddings | Default. Measures linearly accessible information. |
| `knn` | k-nearest neighbours, raw Euclidean | Retrieval and homology-transfer tasks. `--knn_k 1` is 1-NN annotation transfer. |
| `histgb` | Gradient-boosted trees | Non-linear structure. Slow on high dimensions. |

Fine-tuning (full, last-N layers, or LoRA) lives in `finetune_sequence.py` and
`finetune_residue.py`. It costs far more compute and is a separate question from
"what does the representation contain".

## Recipes

**See what tasks exist.**

```bash
python protein_benchmark_suite.py --list_tasks
```

Prints all 43 with their type, metric, and which preset includes them.

**Run everything except ProteinGym.** ProteinGym is large and slow, so it is
opt-in. `--no-fast` gives you the other 32 tasks:

```bash
python protein_benchmark_suite.py -m facebook/esm2_t33_650M_UR50D \
    --no-fast --eval_split test -p linear --output_dir results/esm2_650m
```

Add `--proteingym` when you do want it.

**Run the fast subset on two models.** The suite takes one model at a time; loop
in the shell. Point both at the same `--output_dir` so the results land together:

```bash
for MODEL in facebook/esm2_t6_8M_UR50D facebook/esm2_t33_650M_UR50D; do
    python protein_benchmark_suite.py -m "$MODEL" \
        --fast --eval_split test -p linear --output_dir results/compare
done
```

`--fast` is the default (18 tasks, capped at 100k samples each). `--very-fast`
is a curated 8-task subset for a quick signal.

**Compare models across benchmarks.** Each model writes its own
`bench_<model>.csv`. One pivot gives the table:

```python
import pandas as pd, glob

df = pd.concat(map(pd.read_csv, glob.glob("results/compare/bench_*.csv")))
print(df.pivot_table(index="Task", columns="Model", values="AUC").round(3))
```

Swap `values=` for the metric you care about — `Spearman` for regression tasks,
`F1_Macro` for multilabel, `Accuracy` for multiclass. Every metric the run
computed is a column.

**Get a floor to compare against.** Before believing a model is good at a task,
check what counting amino acid triplets scores:

```bash
python protein_benchmark_suite.py -m kmer --tasks solubility -p linear --device cpu
```

`kmer` is a real model name here — no learning, just 3-mer frequencies. `kmer4`
and `kmer2` also work. If your 650M-parameter model barely beats it, that says
something about the task, not the model.

**Put error bars on it.**

```bash
python protein_benchmark_suite.py -m <model> --tasks stability --bootstrap 1000
```

Adds `<Metric>_CI_low` / `_CI_high` columns by resampling the test predictions.
No refitting, so it costs seconds.

## Gotchas

These are the ones that produce a wrong number rather than an error.

**`--fast` is on by default.** A bare run does 18 tasks, not all 43. Pass
`--no-fast` for the full set.

**Embedding caching is off by default, on purpose.** The cache key for a
HuggingFace model id is just the name, so it does *not* invalidate when the
upstream weights change — you can get yesterday's number for today's model.
Turn it on with `--cache_embeddings` when you are re-benchmarking a fixed model
and want the speedup. Local checkpoint paths are keyed on file size and mtime
and do invalidate correctly.

**`--eval_split test` is not the default.** The default is `validation`. Use
`test` for numbers you intend to report, `validation` while iterating.

**Bootstrap CIs skip AUC and AP,** which need probability outputs and are
computed after the resampled block. Do not combine `--bootstrap` with the
cross-validation path (`--eval_split validation` on a task with no validation
split): fold aggregation averages every numeric column, which would turn the
intervals into a mean of intervals.

**Sequences are truncated at 1024 residues** (`--max_length`). Fine for most
tasks; check it if yours has long proteins.

**Multiclass AUC may be skipped** with a warning when the test split does not
contain every training class. Accuracy, F1 and MCC are still valid.

## Reading the output

One row per task × seed × probe × split. Columns beyond the identifiers are
metrics, and which appear depends on the task type:

| Task type | Metrics |
|---|---|
| binary | Accuracy, F1, F1_Macro, BalancedAccuracy, MCC, AUC, AP |
| multiclass | Accuracy, F1_Weighted, F1_Macro, BalancedAccuracy, MCC, AUC |
| multilabel | Accuracy, F1_Macro, F1_Micro |
| regression | Spearman, Pearson, MSE, MAE, R2 |
| retrieval | Recall@1, Recall@10, Recall@30 |

Prefer **MCC** and **BalancedAccuracy** on imbalanced tasks. Several tasks here
are 90/10 or worse, where always predicting the majority class scores 0.90
accuracy — MCC scores that 0.0, which is the honest answer.

Each task has a `main_metric` (shown by `--list_tasks`) that is the conventional
one to report for it.

## Baselines beyond neural models

A representation claim needs a floor to clear. Three are built in:

| Baseline | Command | Needs |
|---|---|---|
| k-mer frequencies | `-m kmer` | nothing |
| MMseqs2 alignment | `python mmseqs_baseline.py --task <task>` | `mmseqs` on PATH or `$MMSEQS_BIN` |
| phmmer profiles | `python hmmer_baseline.py --task <task>` | `uv sync --extra alignment` |

The alignment baselines call the same `prepare_data` the model side uses, so
they see byte-identical inputs — the comparison is not confounded by
preprocessing.

## The CATH midnight-zone task

`cath_eat` is worth calling out because it is the one task where sequence
alignment is designed to fail. Queries are filtered so no alignment-detectable
relative exists in the lookup set, so it asks whether embeddings recognise a
shared fold that alignment cannot see.

```bash
python protein_benchmark_suite.py -m <model> --tasks cath_eat -p knn --knn_k 1 \
    --eval_split test
```

Use `-p knn --knn_k 1`. That makes the probe literally the reference method:
take the CATH label of the nearest lookup protein by Euclidean distance. A
linear probe would instead fit 6,500 classes over 69,000 rows, which is a
different experiment and not what the published numbers describe.

Accuracy at the homologous-superfamily level:

| Method | Accuracy | Source |
|---|---|---|
| k-mer 3-mers | 0.0 | measured here |
| ESM2-8M | 21.3 | measured here |
| ESM2-650M | 42.7 | measured here |
| MMseqs2 | 35 | Heinzinger 2022 |
| raw ProtT5 | 64 | Heinzinger 2022 |
| ProtTucker(ProtT5) | 76 | Heinzinger 2022 |
| HMMER profiles | 77 | Heinzinger 2022 |

The k-mer 0.0 is the check that matters: amino acid composition carries nothing
about remote homology, so the task is not leaking a shortcut. Our numbers are
not expected to reproduce the published ones exactly — different models and
embedding pipelines — but the splits and scoring are identical, so models
evaluated here are comparable to each other.

Dataset: [`GrimSqueaker/cath43-eat`](https://huggingface.co/datasets/GrimSqueaker/cath43-eat).
Rebuild it with `scripts/build_cath_eat_dataset.py`.

## Install

```bash
uv sync                      # probes, the common case
uv sync --extra finetune     # + peft, for LoRA
uv sync --extra alignment    # + pyhmmer, for the phmmer baseline
uv sync --extra plots        # + matplotlib, for report figures
```

Python ≥ 3.10. MMseqs2 is a system binary and cannot be installed from here —
`conda install -c bioconda mmseqs2`, or a static build from
[the MMseqs2 releases](https://github.com/soedinglab/MMseqs2/releases).

## Tests

```bash
pytest -m "not slow"    # 157 tests, no network
pytest                  # adds tests that download models
```

## Reproducibility

`--seed` seeds Python, NumPy and PyTorch, plus every sklearn probe and dataset
shuffle. `--seed_list 42,43,44` runs multiple seeds in one process, one CSV row
each, reusing the embeddings.

`torch.use_deterministic_algorithms` is deliberately not set: it breaks or badly
slows several embedding kernels, and embedding is inference-only, so it buys
nothing here.

The suite logs its own file path and task count at startup. Older copies of this
code exist with a different task registry, so check that line before trusting a
results directory you did not just produce.

## Data sources and citation

Tasks pull public datasets from HuggingFace; per-task sources and citations are
in [`docs/DATASETS.md`](docs/DATASETS.md). Cite the dataset authors for any task
you report, not just this repo.

## More

- [`docs/DATASETS.md`](docs/DATASETS.md) — every task's source and citation
- [`docs/ADVANCED.md`](docs/ADVANCED.md) — fine-tuning recipes, ProteinGym
  zero-shot, test-time training, checkpoint pitfalls
