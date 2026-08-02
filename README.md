# ProtBench

Benchmark protein models — language models, alignment search, or a k-mer count —
on 43 tasks, scored identically and written to one CSV.

The point is comparability. A protein language model, a LoRA fine-tune, an
MMseqs2 search and a bag of amino-acid triplets all go through the same splits, the
same probes and the same metrics, so differences between them are differences
between the methods and not between two people's evaluation scripts.

```bash
git clone https://github.com/ddofer/ProtBench && cd ProtBench
uv sync

python protein_benchmark_suite.py -m facebook/esm2_t6_8M_UR50D \
    --tasks solubility -p linear --eval_split test
```

That downloads the model and dataset, embeds the sequences, fits a linear probe,
and writes `bench_facebook_esm2_t6_8M_UR50D.csv`. On a GPU it takes a minute.

## How it works

Most tasks use a **frozen probe**: run the model once to turn each protein into
a single vector — its *embedding* — then fit a small classifier on those
vectors. The model's own weights are never updated.

This measures what the pretrained representation already contains, rather than
what the model could learn if you trained it further. It is also cheap: you
embed once and can then fit as many probes as you like.

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

**Run the broad set.** `--no-fast` gives you 32 tasks:

```bash
python protein_benchmark_suite.py -m facebook/esm2_t33_650M_UR50D \
    --no-fast --eval_split test -p linear --output_dir results/esm2_650m
```

The presets are not nested, which surprises people:

| Preset | Tasks | Notes |
|---|---|---|
| `--very-fast` | 8 | Curated scout subset. |
| `--fast` *(default)* | 18 | Includes `scope40_retrieval`. |
| `--no-fast` | 32 | The broad set — but **not** a superset of `--fast`. Drops `scope40_retrieval`. |
| `--proteingym` | +8 | Large and slow, so opt-in. |

`cafa5` and `go_mf` are in **no** preset — they have thousands of labels and
would dominate a sweep. Request them by name with `--tasks`. So is
`scope40_retrieval` if you want it alongside `--no-fast`.

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

**Compare two models.** Use the built-in comparison — it picks each task's
main metric for you and prints the winner and the gap:

```bash
python protein_benchmark_suite.py --compare \
    --compare_model1 results/compare/bench_facebook_esm2_t6_8M_UR50D.csv \
    --compare_model2 results/compare/bench_kmer.csv
```

```text
Task                 Winner Metric  Best_AUC  Other_AUC   Δ_AUC Samples
Solubility (DeepSol)   kmer    AUC    0.6123    0.56096 0.05134     500
```

It writes to `results/benchmarks/` relative to your current directory, which
`--output_dir` does not change.

For more than two models, read the CSVs yourself — but **filter to one split
first**, or you will average test and validation rows together:

```python
import pandas as pd, glob

df = pd.concat(map(pd.read_csv, glob.glob("results/compare/bench_*.csv")))
df = df[df.EvalSplit == "test"]          # do not skip this line
print(df.pivot_table(index="Task", columns="Model", values="AUC").round(3))
```

`pivot_table` averages by default, so without the filter a model benchmarked on
both splits silently reports the mean of the two. Swap `values=` for the metric
you want — `Spearman` for regression, `F1_Macro` for multilabel.

**Get a floor to compare against.** Before believing a model is good at a task,
check what counting amino acid triplets scores:

```bash
python protein_benchmark_suite.py -m kmer --tasks solubility -p linear \
    --eval_split test --device cpu
```

`kmer` is a real model name here — no learning, just 3-mer frequencies. `kmer4`
and `kmer2` also work. If a large model barely beats it, look at the task before
concluding anything about the model: some tasks are largely predictable from
amino acid composition alone.

**Put error bars on it.**

```bash
python protein_benchmark_suite.py -m <model> --tasks stability --bootstrap 1000
```

Adds `<Metric>_CI_low` / `_CI_high` columns by resampling the test predictions.
No refitting, so it costs seconds.

## Gotchas

These are the ones that produce a wrong number rather than an error.

**`--fast` is on by default.** A bare run does 18 tasks, not all 43. See the
preset table above — the presets are not nested.

**Result CSVs accumulate.** Re-running into the same `--output_dir` merges into
the existing `bench_<model>.csv`. A row is replaced only if it matches on task,
split, probe, seed, sample count *and date*; change any of those and you get an
extra row. So one file can hold both a validation and a test result for the same
task, which is how averaging them together happens. Filter on `EvalSplit`
before you aggregate.

**Reduce runtime with `--max_samples N`,** which caps rows per split. Useful for
smoke tests, but small N widens the error bars fast — check with `--bootstrap`
before drawing a conclusion from a capped run.

**The linear probe can hit its iteration limit** on high-dimensional embeddings
and log a `ConvergenceWarning`. The score is still reported. Treat a warned run
as approximate; standardising is already on, so the usual next step is more
samples rather than more iterations.

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
| token_classification | Accuracy, MCC, F1 (per residue, not per protein) |

The less obvious ones:

- **AUC** is the area under the ROC curve. 0.5 is chance, 1.0 is perfect. It is
  threshold-free, so it does not depend on where you cut the probability.
- **AP** is average precision — area under the precision–recall curve. More
  informative than AUC when positives are rare.
- **MCC** is the Matthews correlation coefficient, a correlation between
  prediction and truth on a −1..1 scale. 0 is chance.
- **BalancedAccuracy** is the mean per-class recall.

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

`cath_eat` is the one task where sequence alignment is designed to fail. Its
queries are filtered so that no relative detectable by alignment exists in the
lookup set, so it asks whether a model's embeddings recognise a shared fold that
alignment cannot see.

```bash
python protein_benchmark_suite.py -m <model> --tasks cath_eat -p knn --knn_k 1 \
    --eval_split test
```

Use `-p knn --knn_k 1`: that makes the probe the published method exactly — take
the CATH label of the nearest lookup protein by Euclidean distance. A linear
probe would instead fit 6,500 classes over 69,000 rows, which is a different
experiment.

For a sense of scale, 3-mer frequencies score 0.0 here and ESM2-650M scores
0.43. Reference numbers and full provenance are in
[docs/DATASETS.md](docs/DATASETS.md#the-cath-midnight-zone-task).

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
