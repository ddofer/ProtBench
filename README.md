# ProtBench

Benchmark protein models — language models, alignment search, or a k-mer count —
on 60 tasks, scored identically and written to one CSV.

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

```text
sequences ──▶ model (frozen) ──▶ one vector per protein ──▶ probe ──▶ metrics
```

Three probes are available via `-p`:

| Probe | What it is | Use it when |
| --- | --- | --- |
| `linear` | Logistic/ridge regression on standardised embeddings | Default. Measures linearly accessible information. |
| `knn` | k-nearest neighbours, raw Euclidean | Retrieval and homology-transfer tasks. `--knn_k 1` is 1-NN annotation transfer. |
| `histgb` | Gradient-boosted trees | Non-linear structure. Slow on high dimensions. |

Fine-tuning (full, last-N layers, or LoRA) lives in `finetune_sequence.py` and
`finetune_residue.py`. It costs far more compute and is a separate question from
"what does the representation contain".

## Getting the data

Most tasks download themselves. The dataset is pulled from HuggingFace on first
use and cached under `~/.cache/huggingface/`, so the first run of a task is
slower and needs network access. No account or token is required — every
dataset used here is public.

**Four tasks are built locally instead of downloaded**, and two of them are in
the default preset, so run these before your first sweep:

```bash
python scripts/prep_conservation.py   # conservation_flip  (in --very-fast and --fast)
python scripts/prep_disprot.py        # disprot            (in --fast)
python scripts/prep_flip2.py          # flip2_amylase, flip2_rhomax (in --no-fast)
```

Skip them and those tasks fail with a message naming the script to run; the
rest of the sweep continues and the failure is recorded in the results CSV.

Per-task sources, sizes and citations: [docs/DATASETS.md](docs/DATASETS.md).

## Paper draft

A working Application Note draft and generated paper assets live under
[`paper/`](paper/). Refresh registry-derived tables with:

```bash
python3 scripts/paper_assets.py --out-dir paper/generated
```

The generator imports only `benchmark_tasks.py`, so it can run in a minimal
Python environment without installing the full benchmark stack.

## Recipes

**See what tasks exist.**

```bash
python protein_benchmark_suite.py --list_tasks
```

Prints all 60 with their type, metric, and which preset includes them.

**Run the broad set.** `--no-fast` gives you 37 tasks:

```bash
python protein_benchmark_suite.py -m facebook/esm2_t33_650M_UR50D \
    --no-fast --eval_split test -p linear --output_dir results/esm2_650m
```

The presets are not nested, which surprises people:

| Preset | Tasks | Notes |
| --- | --- | --- |
| `--very-fast` | 8 | Curated scout subset. |
| `--fast` *(default)* | 18 | Includes `scope40_retrieval`. |
| `--no-fast` | 37 | The broad set — but **not** a superset of `--fast`. Drops `scope40_retrieval`. |
| `--proteingym` | +8 | Large and slow, so opt-in. |

`cafa5`, `go_mf`, `go_bp` and `go_cc` are in **no** preset — they have thousands
of labels and would dominate a sweep. Request them by name with `--tasks`. So is
`scope40_retrieval` if you want it alongside `--no-fast`.

The ten held-out secondary-structure sets (`ss3_casp12` … `ss8_ts115`) are also
opt-in. They train on the same data as `ss3`/`ss8` and differ only in which
standard test set they score against, so they answer "how does this compare to
the published CASP/CB513 numbers", not "is this model good".

**They require `--eval_split test`** — they have no validation split, so the
default `validation` cross-validates on the shared training file instead, and
all five `ss8_*` tasks then report the same number. The run warns when this
happens, and the CSV records it as `EvalStrategy=validation_cv4_train`.

```bash
python protein_benchmark_suite.py -m <model> --tasks ss8_cb513 --eval_split test
```

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

It writes the comparison CSV to `--output_dir` (default `results/benchmarks/`).

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

**`--fast` is on by default.** A bare run does 18 tasks, not all 60. See the
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

**Three temperature tasks mean three different things.** `deepet_topt` is the
optimal growth temperature of the *source organism*; `thermostability` and
`meltome` are melting temperatures of the *protein*. They are not replicates and
should not be averaged together.

**`ppi_affinity` has a 200-pair test split.** That is small enough that a single
run moves around a lot — use `--bootstrap` before reading anything into a gap.

## Reading the output

One row per task × seed × probe × split. Columns beyond the identifiers are
metrics, and which appear depends on the task type:

| Task type | Metrics |
| --- | --- |
| binary | Accuracy, F1, F1_Macro, BalancedAccuracy, MCC, AUC, AP |
| multiclass | Accuracy, F1_Weighted, F1_Macro, BalancedAccuracy, MCC, AUC |
| multilabel | Accuracy, F1_Macro, F1_Micro |
| regression | Spearman, Pearson, MSE, MAE, R2 |
| retrieval | Recall@1, Recall@10, Recall@30 |
| token_classification | Accuracy, MCC, F1 (per residue, not per protein) |
| contact_prediction | P@L, P@L/2, P@L/5 × short / medium / long range |

The less obvious ones:

- **AUC** is the area under the ROC curve. 0.5 is chance, 1.0 is perfect. It is
  threshold-free, so it does not depend on where you cut the probability.
- **AP** is average precision — area under the precision–recall curve. More
  informative than AUC when positives are rare.
- **MCC** is the Matthews correlation coefficient, a correlation between
  prediction and truth on a −1..1 scale. 0 is chance.
- **BalancedAccuracy** is the mean per-class recall.
- **P@L/k** is contact-prediction precision: rank every residue pair, take the
  top `L/k` (where `L` is the sequence length), and report the fraction that are
  real contacts. `_short` / `_medium` / `_long` restrict to pairs `6-11`,
  `12-23` and `24+` apart in sequence. Long-range is the hard one and the
  conventional headline. Chance is the contact rate itself — **0.028** on this
  test split, so read `P@L/5_long` against that, not against 0.5.

Residues with no ground truth are excluded from residue-level scoring rather
than treated as a class of their own — `ss8`'s source marks 11.5% of its test
residues "unassigned", almost all at chain termini, and scoring those would
inflate accuracy with a position-predictable pseudo-class. The run log names the
count it dropped.

Prefer **MCC** and **BalancedAccuracy** on imbalanced tasks. Several tasks here
are 90/10 or worse, where always predicting the majority class scores 0.90
accuracy — MCC scores that 0.0, which is the honest answer.

Each task has a `main_metric` (shown by `--list_tasks`) that is the conventional
one to report for it.

## Baselines beyond neural models

A representation claim needs a floor to clear. Three are built in:

| Baseline | Command | Needs |
| --- | --- | --- |
| k-mer frequencies | `-m kmer` | nothing |
| MMseqs2 alignment | `python mmseqs_baseline.py --task <task>` | `mmseqs` on PATH or `$MMSEQS_BIN` |
| phmmer profiles | `python hmmer_baseline.py --task <task>` | `uv sync --extra alignment` |

The alignment baselines call the same `prepare_data` the model side uses, so
they see byte-identical inputs — the comparison is not confounded by
preprocessing.

## Checking your corpus for test-set leakage

Structural benchmarks here — `cath_eat`, `scope40_retrieval`, `remote_homology`,
`contact_probe` — are drawn from the PDB. Any pretraining corpus built from
UniRef or the PDB overlaps them, so a model can score well by recall rather than
generalisation. `decontaminate.py` searches each task's test split against your
corpus and reports what to drop:

```bash
python decontaminate.py --corpus pretrain.fasta \
    --tasks cath_eat remote_homology scope40_retrieval \
    --min-seq-id 0.3 --coverage 0.8 -o stoplist.txt

python decontaminate.py --corpus pretrain.fasta --tasks cath_eat \
    --write-filtered clean.fasta
```

It pulls the test sequences through the same `prepare_data` the benchmark uses,
so you filter against exactly what you will later be scored on. Needs `mmseqs`
(same requirement as the MMseqs2 baseline).

The defaults (30% identity over 80% coverage) are the usual redundancy-reduction
line. **Lower them for fold-level tasks** — remote homologues leak fold
information at well under 30% identity, which is the entire premise of
`cath_eat`. Coverage uses the weaker of query and target, so a short test protein
contained inside a long corpus protein does not drag in every sequence sharing a
common domain.

## Contact prediction

Contact prediction asks whether a representation encodes tertiary structure: for
every pair of residues, are they within 8 Å in the folded protein? Labels come
from CB coordinates in the TAPE ProteinNet set, but the model only ever sees the
primary sequence — no MSA, no `.a3m` file, no structure input.

There are two paths, and they answer different questions:

| | `contact_probe` | `contact_catjac` |
| --- | --- | --- |
| Method | Supervised pairwise linear probe on frozen per-residue embeddings | Categorical Jacobian, zero-shot |
| Runs on | Any model the registry loads | Models with a reachable MLM head |
| Asks | Is contact information linearly readable from the embeddings? | Has the model learned the coupling directly? |

```bash
# probe: a task in the suite, one row in bench_<model>.csv like anything else
python protein_benchmark_suite.py -m <model> --tasks contact_probe --eval_split test

# categorical Jacobian: separate script, because it needs the MLM head
python contact_catjac.py -m <model>
```

The Jacobian mutates each position to all 20 amino acids and measures how the
model's predictions shift everywhere else — a model that learned a coupling
between two residues changes its mind at one when the other changes. That costs
`L × 20` forward passes per protein, so `--max_len` (default 512) skips longer
ones and **logs which**, and `--max_proteins` caps the count. Both paths write
the same metric columns, so the numbers sit side by side.

Measured here on all 40 test proteins with ESM2-650M: the Jacobian reaches
`P@L/5_long` = 0.454, the frozen probe 0.135, against a 0.028 chance floor. The
Jacobian figure matches the ~0.45 published for ESM2-650M attention contact
heads, without a trained head — a linear probe on pooled pair features cannot
express coupling the way the Jacobian measures it directly. Read the per-protein
spread too (median 0.488, range 0.00–0.90): free-modeling CASP targets score near
zero while template-based ones exceed 0.8, so a single mean hides most of it.

`contact_probe` trains on `--contact_train_proteins` proteins (default 400) out
of the 25k available; the full split does not fit in memory once each protein
expands into O(L²) pairs.

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

For the full C/A/T/H breakdown, run `uv run python cath_levels.py --selfcheck`
first and then pass `--models tag=model_or_path`. `train_cath_tucker_head.py`
is an optional ProtTucker-style projection-head reproduction on frozen CATH
embeddings; its selfcheck validates the hard-negative mining and loss semantics,
but real training/evaluation still needs the model embeddings and GPU runtime.

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
