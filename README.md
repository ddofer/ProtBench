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

Four probes are available via `-p`:

| Probe | What it is | Use it when |
| --- | --- | --- |
| `linear` | Logistic/ridge regression (sklearn lbfgs, CPU) on standardised embeddings | Default. Measures linearly accessible information. |
| `torch_linear` | The same linear head, but `nn.Linear` trained by AdamW ([skorch](https://skorch.readthedocs.io)), lr 1e-3, 5 % inner val split, early stopping (patience 2, last weights kept; 10x lr and patience 5 for multilabel / 100+ classes) | Same question as `linear`, different solver. Faster on big inputs: residue-level tasks (10^5-10^6 rows) and multilabel GO/EC (one multi-output head instead of one sklearn fit per label). Results carry `Probe=torch_linear`, so the two stay separable in the CSV. |
| `knn` | k-nearest neighbours, raw Euclidean | Retrieval and homology-transfer tasks. `--knn_k 1` is 1-NN annotation transfer. |
| `histgb` | Gradient-boosted trees | Non-linear structure. Slow on high dimensions. |

Every row also records `ProbeFitSec` — wall-clock seconds of the head fit
alone (embedding time excluded), so probes can be compared on speed as well as
score. `torch_linear` and `linear` share every other stage (same embeddings,
same `StandardScaler`, same metrics); `make_probe_model` in
`protein_benchmark_suite.py` is the single registry both come from, and the
head itself is `torch_linear_head.py` (~100 lines, sklearn API).

### `linear` vs `torch_linear` — measured (ESM2-8M, `--very-fast`, test split, one B300, shared node)

| Task | metric | `linear` | `torch_linear` | fit s `linear` | fit s `torch_linear` |
| --- | --- | --- | --- | --- | --- |
| remote_homology (1195 classes) | Acc | 0.562 | 0.561 | 61.3 | 2.2 |
| solubility | Acc | 0.621 | 0.626 | 1.2 | 7.2 |
| metal_ion_binding | Acc | 0.683 | 0.686 | 0.1 | 1.0 |
| fluorescence | Spearman | 0.572 | 0.550 | 0.1 | 10.4 |
| stability | Spearman | 0.690 | 0.696 | 0.3 | 10.6 |
| beta_lactamase_peer (4-fold CV) | Spearman | 0.638 | 0.577 | 0.0 | 2.2 |
| ss3 (residue, ~400k rows) | Acc | 0.746 | 0.744 | 11.5 | 72.4 |
| ec_classification (multilabel, 572 labels) | F1_Micro | 0.655 | 0.641 | 156.3 | 21.7 |

Reading: scores match within ~0.01 on six of eight tasks, and trail on the two
small regression sets (fluorescence -0.02, beta-lactamase -0.06: ridge's closed
form beats an early-stopped AdamW head there). Speed: `torch_linear` wins only
where the sklearn solver scales badly — many classes (remote_homology 28x) and
multilabel OvR (EC 7x). For binary / low-class tasks and hidden sizes up to
~1280 the per-step overhead of a minibatch loop makes it *slower* than
lbfgs/ridge, GPU or not. Defaults were picked from lr x patience sweeps on
these cached embeddings (lr 1e-3 beat 1e-2 on 4/5 sequence tasks; sparse
multilabel needs 1e-2 and patience 5; a 1000-step floor so few-hundred-sample
tasks are not under-trained) — `TorchLinearHead(...)` in `torch_linear_head.py`
exposes `lr`, `batch_size`, `patience`, `max_epochs` if you want to retune.

For comparison, the HF-Trainer frozen-backbone path (`finetune_sequence.py
--mode probe`, solubility, 3 epochs) took 122 s wall (86 s training: the
backbone forward is re-run every epoch) for Acc 0.678 / AUC 0.773 on the
test split, vs embed-once + probe ≈ 20 s total. Embed once.

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

## Benchmarking a whole model: `scripts/run_bench.py`

One command per model or checkpoint. It picks the probe that is fastest for
each task, skips tasks this model already has results for, and writes a
readable summary:

```bash
python scripts/run_bench.py -m Synthyra/ESMplusplus_small              # --preset fast (17 tasks)
python scripts/run_bench.py -m /path/to/checkpoint --preset no-fast    # everything
python scripts/run_bench.py -m MODEL --tasks solubility ss3            # explicit tasks
python scripts/run_bench.py -m MODEL --proteingym                      # probes + ProteinGym zero-shot
python scripts/run_bench.py -m MODEL --finetune lora                   # probes + LoRA fine-tuning
```

Outputs, all under `--output_dir` (default `results/benchmarks/`):

| File | What it is |
| --- | --- |
| `bench_<model>.csv` | the usual per-task result rows |
| `SUMMARY_<model>.md` | one markdown row per task: main metric, value, probe, eval strategy, fit seconds |
| `mlm_zeroshot_<model>.jsonl` | ProteinGym zero-shot records, with `--proteingym` |
| `results/bench_results_all.csv` | long-format table across every model (via `collect_bench_results.py`) |

**Resumable.** A task counts as done when the CSV holds a *successful* row for
it with the same probe, eval split and sample count — so an interrupted sweep
continues where it stopped, a failed task is retried next run, and a capped
`--very-fast` scout row never makes a full sweep look complete. `--force`
re-runs everything.

**Probe per task.** The rule lives in the suite (`-p auto`, which this script
passes), so the plain CLI gets it too: `linear` everywhere except where sklearn
scales badly — multilabel (OvR fits one model per label), 1000+ class tasks
(`remote_homology`, `cath_eat`) and residue tasks (full-data `ss3` is ~2.8M rows;
615 s single-core in lbfgs against 8 s on the torch head) get `torch_linear`.
The `Probe` column records which one ran, so rows stay comparable across models.

**ProteinGym** (`--proteingym`) runs the masked-marginal scorer in
`proteingym_mlm_zeroshot.py` — substitutions by default, `--proteingym_indels`
for the rest. One masked forward per *mutated position* serves every variant at
that position, so all 2.47M DMS substitutions cost ~86k forwards. The suite's
own cosine zero-shot tasks are default-off (`PLM_BENCH_PGYM_COSINE=1` restores
them): one forward per mutant for a much weaker score. See
[docs/ADVANCED.md](docs/ADVANCED.md).

**Fine-tuning** is opt-in (`--finetune lora|last_n|full`) and runs
`finetune_sequence.py` with early stopping on the sequence-level tasks only;
those results land in `finetune_sequence_<model>.jsonl` and are folded into
`bench_results_all.csv`. Frozen probes stay the default: they answer "what is
in the representation", cost one forward pass, and do not need a
hyperparameter search to be fair.

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
| `--fast` *(default)* | 17 | Includes `scope40_retrieval` (family); `--no-fast` adds `scope40_retrieval_superfamily` and `_fold`. |
| `--no-fast` | 32 | The broad set — but **not** a superset of `--fast`. Drops `scope40_retrieval`. |
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

## Telling results apart

Every result row records **`CodeVersion`** (git short SHA, `-dirty` if the tree
had uncommitted changes) alongside `Date`, `Probe`, `EvalSplit`, `EvalStrategy`
and `Samples`. It is part of the dedup identity, so a re-run under different code
appends a row instead of overwriting the old one, and
`results/bench_results_all.csv` carries it as `code_version`.

That matters because some changes move numbers legitimately:

| Change | What it affects | How to spot the old rows |
| --- | --- | --- |
| Residue CV fallback moved from residue-level `KFold` to protein-level `GroupKFold` | `ss3`/`ss8`/`conservation`/`disprot` rows scored with `EvalStrategy=validation_cv4_train` — the old ones leaked residues of a protein across folds and read **high** | no `CodeVersion`, or a SHA older than `943cff6` |
| ProteinGym zero-shot moved from embedding cosine to masked-marginal | different files entirely: cosine rows are in `bench_<model>.csv` with `EvalMode=proteingym_zeroshot`; masked-marginal lives in `mlm_zeroshot_<model>.jsonl` with `mode=mlm_zeroshot` | file + `EvalMode` |
| Indel `RED` arm sign corrected, `--indel_pll_passes` default 32 → 16 | ProteinGym indel scores | the JSONL record's `code_version`, plus `metric.indel_score_mode` / `metric.indel_pll_passes` |
| `-p auto` routing | which probe ran | the `Probe` column already records the resolved probe, never `auto` |

Rows written before the stamp existed read `unknown`. **They are not
interchangeable with stamped rows** — treat a comparison that mixes them as
suspect unless the affected paths above are irrelevant to it.

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

## Reading the output

One row per task × seed × probe × split. Columns beyond the identifiers are
metrics, and which appear depends on the task type:

| Task type | Metrics |
| --- | --- |
| binary | Accuracy, F1, F1_Macro, BalancedAccuracy, MCC, AUC, AP |
| multiclass | Accuracy, F1_Weighted, F1_Macro, BalancedAccuracy, MCC, AUC |
| multilabel | Accuracy, F1_Macro, F1_Micro |
| regression | Spearman, Pearson, MSE, MAE, R2 |
| retrieval | Recall@1, Recall@10, Recall@30, MAP, plus `eligible_*` over queries with >=1 same-label gallery item (`n_eligible_queries`) |
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
first and then pass `--models tag=model_or_path`. To persist per-query predictions
and report the Proteva corpus-identity strata, also pass the pinned audit table:

```bash
uv run python cath_levels.py \
    --models v4=/private/path/to/checkpoint \
    --identity-table /private/path/to/cath_eat_query_identity.tsv \
    --out /private/path/to/cath_results

# Recompute exact/non-exact and >=90%/<90% tables without loading the model or embeddings.
uv run python cath_levels.py --rescore-only \
    --identity-table /private/path/to/cath_eat_query_identity.tsv \
    --out /private/path/to/cath_results
```

This writes one 219-row JSONL per model plus full and stratified JSON/Markdown.
The H-level denominators are full 150, exact/non-exact 102/48, and >=90%/<90%
145/5; the five-query result is explicitly diagnostic. Existing checkpoints are
corpus-contaminated, so stratification does not make the absolute score clean.

`train_cath_tucker_head.py`
is an optional ProtTucker-style projection-head reproduction on frozen CATH
embeddings; its selfcheck validates the hard-negative mining and loss semantics,
but real training/evaluation still needs the model embeddings and GPU runtime.

## Big files on a small root filesystem

`embed_cache/` grows to tens of GB per model (residue tasks dominate: ss3 is
~2.8M residues x hidden). On a machine with a small root filesystem, point it at
bulk storage — either per run or once, with a symlink:

```bash
python protein_benchmark_suite.py -m MODEL --embed_cache_dir /bulk/protbench_cache
ln -s /bulk/protbench_cache embed_cache    # or make the default path a symlink
```

The locally-built datasets under `data/` are portable the same way: they are
plain `datasets.save_to_disk` dumps, so a directory copied or symlinked from
another machine works as-is (see [docs/DATASETS.md](docs/DATASETS.md)).

## Install

```bash
uv sync                      # probes and tests, the common case
uv sync --extra finetune     # + peft, for LoRA
uv sync --extra alignment    # + pyhmmer, for the phmmer baseline
uv sync --extra plots        # + matplotlib, for report figures
uv sync --extra kernels      # + kernels, matching whatever transformers asks for
```

**`uv sync` prunes.** Those lines are alternatives, not steps: `uv sync --extra
kernels` *uninstalls* whatever `--extra finetune` put there. Pass every extra you
want in one command — `uv sync --extra finetune --extra alignment`.

`pytest` comes from the `dev` dependency group, which `uv sync` installs by
default, so it survives all of the above.

Python ≥ 3.10. MMseqs2 is a system binary and cannot be installed from here —
`conda install -c bioconda mmseqs2`, or a static build from
[the MMseqs2 releases](https://github.com/soedinglab/MMseqs2/releases).

## Tests

```bash
pytest -m "not slow"    # 254 tests, no network
pytest                  # adds tests that download models
```

Three further tests cover the fine-tuning `TrainingArguments` builder and need
`accelerate`, which arrives with `uv sync --extra finetune`. Without it they fail
on import rather than skip, so a bare sync reports 254 passed and 3 failed — not
a regression.

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
- `decontaminate.py` — filter a corpus against benchmark test sets
- `contact_catjac.py` — zero-shot contact prediction from an MLM head
