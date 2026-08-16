# Supplementary Notes for the ProtBench Application Note

Generated tables live under `paper/generated/`; refresh them with
`python3 scripts/paper_assets.py --out-dir paper/generated`.

## Result Inclusion Rules

Main-text result tables may use:

- Common vanilla models, especially ESM-2/ESM-C-style off-the-shelf baselines.
- Published ProtSent and ProteinJEPA results when the corresponding manuscript
  or result artifact is intended to be public.
- ProtEva rows only when they are clearly vanilla/common-model baselines.

Excluded unless explicitly approved:

- ProtEva continued-pretraining checkpoints.
- ProtEva model-soup or Pareto-search outputs.
- Private or unreleased checkpoint trajectories.
- Dense ProteinJEPA trained-model grids if they distract from the resource-paper
  message; move them to supplement or omit them.

## CATH Midnight-Zone Status

The `cath_eat` task is included in the live task registry as a CATH v4.3
superfamily transfer stress test. It should be discussed as a benchmark design
example unless validated result artifacts are present.

Current checked status:

- `uv run python cath_levels.py --selfcheck` passes.
- `uv run python train_cath_tucker_head.py --selfcheck` passes.
- No committed `results/cath_eat/cath_levels.json`, `CATH_LEVELS.md`, or
  CATH benchmark CSV is present in this repository snapshot.

Manuscript rule: do not report new CATH performance numbers until a generated
artifact is committed or otherwise supplied. The main paper may state that
ProtBench includes the task and that the scorer/Tucker-head reproduction scripts
are self-checked.

## Dataset Provenance Checks

The main text cites original dataset papers rather than Hugging Face dataset
cards where sources are known. Remaining high-priority checks are:

- `AI4Protein/EC` and `AI4Protein/GO_MF`: trace the original source and label
  definitions before making a formal dataset claim.
- `biomap-research/*` rehosts: trace original sources for antibiotic resistance,
  temperature stability, enzyme catalytic efficiency, variant effect, remote
  homology, material production, metal ion binding, optimal pH, peptide-HLA,
  stability, and cloning.
- `proteinea/solubility`: verify the DeepSol mapping and split semantics.
- `biomap-research/fitness_prediction`: verify the GB1 mapping and source paper.
- `SaProtHub/DATASET-CAPE-RhlA-seqlabel`: verify the CAPE/RhlA source.
- Any model in the main result table should have a model paper citation or a
  clearly marked unpublished/internal status.
Full per-task provenance notes remain in `docs/DATASETS.md`.
