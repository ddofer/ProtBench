# Supplementary Notes for the ProtBench Application Note

## Generated Assets

Refresh generated tables from the repository root:

```bash
python3 scripts/paper_assets.py --out-dir paper/generated
```

Generated files:

- `paper/generated/task_coverage.md` - task and preset counts from the live
  registry.
- `paper/generated/task_inventory.tsv` - one row per registered task.
- `paper/generated/result_source_manifest.md` - result sources separated into
  safe candidates and excluded/private candidates.
- `paper/generated/representative_results.md` - optional public result slices
  from sibling project folders when those local files exist.

Generated figure files:

- `paper/figures/protbench_graphical_abstract.svg` - workflow/graphical abstract.
- `paper/figures/task_provenance_landscape.svg` - task coverage by provenance tier.

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
superfamily-transfer stress test. It should be discussed as a benchmark design
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

## Citation Checks Before Submission

The main text should cite original dataset papers rather than Hugging Face
dataset cards. The current high-priority checks are:

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

## Draft Compression Rules

Use these before submitting to a four-page venue:

- Keep the main table compact; move full task grids to generated supplement.
- Avoid broad claims such as "comprehensive" or "state of the art" unless the
  sentence states the exact scope.
- Keep implementation details that do not affect interpretation in the
  supplement.
- Retain the limitations paragraph. It prevents the resource claim from sounding
  larger than the evidence supports.
