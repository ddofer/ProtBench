# ProtBench Paper Draft

This directory holds the working Application Note draft and reproducible paper
assets for ProtBench.

## Files

- `protbench_application_note.md` - short main-text draft, written for a
  Bioinformatics Application Note style submission.
- `supplement.md` - supplementary methods, result-source boundaries, and open
  citation checks.
- `references.bib` - working bibliography for the draft.
- `generated/` - tables generated from the live task registry and local public
  result artifacts.
- `figures/` - deterministic SVG figure outputs generated from code.

## Refresh Generated Tables

From the repository root:

```bash
python3 scripts/paper_assets.py --out-dir paper/generated
```

The generator imports only `benchmark_tasks.py` and uses the Python standard
library. If the sibling result folders are not present, it still writes the task
coverage and result-source manifest, and marks representative result tables as
not generated.

## Refresh Figures

```bash
python3 scripts/generate_paper_figures.py --out-dir paper/figures
```

The figure generator uses only the Python standard library and emits SVG vector
figures. The graphics are intentionally schematic: they document the evaluation
contract and task/provenance landscape rather than adding decorative artwork.

## Submission Notes

Before submission, fill in author metadata, affiliations, contact email,
repository release/DOI, and any journal-specific formatting. The main text is
kept deliberately short; extended task inventory and private-result exclusion
notes belong in the supplement while the main text stays compact.
