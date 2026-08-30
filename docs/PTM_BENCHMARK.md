# PTM benchmark protocol

ProtBench provides the reusable “exam”; Proteva supplies private checkpoints and
corpus provenance. None of these components updates encoder weights.

## Inputs and metrics

- `ptm_inputs.py` audits the PTM-Mamba annotation/vocabulary table. It refuses
  to turn pretraining annotations into a fake test split.
- `ptm_benchmark.py` stores structured site/type examples, coverage, site
  AUPRC/AUROC/MCC/F1, type Top-1/Top-3/MRR, and deterministic predictions.
- `ptm_frozen_probes.py` fits deterministic residue or centered-window probes
  and a residue-identity baseline.

## External assets

- `ptm_site_assets.py` loads ProteinBERT phosphosite and TransPTM NHAC, reports
  exact author-split overlap, and keeps comparability and deduplicated views
  separate.
- `ptm_ppi_assets.py` pins and audits `RosettaCommons/PTMint`. Its provided
  cluster split is not exact-target/pair-disjoint, so it is a reproduction
  split; a target-grouped primary split must be frozen before model scoring.
- `ptm_ppi.py` preserves PTM-Mamba's binder-cancelling features only as a
  labeled reproduction. Primary PPI features retain the partner and explicit
  partner-target interaction terms.

“Deduplicated” above refers to overlap within an external asset. A separate
search against the Proteva pretraining corpus is required before calling an
external PTM result corpus-clean.
