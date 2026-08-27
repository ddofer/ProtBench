# PTM benchmark components

ProtBench owns reusable PTM inputs, metrics, prediction persistence, external
asset audits, and PPI feature protocols:

- `ptm_inputs.py`: PTM-Mamba annotation/vocabulary audit. It refuses to treat
  the annotation CSV as an evaluation split.
- `ptm_benchmark.py`: structured sequence/site/type records, adapter contract,
  site AUPRC/AUROC/MCC/F1, type Top-1/Top-3/MRR, coverage reporting, and
  deterministic JSONL or JSONL-gzip persistence.
- `ptm_site_assets.py`: pinned ProteinBERT phosphosite and TransPTM NHAC loaders,
  split-overlap audits, and explicit comparability-versus-deduplication policy.
- `ptm_frozen_probes.py`: deterministic sampled residue probes, centered-window
  probes, and a residue-identity baseline for external PTM site benchmarks.
- `ptm_ppi.py`: a labeled reproduction of the binder-canceling supplementary
  PPI features and the binder-aware frozen-probe features used by the primary
  protocol.

Model/checkpoint loading and Proteva corpus provenance remain in Proteva. No
encoder weights are updated by these components.
