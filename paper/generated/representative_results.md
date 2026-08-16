# Representative Public Result Tables

Generated from local sibling-project result artifacts. These tables intentionally exclude private ProtEva continued-pretraining and model-soup outputs.

## Vanilla ESM-2 Scale Slice

Rows are linear-probe test-split results. Empty cells mean the metric was not present in the source row.

| Task | Metric | ESM-2 35M | ESM-2 150M |
| --- | --- | --- | --- |
| Remote Homology (Fold) | Accuracy | 0.647 | 0.688 |
| Solubility (DeepSol) | AUC | 0.697 | 0.721 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.994 | 0.994 |
| Neuropeptide Precursor Prediction (ProFET/NeuroPID) | AUC | 0.975 | 0.980 |
| beta-lactamase-PEER | Spearman | 0.733 | 0.811 |
| Metal Ion Binding | AUC | 0.791 | 0.792 |
| Variant Effect (GB1) | Spearman | 0.844 | 0.843 |
| Fluorescence (TAPE) | Spearman | 0.592 | 0.579 |
| Stability (Biomap) | Spearman | 0.437 | 0.704 |
| SCOPe-40 Structural Retrieval | Recall@10 | 0.584 | 0.591 |

Sources:

- ESM-2 35M: `../protJepa/results/bench_pb/vanilla/bench_facebook_esm2_t12_35M_UR50D.csv`
- ESM-2 150M: `../protJepa/results/bench_pb/vanilla_150m/bench_facebook_esm2_t30_150M_UR50D.csv`

## Published ProtSent 35M Slice

Rows are linear-probe test-split results. Empty cells mean the metric was not present in the source row.

| Task | Metric | ESM-2 35M | ProtSent-V1 35M | ProtSent-V2 35M |
| --- | --- | --- | --- | --- |
| Remote Homology (Fold) | Accuracy | 0.687 | 0.690 | 0.702 |
| Solubility (DeepSol) | AUC | 0.696 | 0.693 | 0.698 |
| Signal Peptide Prediction (SignalP/ProteinBERT) | AUC | 0.994 | 0.995 | 0.996 |
| Metal Ion Binding | AUC | 0.790 | 0.760 | 0.747 |
| Variant Effect (GB1) | Spearman | 0.816 | 0.825 | 0.813 |
| Fluorescence (TAPE) | Spearman | 0.591 | 0.591 | 0.588 |
| Stability (Biomap) | Spearman | 0.440 | 0.511 | 0.388 |

Sources:

- ESM-2 35M: `../ProtSent/results/benchmarks/v3/esm2_35m_linear/bench__storage_models_ESM2-35M.csv`
- ProtSent-V1 35M: `../ProtSent/results/benchmarks/v3/protsent_old_linear/bench_oriel9p_protsent-esm2-35M.csv`
- ProtSent-V2 35M: `../ProtSent/results/benchmarks/v3/protsent_v3_linear/bench_models_protsent_esm2_35m_v3_final.csv`
