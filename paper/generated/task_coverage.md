# ProtBench Task Coverage

Generated from `benchmark_tasks.py`.

| Problem type | Tasks | Main metric(s) |
| --- | --- | --- |
| Binary classification | 12 | AUC |
| Multiclass classification | 5 | AUC, Accuracy |
| Multilabel classification | 3 | F1_Macro, F1_Micro |
| Regression | 17 | MSE, Spearman |
| Residue-level token classification | 5 | F1_Macro, MCC, Spearman |
| Retrieval | 1 | Recall@10 |

| Preset/scope | Tasks | Notes |
| --- | --- | --- |
| All registered tasks | 43 | Full registry, including opt-in tasks. |
| Standard probe tasks | 35 | Non-ProteinGym tasks. |
| ProteinGym tasks | 8 | Four supervised and four zero-shot variant-effect tasks. |
| --very-fast | 8 | Curated scout subset. |
| --fast | 18 | Default sweep: FAST_TASKS plus retrieval tasks. |
| --no-fast/default | 32 | Broad non-ProteinGym sweep, excluding retrieval and very large multilabel tasks. |
