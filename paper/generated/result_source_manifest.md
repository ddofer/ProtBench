# Paper Result Source Manifest

This manifest separates result artifacts that may feed the ProtBench paper from artifacts that require explicit approval before use.

## Safe candidates

| Source | Path pattern | Use |
| --- | --- | --- |
| ProtBench in-repo baselines | results/benchmarks/hmmer_baseline.json; results/benchmarks/mmseqs_baseline.json | Alignment baseline summaries already tracked in this repository. |
| ProtJepa vanilla ESM-2 | ../protJepa/results/bench_pb/vanilla*/bench_facebook_esm2_*.csv | Use only off-the-shelf ESM-2 rows in the main paper unless trained-model rows are explicitly approved for the supplement. |
| ProtSent public benchmark outputs | ../ProtSent/results/benchmarks/v3; ../ProtSent/results/benchmarks/v2_150m; ../ProtSent/results/benchmarks/COMPARISON.md | Published/public ProtSent and vanilla comparison artifacts. |
| ProtEva vanilla baseline candidates | ../proteva/results/soup_pareto_20260816/vanilla_esmc/bench_Synthyra_ESMplusplus_small.csv | Use only clearly vanilla/common-model rows; exclude private continued-pretraining checkpoints. |

## Excluded unless explicitly approved

| Source | Path pattern | Reason |
| --- | --- | --- |
| ProtEva continued-pretraining and soup outputs | ../proteva/results/benchmarks; ../proteva/results/soup_pareto_*/ | Private in-progress results; do not publish without explicit approval. |
| ProtJepa trained-model grids | ../protJepa/results/bench_pb/*_jepa*; ../protJepa/results/bench_pb/*_mlm*; ../protJepa/results/bench_pb/*_ladder* | Supplement-only candidates; main text should not depend on them unless the relevant manuscript is public-ready. |
