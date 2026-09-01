# ProtBench project memory

## Role

ProtBench is the shared, reusable frozen-embedding and probe framework. Proteva
owns model-specific launchers, corpus audits, and private paper interpretation;
avoid duplicating those implementations here.

## Paper-evaluation integration

- Prediction artifacts and optional per-example outputs are common framework
  features so saved predictions can reproduce aggregate and homology-stratified
  metrics without new embedding runs.
- `PLM_BENCH_STRICT=1` makes a failed paper-screen command fail its campaign
  stage; ordinary interactive benchmark use remains non-strict.
- `scripts/profile_models.py` is the reusable common-device BF16 profiler. It
  reports checkpoint fingerprint, parameters, file size, loading, latency,
  throughput, and peak VRAM.
- The active Proteva campaign and its protocol/handoff are documented in
  `../proteva/results/paper_eval_20260901/README.md`. Raw artifacts belong outside
  Git; only compact reproduced tables are committed in Proteva.

## Commands

```bash
cd /home/ddofer/ProtBench
pytest -m 'not slow'
ruff check <changed-files>
python scripts/profile_models.py --help
```

## Interpretation guardrails

- A benchmark stage is complete only when saved predictions reproduce its metric.
- Keep paired comparisons on identical examples and bootstrap proteins rather
  than residues where applicable.
- Do not present existing CATH-EAT results as clean absolute evidence. Do not
  turn Proteva's Arm-A or combined Arm-B contrasts into causal claims.
