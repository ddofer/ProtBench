# contrib/proteva

Scripts that only make sense for the Proteva / ProtJEPA / ProtSent line of work.
They are kept because they still run and someone may want them, not because
they are part of the benchmark.

**If you are here to benchmark a protein model, you want the
[top-level README](../../README.md) instead.** Nothing in this directory is
needed for that.

| File | What it does | Why it is not top level |
|---|---|---|
| `proteingym_aux_zeroshot.py` | ProteinGym zero-shot scoring using a Proteva checkpoint's auxiliary prediction heads | Needs `plm.hf`, which is not in this repo |
| `zero_shot_dms.py` | Shared DMS scoring helpers | Only imported by the above |
| `compare_to_vanilla.py` | Pivots results against a vanilla-AMPLIFY baseline | Hardcodes that specific baseline substring |
| `report.py` | Markdown report over a fixed checkpoint trajectory | Hardcodes `vanilla → step0 → epoch1 → epoch3` |
| `wt_tta_smoke.py` | Smoke test for test-time training | Expects a particular local checkpoint layout |

The test-time-training engine itself (`wt_test_time_training.py`, the `--tta`
flag) is general and stayed at the top level.

These carry hardcoded paths and baseline names from the machine they were
written on. Read before running rather than expecting defaults to work.
