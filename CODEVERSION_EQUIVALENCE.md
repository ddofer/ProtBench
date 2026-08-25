# `e4dd5f1-dirty` == `0ec12b1`

Benchmark rows are stamped with `code_version` (the git SHA, plus `-dirty` when the
tree has uncommitted changes). The Stage-2 campaign produced every row under
**`e4dd5f1-dirty`**; commit `0ec12b1` is exactly that working tree, committed with no
edits in between.

**The two stamps are the same code.** Rows carrying either may be compared directly.
Any analysis filtering on `CodeVersion` must accept both:

```python
CAMPAIGN_VERSIONS = {"e4dd5f1-dirty", "0ec12b1"}
```

## What is NOT comparable

`origin/main` is ahead by `f7cfd95` (SCOPe superfamily/fold task keys, MAP and
eligible-query metrics) and `e6aa519` (--fast task count). Those change
`protein_benchmark_suite.py` by 89 lines. **Do not merge them until the campaign's
remaining milestone benches are done** (v4 3-epoch at 60k/83k/100k/110741), or those
milestones cannot be compared to the 36-task tables already published for vanilla,
v3-83k, v4-36k, Base-10k, ArmA-10k and ArmB-10k.
