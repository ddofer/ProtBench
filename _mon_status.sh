#!/bin/bash
# Status frame for the bench tmux monitor (bench_monitor.sh). Loops every 20s.
B=/data/proteva/plm/results/bench
while true; do
  clear; date '+%H:%M:%S'
  echo "== indel output (jsonl; nonzero = that model DONE) =="
  for d in idx_full_epoch3 idx_full_vanilla idx_full_step0 idx_full_epoch1 \
           clinidx_epoch3 clinidx_vanilla clinidx_step0 clinidx_epoch1; do
    n=$(ls "$B/$d"/mlm_zs_*/*.jsonl 2>/dev/null | wc -l); echo "  $d: $n"
  done
  echo "== LoRA test results by model =="
  /data/proteva/plm/.venv/bin/python - <<'PY' 2>/dev/null
import json,glob,collections
c=collections.Counter()
for f in glob.glob('/data/proteva/plm/results/bench/lora_clean/**/*.jsonl',recursive=True):
    for l in open(f):
        l=l.strip()
        if l and json.loads(l).get('split')=='test':
            c[json.loads(l).get('notes','?').replace('clean_phase2_','')]+=1
print({m:c.get(m,0) for m in ['vanilla','step0','epoch1','epoch3']})
PY
  echo "== running indel procs (out | k | elapsed) =="
  for p in $(pgrep -f "proteingym_mlm_zeroshot.py.*indel" 2>/dev/null); do
    cl=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    out=$(echo "$cl" | grep -oE '(idx_full|clinidx)_[a-z0-9]+|/tmp/[a-z_/]+' | head -1)
    k=$(echo "$cl" | grep -oE 'indel_pll_passes [0-9]+' | awk '{print $2}')
    e=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
    echo "  $out | k=$k | $e"
  done
  sleep 20
done
