#!/bin/bash
# tmux dashboard for benchmark / indel jobs.
#   start:  bash plm/bench/bench_monitor.sh
#   watch:  tmux attach -t benchmon      (detach: Ctrl-b then d)
#   stop:   tmux kill-session -t benchmon
# 3 panes: GPU usage | live status (output + LoRA counts + running indel jobs) |
#          progress log tail (indel-assay + within-assay % from the new prints).
set -e
S=benchmon
HERE="$(cd "$(dirname "$0")" && pwd)"
tmux kill-session -t "$S" 2>/dev/null || true
tmux new-session -d -s "$S"

# pane 0: GPU
tmux send-keys -t "$S" "watch -n5 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader" C-m
# pane 1: status loop
tmux split-window -v -t "$S"
tmux send-keys -t "$S" "bash $HERE/_mon_status.sh" C-m
# pane 2: progress log tail (catches the new 'indel assay i/N' + 'seqs (%)' lines)
tmux split-window -h -t "$S"
tmux send-keys -t "$S" "tail -F /tmp/indel_strided_logs/*.log /tmp/bench_phase2_logs/*.log /tmp/indel_*/*.log 2>/dev/null | grep --line-buffered -E 'indel assay|seqs \(|DONE|START|Terminate|Error|Trace'" C-m

tmux select-layout -t "$S" tiled
echo "benchmon ready -> tmux attach -t benchmon  (detach Ctrl-b d)"
