#!/bin/bash
# LoRA fine-tune sweep — next-run launcher. One tmux window per task so the HF
# Trainer's tqdm bar is visible live (the old /tmp scripts piped to log files,
# which kills the bar). Each window also tees to a log.
#
#   FlashAttention-2 (proteva): PROTEVA_FT_FLASH=fa2-varlen routes the padded
#   batch through model.py's varlen path (mask -> cu_seqlens, unpad, FA2, scatter
#   — validated parity + finite backward). It forces bf16 attention, so it is set
#   ONLY for classification tasks; regression keeps --fp32 + dense SDPA (the fp32
#   forward is needed to rank fine fitness differences).
#
#   batch 64 + lr 3e-4 are the argparse defaults now (sqrt-scaled from the old
#   batch-8 / lr-1e-4); max 10 epochs + early-stop patience 1 are passed below.
#
# Usage:  bash bench/run_lora_sweep.sh <model_tag> [gpu]
#   model_tag in {vanilla,step0,epoch1,epoch3}   (vanilla has no seq-cls head)
set -u
TAG=${1:?model tag: vanilla|step0|epoch1|epoch3}
GPU=${2:-0}
PY=/data/proteva/plm/.venv/bin/python
BENCH=/data/proteva/plm/bench
export PYTHONPATH=/data/proteva/plm:/data/proteva
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT=/data/proteva/plm/results/bench/lora_sweep/$TAG
LOG=/tmp/lora_sweep_logs/$TAG; mkdir -p "$OUT" "$LOG"

declare -A MODEL=(
  [vanilla]="chandar-lab/AMPLIFY_120M"
  [step0]="/data/proteva/cache/ckpts/arch_warminit_step0"
  [epoch1]="/data/proteva/cache/ckpts/stage2_3ep_epoch1_step139754"
  [epoch3]="/data/proteva/cache/ckpts/hf_stage2_final"
)
MDL=${MODEL[$TAG]:?unknown tag}

# Regression -> fp32 + flash OFF (ranking needs the fp32 forward). r16/a32.
# Everything else -> bf16 classification + flash fa2-varlen. r64/a64.
REGR="beta_lactamase_peer fluorescence enzyme_catalytic_efficiency variant_effect chezod_disorder"
SEQ_TASKS="remote_homology solubility peptide_hla ec_classification subcellular_loc signalp_binary metal_ion_binding profet_np_sp_cleaved beta_lactamase_peer fluorescence enzyme_catalytic_efficiency variant_effect"

SESSION="lora_$TAG"
tmux new-session -d -s "$SESSION" 2>/dev/null || true
for t in $SEQ_TASKS; do
  if [[ " $REGR " == *" $t "* ]]; then PREC="--fp32"; FLASH="off"; R=16; A=32
  else PREC=""; FLASH="fa2-varlen"; R=64; A=64; fi
  CMD="PROTEVA_FT_FLASH=$FLASH CUDA_VISIBLE_DEVICES=$GPU $PY $BENCH/finetune_sequence.py \
    --model_name '$MDL' --task $t --mode lora \
    --num_train_epochs 10 --learning_rate 3e-4 $PREC \
    --lora_r $R --lora_alpha $A --early_stop --early_stop_patience 1 \
    --notes 'lora_sweep_$TAG' --output_dir '$OUT' 2>&1 | tee '$LOG/$t.log'"
  tmux new-window -t "$SESSION" -n "$t" "bash -lc \"$CMD; echo DONE $t; sleep 5\""
done
echo "launched sweep in tmux session '$SESSION' (attach: tmux attach -t $SESSION)"
echo "tasks: $SEQ_TASKS"
