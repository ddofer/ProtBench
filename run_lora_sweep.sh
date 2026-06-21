#!/bin/bash
# LoRA fine-tune sweep — next-run launcher. One tmux window per task so the HF
# Trainer's tqdm bar is visible live (the old /tmp scripts piped to log files,
# which kills the bar). Each window also tees to a log.
#
#   Attention: dense SDPA (the head's default). FlashAttention-2 was tested and
#   is SLOWER for these padded FT batches (the varlen path's per-layer GPU sync +
#   unpad overhead beat its kernel savings, and the FFN runs full-length anyway),
#   so it is not used. See hf/sequence_classification.py:_FINETUNE_FLASH_MODE.
#
#   batch 64 + lr 3e-4 are the argparse defaults now (sqrt-scaled from the old
#   batch-8 / lr-1e-4); max 10 epochs + early-stop patience 1 are passed below.
#   Regression tasks override to --fp32 + r16/a32; classification uses r64/a64.
#
# Usage:  bash bench/run_lora_sweep.sh <model_tag> [gpu_or_csv] [seed]
#   model_tag in {vanilla,step0,epoch1,epoch3,epoch4}   (vanilla has no seq-cls head)
set -u
TAG=${1:?model tag: vanilla|step0|epoch1|epoch3|epoch4}
GPU_ARG=${2:-0}
SEED=${3:-42}
PY=/data/proteva/plm/.venv/bin/python
BENCH=/data/proteva/plm/bench
export PYTHONPATH=/data/proteva/plm:/data/proteva
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT=/data/proteva/plm/results/bench/lora_sweep/$TAG/seed_$SEED
LOG=/tmp/lora_sweep_logs/$TAG/seed_$SEED; mkdir -p "$OUT" "$LOG"
IFS=',' read -r -a GPUS <<< "$GPU_ARG"

declare -A MODEL=(
  [vanilla]="chandar-lab/AMPLIFY_120M"
  [step0]="/data/proteva/cache/ckpts/arch_warminit_step0"
  [epoch1]="/data/proteva/cache/ckpts/stage2_3ep_epoch1_step139754"
  [epoch3]="/data/proteva/cache/ckpts/hf_stage2_final"
  [epoch4]="/data/proteva/cache/ckpts/hf_stage2_epoch4_2048_pssmhead_plddt_4gpu_b144_nollrd_lr7e5"
)
MDL=${MODEL[$TAG]:?unknown tag}

# Regression -> --fp32 (ranking needs the fp32 forward) + r16/a32.
# Everything else -> bf16 classification + r64/a64.
REGR="beta_lactamase_peer fluorescence enzyme_catalytic_efficiency variant_effect chezod_disorder"
SEQ_TASKS="remote_homology solubility peptide_hla ec_classification subcellular_loc signalp_binary metal_ion_binding profet_np_sp_cleaved beta_lactamase_peer fluorescence enzyme_catalytic_efficiency variant_effect"

SESSION="lora_${TAG}_s${SEED}"
tmux new-session -d -s "$SESSION" 2>/dev/null || true
for gpu_idx in "${!GPUS[@]}"; do
  GPU=${GPUS[$gpu_idx]}
  worker="$LOG/gpu_${GPU}.sh"
  {
    echo "set -uo pipefail"
    echo "export PYTHONPATH=/data/proteva/plm:/data/proteva"
    echo "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"
  } > "$worker"
  task_idx=0
  for t in $SEQ_TASKS; do
    if [ $((task_idx % ${#GPUS[@]})) -eq "$gpu_idx" ]; then
      if [[ " $REGR " == *" $t "* ]]; then PREC="--fp32"; R=16; A=32
      else PREC=""; R=64; A=64; fi
      cat >> "$worker" <<EOF
echo START $t GPU=$GPU seed=$SEED \$(date -u +%Y-%m-%dT%H:%M:%SZ)
CUDA_VISIBLE_DEVICES=$GPU $PY $BENCH/finetune_sequence.py \
  --model_name '$MDL' --task $t --mode lora \
  --num_train_epochs 10 --learning_rate 3e-4 $PREC \
  --lora_r $R --lora_alpha $A --early_stop --early_stop_patience 1 \
  --seed $SEED --notes 'lora_sweep_${TAG}_seed_${SEED}' --output_dir '$OUT' 2>&1 | tee '$LOG/$t.log'
echo DONE $t GPU=$GPU seed=$SEED \$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
    fi
    task_idx=$((task_idx + 1))
  done
  tmux new-window -t "$SESSION" -n "gpu${GPU}" "bash '$worker'; echo DONE gpu=$GPU; sleep 5"
done
echo "launched sweep in tmux session '$SESSION' (attach: tmux attach -t $SESSION)"
echo "gpus: $GPU_ARG; seed: $SEED"
echo "tasks: $SEQ_TASKS"
