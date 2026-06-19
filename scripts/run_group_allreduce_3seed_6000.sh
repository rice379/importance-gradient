#!/usr/bin/env bash
set -euo pipefail

# Baseline group-allreduce run for three seeds.
# Run this from the repository root.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

NPROC=${NPROC:-2}
MAX_STEPS=${MAX_STEPS:-6000}
MODEL=${MODEL:-facebook/opt-1.3b}
DATASET=${DATASET:-uiyunkim-hub/pubmed-abstract}
SCORE_CSV=${SCORE_CSV:-component_scores.csv}

mkdir -p logs_real_bucket_6000

for SEED in 42 43 44; do
  OUTDIR="runs_real_bucket/final_group_allreduce_seed${SEED}_${MAX_STEPS}"
  LOG="logs_real_bucket_6000/final_group_allreduce_seed${SEED}_${MAX_STEPS}.log"

  echo "[RUN] group_allreduce seed=${SEED}"
  echo "[OUT] ${OUTDIR}"

  torchrun --standalone --nproc_per_node="${NPROC}" examples_train_importance_gradient_ddp.py \
    --model_name_or_path "${MODEL}" \
    --dataset_name "${DATASET}" \
    --score_csv "${SCORE_CSV}" \
    --sync_mode periodic \
    --low_importance_period 4 \
    --use_residual_accumulation \
    --comm_backend group_allreduce \
    --bucket_cost_mode dense \
    --payload_mode dense \
    --max_steps "${MAX_STEPS}" \
    --eval_interval 500 \
    --log_interval 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_checkpointing \
    --dtype bf16 \
    --seed "${SEED}" \
    --output_dir "${OUTDIR}" 2>&1 | tee "${LOG}"
done
