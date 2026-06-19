#!/usr/bin/env bash
set -euo pipefail

# One-step script for RQ3: low-importance release-frequency sweep.
# Run from your training directory, e.g.:
#   cd ~/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
#   bash run_rq3_frequency_sweep.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=false

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29521}

# Change this to your actual training entry if the filename is different.
TRAIN_ENTRY=${TRAIN_ENTRY:-rq2_partition_ablation.py}

OUT_ROOT=${OUT_ROOT:-runs/rq3_frequency_opt13b_pubmed}

# This should point to the offline/Knee threshold outputs generated before RQ3.
THRESHOLD_DIR=${THRESHOLD_DIR:-threshold_compare_outputs}

# Use the same default workload used in your detailed ablations.
MODEL_NAME=${MODEL_NAME:-facebook/opt-1.3b}
DATASET_PATH=${DATASET_PATH:-pubmed_data}
DATASET_FORMAT=${DATASET_FORMAT:-alpaca}

# Sweep settings required by the new RQ3.
RELEASE_PERIODS=${RELEASE_PERIODS:-1,2,4,8,16}
SEEDS=${SEEDS:-42,43,44}

# Common training arguments. Keep every item fixed across the sweep.
COMMON_TRAIN_ARGS="
  --model_name ${MODEL_NAME}
  --dataset_path ${DATASET_PATH}
  --dataset_format ${DATASET_FORMAT}
  --threshold_output_dir ${THRESHOLD_DIR}
  --methods knee
  --candidate_key_role low
  --max_length 256
  --train_batch_size 1
  --eval_batch_size 1
  --total_steps 1600
  --eval_interval 100
  --learning_rate 1e-6
  --topk_ratio 1.0
  --delayed_keep_ratio 1.0
  --limit_train_samples 4096
  --limit_eval_samples 512
"

python rq3_frequency_ablation_driver.py \
  --train_entry "${TRAIN_ENTRY}" \
  --use_torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  --output_root "${OUT_ROOT}" \
  --release_periods "${RELEASE_PERIODS}" \
  --seeds "${SEEDS}" \
  --release_period_arg "--release_period" \
  --residual_on_args "--use_residual_compensation" \
  --residual_off_args "--disable_residual_compensation" \
  --train_args "${COMMON_TRAIN_ARGS}"

printf '\nRQ3 finished. Summary files:\n'
printf '  %s/rq3_frequency_summary.csv\n' "${OUT_ROOT}"
printf '  %s/rq3_frequency_summary.tex\n' "${OUT_ROOT}"
