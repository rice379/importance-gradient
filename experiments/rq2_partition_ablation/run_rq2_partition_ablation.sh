#!/usr/bin/env bash
set -euo pipefail

# One-step launcher for RQ2: partition-method ablation.
# Edit these variables only if your paths or hardware differ.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=false

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29529}

MODEL_NAME=${MODEL_NAME:-facebook/opt-1.3b}
DATASET_PATH=${DATASET_PATH:-alpaca_data}
DATASET_FORMAT=${DATASET_FORMAT:-alpaca}
THRESHOLD_OUTPUT_DIR=${THRESHOLD_OUTPUT_DIR:-threshold_compare_outputs}
OUTPUT_DIR=${OUTPUT_DIR:-rq2_partition_outputs}

MAX_LENGTH=${MAX_LENGTH:-256}
TRAIN_BS=${TRAIN_BS:-2}
EVAL_BS=${EVAL_BS:-2}
TOTAL_STEPS=${TOTAL_STEPS:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
LEARNING_RATE=${LEARNING_RATE:-2e-5}

# Baseline sparse-gradient ratio and ImportanceGradient online policy.
TOPK_RATIO=${TOPK_RATIO:-0.001}
R_LOW=${R_LOW:-4}
DELAYED_KEEP_RATIO=${DELAYED_KEEP_RATIO:-0.25}

# RQ2 compares different component partitioning methods.
METHODS=(
  knee
  otsu
  gmm_2
  kmeans_2
  percentile_30
  fixed_ratio_20
)

# In the earlier threshold/freezing script, candidates are usually the components
# selected for freezing. For ImportanceGradient, this normally corresponds to
# low-importance components. Change to "important" only if your JSON uses
# candidates as important components.
CANDIDATE_KEY_ROLE=${CANDIDATE_KEY_ROLE:-low}

mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/rq2_partition_ablation.py" \
  --model_name "${MODEL_NAME}" \
  --dataset_path "${DATASET_PATH}" \
  --dataset_format "${DATASET_FORMAT}" \
  --threshold_output_dir "${THRESHOLD_OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --methods "${METHODS[@]}" \
  --candidate_key_role "${CANDIDATE_KEY_ROLE}" \
  --max_length "${MAX_LENGTH}" \
  --train_batch_size "${TRAIN_BS}" \
  --eval_batch_size "${EVAL_BS}" \
  --total_steps "${TOTAL_STEPS}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --learning_rate "${LEARNING_RATE}" \
  --topk_ratio "${TOPK_RATIO}" \
  --release_period "${R_LOW}" \
  --delayed_keep_ratio "${DELAYED_KEEP_RATIO}" \
  --include_topk_sync \
  --seeds 42 43 44
