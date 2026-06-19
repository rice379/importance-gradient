#!/usr/bin/env bash
set -euo pipefail

# One-step script for RQ4: balanced payload materialization / packing ablation.
# It runs four materialization policies and summarizes their bucket skew and runtime.
#
# Usage examples:
#   bash run_rq4_packing_ablation.sh
#   GPUS=2 SCORE_CSV=/path/to/profile.csv bash run_rq4_packing_ablation.sh
#
# If SCORE_CSV is not set and rq1_profile_boundary.py exists, this script first
# generates a default profile and uses it for the RQ4 runs.

GPUS=${GPUS:-2}
MODEL=${MODEL:-facebook/opt-1.3b}
DATASET=${DATASET:-uiyunkim-hub/pubmed-abstract}
LOCAL_DATASET_PATH=${LOCAL_DATASET_PATH:-}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-800}
PROFILE_MAX_STEPS=${PROFILE_MAX_STEPS:-1000}
PROFILE_WINDOWS=${PROFILE_WINDOWS:-200:1000}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-100000}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-2000}
MAX_LENGTH=${MAX_LENGTH:-256}
LOG_INTERVAL=${LOG_INTERVAL:-20}
EVAL_INTERVAL=${EVAL_INTERVAL:-0}

LOW_IMPORTANCE_PERIOD=${LOW_IMPORTANCE_PERIOD:-4}
PAYLOAD_KEEP_RATIO=${PAYLOAD_KEEP_RATIO:-0.25}
PAYLOAD_ROTATION_INTERVAL=${PAYLOAD_ROTATION_INTERVAL:-4}
BUCKET_NUM=${BUCKET_NUM:-4}
BUCKET_BLOCK_SIZE_NUMEL=${BUCKET_BLOCK_SIZE_NUMEL:-262144}
WARMUP_STEPS=${WARMUP_STEPS:-100}

OUT_ROOT=${OUT_ROOT:-runs/rq4_packing_ablation}
POLICIES=${POLICIES:-"direct_filtering sequential greedy risk_aware"}

mkdir -p "${OUT_ROOT}"

# -----------------------------------------------------------------------------
# 1) Resolve or generate the importance profile used by the gate.
# -----------------------------------------------------------------------------
if [[ -z "${SCORE_CSV:-}" ]]; then
  DEFAULT_PROFILE="${OUT_ROOT}/profile_seed${SEED}/profiles/profile_rq4_profile_w${PROFILE_WINDOWS/:/_}.csv"
  if [[ ! -f "${DEFAULT_PROFILE}" ]]; then
    echo "[RQ4] SCORE_CSV not provided. Generating a profile first..."
    PROFILE_ARGS=(
      --model_name_or_path "${MODEL}"
      --dataset_name "${DATASET}"
      --local_files_only "${LOCAL_FILES_ONLY}"
      --max_length "${MAX_LENGTH}"
      --max_train_samples "${MAX_TRAIN_SAMPLES}"
      --max_eval_samples "${MAX_EVAL_SAMPLES}"
      --max_steps "${PROFILE_MAX_STEPS}"
      --seed "${SEED}"
      --profile_windows "${PROFILE_WINDOWS}"
      --experiment_id "rq4_profile"
      --output_dir "${OUT_ROOT}/profile_seed${SEED}"
    )
    if [[ -n "${LOCAL_DATASET_PATH}" ]]; then
      PROFILE_ARGS+=(--local_dataset_path "${LOCAL_DATASET_PATH}")
    fi
    torchrun --standalone --nproc_per_node="${GPUS}" rq1_profile_boundary.py "${PROFILE_ARGS[@]}"
  fi
  SCORE_CSV="${DEFAULT_PROFILE}"
fi

echo "[RQ4] Using SCORE_CSV=${SCORE_CSV}"

# -----------------------------------------------------------------------------
# 2) Run packing policies.
# -----------------------------------------------------------------------------
for POLICY in ${POLICIES}; do
  RUN_DIR="${OUT_ROOT}/${POLICY}"
  mkdir -p "${RUN_DIR}"
  echo "[RQ4] Running policy=${POLICY}, output=${RUN_DIR}"

  TRAIN_ARGS=(
    --model_name_or_path "${MODEL}"
    --dataset_name "${DATASET}"
    --local_files_only "${LOCAL_FILES_ONLY}"
    --max_length "${MAX_LENGTH}"
    --max_train_samples "${MAX_TRAIN_SAMPLES}"
    --max_eval_samples "${MAX_EVAL_SAMPLES}"
    --max_steps "${MAX_STEPS}"
    --eval_interval "${EVAL_INTERVAL}"
    --log_interval "${LOG_INTERVAL}"
    --seed "${SEED}"
    --score_csv "${SCORE_CSV}"
    --sync_mode periodic
    --low_importance_period "${LOW_IMPORTANCE_PERIOD}"
    --use_residual_accumulation
    --materialization_policy "${POLICY}"
    --bucket_num "${BUCKET_NUM}"
    --bucket_block_size_numel "${BUCKET_BLOCK_SIZE_NUMEL}"
    --payload_low_keep_ratio "${PAYLOAD_KEEP_RATIO}"
    --payload_rotation_interval "${PAYLOAD_ROTATION_INTERVAL}"
    --risk_use_adaptive_switch
    --output_dir "${RUN_DIR}"
  )
  if [[ -n "${LOCAL_DATASET_PATH}" ]]; then
    TRAIN_ARGS+=(--local_dataset_path "${LOCAL_DATASET_PATH}")
  fi

  torchrun --standalone --nproc_per_node="${GPUS}" rq4_train_packing_ablation.py "${TRAIN_ARGS[@]}"
done

# -----------------------------------------------------------------------------
# 3) Summarize.
# -----------------------------------------------------------------------------
python rq4_summarize_packing.py \
  --input_root "${OUT_ROOT}" \
  --output_dir "${OUT_ROOT}/analysis" \
  --warmup_steps "${WARMUP_STEPS}"

echo "[done] RQ4 outputs: ${OUT_ROOT}"
