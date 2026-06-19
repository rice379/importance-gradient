#!/usr/bin/env bash
set -euo pipefail

# One-step RQ5 ADTopk end-to-end rerun with log-window averaged metrics.
# Default: seed 42 only, methods: ADTopk Sync / IAD-ADTopk / IG-ADTopk.
# After confirming metrics, set SEEDS="42 43 44" to run the full 3-seed version.

ROOT_DIR=${ROOT_DIR:-$(pwd)}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT=${RUN_ROOT:-${ROOT_DIR}/runs/rq5_adtopk_windowavg_seed42}
NPROC=${NPROC:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES

MODEL_NAME=${MODEL_NAME:-facebook/opt-1.3b}
DATASET_NAME=${DATASET_NAME:-uiyunkim-hub/pubmed-abstract}
DATASET_CONFIG_NAME=${DATASET_CONFIG_NAME:-}
DATASET_SPLIT=${DATASET_SPLIT:-train}
LOCAL_DATASET_PATH=${LOCAL_DATASET_PATH:-}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}

MAX_LENGTH=${MAX_LENGTH:-256}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-100000}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-2000}
PROFILE_MAX_TRAIN_SAMPLES=${PROFILE_MAX_TRAIN_SAMPLES:-20000}

TRAIN_BSZ=${TRAIN_BSZ:-1}
EVAL_BSZ=${EVAL_BSZ:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_STEPS=${MAX_STEPS:-3000}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
LOG_INTERVAL=${LOG_INTERVAL:-20}
LR=${LR:-1e-6}
WARMUP_STEPS=${WARMUP_STEPS:-100}
DTYPE=${DTYPE:-bf16}

PROFILE_WARMUP_STEPS=${PROFILE_WARMUP_STEPS:-200}
PROFILE_STEPS=${PROFILE_STEPS:-400}
SCORE_ALPHA=${SCORE_ALPHA:-0.7}
THRESHOLD_METHOD=${THRESHOLD_METHOD:-knee}

ADTOPK_RATIO=${ADTOPK_RATIO:-0.01}
LOW_IMPORTANCE_PERIOD=${LOW_IMPORTANCE_PERIOD:-4}
DELAYED_KEEP_RATIO=${DELAYED_KEEP_RATIO:-0.25}
EFFECTIVE_LOW_COST_RATIO=${EFFECTIVE_LOW_COST_RATIO:-0.25}
BUCKET_NUM=${BUCKET_NUM:-4}
# For IG, if packing overhead is still high, first try: 1048576 / 2097152 / 4194304.
BUCKET_BLOCK_SIZE_NUMEL=${BUCKET_BLOCK_SIZE_NUMEL:-262144}

SEEDS_STR=${SEEDS:-42}
read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

mkdir -p "${RUN_ROOT}"

cd "${ROOT_DIR}"

COMMON_DATA_ARGS=(
  --model_name_or_path "${MODEL_NAME}"
  --dataset_name "${DATASET_NAME}"
  --dataset_split "${DATASET_SPLIT}"
  --max_length "${MAX_LENGTH}"
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --profile_max_train_samples "${PROFILE_MAX_TRAIN_SAMPLES}"
  --max_eval_samples "${MAX_EVAL_SAMPLES}"
  --per_device_train_batch_size "${TRAIN_BSZ}"
  --per_device_eval_batch_size "${EVAL_BSZ}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate "${LR}"
  --warmup_steps "${WARMUP_STEPS}"
  --dtype "${DTYPE}"
  --adtopk_ratio "${ADTOPK_RATIO}"
  --low_importance_period "${LOW_IMPORTANCE_PERIOD}"
)

if [[ -n "${DATASET_CONFIG_NAME}" ]]; then
  COMMON_DATA_ARGS+=(--dataset_config_name "${DATASET_CONFIG_NAME}")
fi
if [[ -n "${LOCAL_DATASET_PATH}" ]]; then
  COMMON_DATA_ARGS+=(--local_dataset_path "${LOCAL_DATASET_PATH}")
fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  COMMON_DATA_ARGS+=(--local_files_only)
fi

for SEED in "${SEEDS_ARR[@]}"; do
  echo "==================== Seed ${SEED}: ADTopk profile ===================="
  PROFILE_DIR="${RUN_ROOT}/seed${SEED}/profile"
  SCORE_CSV="${PROFILE_DIR}/adtopk_component_scores.csv"

  python "${SCRIPT_DIR}/rq5_adtopk_train_windowavg.py" \
    --run_mode profile \
    "${COMMON_DATA_ARGS[@]}" \
    --profile_warmup_steps "${PROFILE_WARMUP_STEPS}" \
    --profile_steps "${PROFILE_STEPS}" \
    --score_alpha "${SCORE_ALPHA}" \
    --threshold_method "${THRESHOLD_METHOD}" \
    --seed "${SEED}" \
    --output_dir "${PROFILE_DIR}"

  echo "==================== Seed ${SEED}: ADTopk Sync ===================="
  torchrun --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/rq5_adtopk_train_windowavg.py" \
    --run_mode train \
    --method adtopk_sync \
    "${COMMON_DATA_ARGS[@]}" \
    --score_csv "${SCORE_CSV}" \
    --max_steps "${MAX_STEPS}" \
    --eval_interval "${EVAL_INTERVAL}" \
    --log_interval "${LOG_INTERVAL}" \
    --seed "${SEED}" \
    --output_dir "${RUN_ROOT}/seed${SEED}/adtopk_sync"

  echo "==================== Seed ${SEED}: Importance-Aware Delay on ADTopk ===================="
  torchrun --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/rq5_adtopk_train_windowavg.py" \
    --run_mode train \
    --method iad_adtopk \
    "${COMMON_DATA_ARGS[@]}" \
    --score_csv "${SCORE_CSV}" \
    --max_steps "${MAX_STEPS}" \
    --eval_interval "${EVAL_INTERVAL}" \
    --log_interval "${LOG_INTERVAL}" \
    --seed "${SEED}" \
    --output_dir "${RUN_ROOT}/seed${SEED}/iad_adtopk"

  echo "==================== Seed ${SEED}: ImportanceGradient on ADTopk ===================="
  torchrun --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/rq5_adtopk_train_windowavg.py" \
    --run_mode train \
    --method ig_adtopk \
    "${COMMON_DATA_ARGS[@]}" \
    --score_csv "${SCORE_CSV}" \
    --max_steps "${MAX_STEPS}" \
    --eval_interval "${EVAL_INTERVAL}" \
    --log_interval "${LOG_INTERVAL}" \
    --seed "${SEED}" \
    --bucket_num "${BUCKET_NUM}" \
    --bucket_block_size_numel "${BUCKET_BLOCK_SIZE_NUMEL}" \
    --bucket_scheduler_mode risk_aware \
    --bucket_use_adaptive_switch \
    --effective_low_cost_ratio "${EFFECTIVE_LOW_COST_RATIO}" \
    --effective_payload_low_keep_ratio "${DELAYED_KEEP_RATIO}" \
    --effective_payload_rotation_interval "${LOW_IMPORTANCE_PERIOD}" \
    --output_dir "${RUN_ROOT}/seed${SEED}/ig_adtopk"
done

python "${SCRIPT_DIR}/summarize_rq5_adtopk_windowavg.py" \
  --run_root "${RUN_ROOT}" \
  --methods adtopk_sync iad_adtopk ig_adtopk \
  --seeds ${SEEDS_STR} \
  --baseline_method adtopk_sync \
  --burnin_steps "${WARMUP_STEPS}"

echo "[DONE] RQ5 ADTopk window-average rerun finished. Results in: ${RUN_ROOT}"
