#!/usr/bin/env bash
set -euo pipefail

# One-command script for RQ1: cross-workload component-importance stability.
# It runs the profiler on several workloads under the same model and seed, then
# computes pairwise workload-stability metrics.
#
# Default workloads:
#   PubMed      -> domain-specific language modeling
#   WikiText103 -> general language modeling
#   Alpaca      -> instruction tuning
#
# Example:
#   bash run_rq1_workload_stability.sh
#
# Common overrides:
#   GPUS=8 MODEL=facebook/opt-1.3b LOCAL_FILES_ONLY=0 bash run_rq1_workload_stability.sh
#   OUT=runs/rq1_workload_quick MAX_STEPS=200 MAX_TRAIN_SAMPLES=2000 bash run_rq1_workload_stability.sh

GPUS=${GPUS:-4}
OUT=${OUT:-runs/rq1_workload_stability}
MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_LABEL=${MODEL_LABEL:-opt-1.3b}
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-1600}
PROFILE_WINDOWS=${PROFILE_WINDOWS:-400:1600}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-100000}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-2000}
TOPK_RATIO=${TOPK_RATIO:-0.01}
ALPHA=${ALPHA:-0.7}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-256}
DTYPE=${DTYPE:-bf16}
LR=${LR:-2e-5}
WARMUP_STEPS=${WARMUP_STEPS:-100}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}
TRACE_EVERY=${TRACE_EVERY:-1}
LOG_INTERVAL=${LOG_INTERVAL:-20}
EVAL_INTERVAL=${EVAL_INTERVAL:-0}

mkdir -p "$OUT"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "0" ]]; then
  MODEL_ARGS+=(--no_local_files_only)
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  MODEL_ARGS+=(--gradient_checkpointing)
fi

# Format per row:
#   workload_name|dataset_name|dataset_config_name|dataset_split|local_dataset_path|text_fields
# Leave empty fields blank. You can edit this list to add/remove workloads.
WORKLOAD_SPECS=(
  "pubmed|uiyunkim-hub/pubmed-abstract||train||abstract"
  "wikitext103|Salesforce/wikitext|wikitext-103-raw-v1|train||text"
  "alpaca|tatsu-lab/alpaca||train||instruction,input,output"
)

for SPEC in "${WORKLOAD_SPECS[@]}"; do
  IFS='|' read -r WORKLOAD DATASET CONFIG SPLIT LOCAL_DATASET TEXT_FIELDS <<< "$SPEC"
  PROFILE_OUT="$OUT/profile_${MODEL_LABEL}_${WORKLOAD}_seed${SEED}"

  DATA_ARGS=(--workload_name "$WORKLOAD" --dataset_name "$DATASET" --dataset_split "$SPLIT")
  if [[ -n "$CONFIG" ]]; then
    DATA_ARGS+=(--dataset_config_name "$CONFIG")
  fi
  if [[ -n "$LOCAL_DATASET" ]]; then
    DATA_ARGS+=(--local_dataset_path "$LOCAL_DATASET")
  fi
  if [[ -n "$TEXT_FIELDS" ]]; then
    DATA_ARGS+=(--text_fields "$TEXT_FIELDS")
  fi

  echo "[run] workload=$WORKLOAD dataset=$DATASET config=${CONFIG:-none} split=$SPLIT out=$PROFILE_OUT"
  torchrun --standalone --nproc_per_node="$GPUS" "$SCRIPT_DIR/rq1_profile_workload_importance.py" \
    --model_name_or_path "$MODEL" \
    --model_label "$MODEL_LABEL" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    --seed "$SEED" \
    --max_steps "$MAX_STEPS" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --dtype "$DTYPE" \
    --learning_rate "$LR" \
    --warmup_steps "$WARMUP_STEPS" \
    --weight_decay "$WEIGHT_DECAY" \
    --topk_ratio "$TOPK_RATIO" \
    --alpha "$ALPHA" \
    --profile_windows "$PROFILE_WINDOWS" \
    --trace_every "$TRACE_EVERY" \
    --log_interval "$LOG_INTERVAL" \
    --eval_interval "$EVAL_INTERVAL" \
    --experiment_id "${MODEL_LABEL}_${WORKLOAD}_seed${SEED}" \
    --output_dir "$PROFILE_OUT"
done

python "$SCRIPT_DIR/rq1_analyze_workload_stability.py" \
  --profile_glob "$OUT/profile_${MODEL_LABEL}_*/profiles/profile_*.csv" \
  --output_dir "$OUT/analysis" \
  --group_by model_name

echo "[done] RQ1 workload-stability outputs: $OUT"
echo "[done] Pairwise metrics: $OUT/analysis/rq1_workload_pairwise.csv"
echo "[done] Summary: $OUT/analysis/rq1_workload_summary_by_model.csv"
