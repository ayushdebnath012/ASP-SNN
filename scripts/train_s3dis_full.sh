#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
TEST_AREA="${TEST_AREA:-5}"
LOG_FILE="${LOG_FILE:-logs/s3dis_full_gpu${GPU}.log}"

echo "Running S3DIS full training on GPU ${GPU}"
echo "Log file: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u train_s3dis.py \
  --config configs/s3dis_seg.yaml \
  --set epochs="${EPOCHS}" batch_size="${BATCH_SIZE}" \
        num_workers="${NUM_WORKERS}" eval_interval="${EVAL_INTERVAL}" \
        log_interval="${LOG_INTERVAL}" test_area="${TEST_AREA}" \
  2>&1 | tee "${LOG_FILE}"
