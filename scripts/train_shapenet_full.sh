#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
LOG_FILE="${LOG_FILE:-logs/shapenet_full_gpu${GPU}.log}"

echo "Running ShapeNetPart full training on GPU ${GPU}"
echo "Log file: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 python -u train_shapenet.py \
  --config configs/shapenet_seg.yaml \
  --set epochs="${EPOCHS}" batch_size="${BATCH_SIZE}" \
        num_workers="${NUM_WORKERS}" eval_interval="${EVAL_INTERVAL}" \
  2>&1 | tee "${LOG_FILE}"
