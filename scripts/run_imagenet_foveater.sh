#!/usr/bin/env bash
set -euo pipefail

# Full ImageNet/ImageNet-100 training:
#   DATA_DIR=/path/to/imagenet NUM_CLASSES=1000 bash scripts/run_imagenet_foveater.sh
#   DATA_DIR=/path/to/imagenet100 NUM_CLASSES=100 bash scripts/run_imagenet_foveater.sh

DATA_DIR="${DATA_DIR:-data/imagenet}"
NUM_CLASSES="${NUM_CLASSES:-1000}"

python train_imagenet_foveater.py \
  --config configs/imagenet_foveater.yaml \
  --set data_dir="${DATA_DIR}" num_classes="${NUM_CLASSES}"
