#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data

DATASET="${DATASET:-majdouline20/shapenetpart-dataset}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-data/kaggle_shapenetpart}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found."
  echo "Install/authenticate it first:"
  echo "  pip install kaggle"
  echo "  mkdir -p ~/.kaggle"
  echo "  # put kaggle.json from Kaggle account settings into ~/.kaggle/"
  echo "  chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

mkdir -p "${DOWNLOAD_DIR}"

echo "Downloading Kaggle ShapeNetPart dataset: ${DATASET}"
kaggle datasets download -d "${DATASET}" -p "${DOWNLOAD_DIR}" --unzip

echo "Converting raw PartAnnotation to HDF5 expected by train_shapenet.py"
python datasets/download.py --shapenet_raw "${DOWNLOAD_DIR}"

echo "ShapeNetPart is ready."
echo "Run training with:"
echo "  bash scripts/train_shapenet_full.sh"
