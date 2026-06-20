#!/usr/bin/env bash
#SBATCH --job-name=asp-modelnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT="${ASP_SNN_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$ROOT"
source "${ASP_SNN_VENV:-$ROOT/.venv}/bin/activate"

OUTPUT_ROOT="${ASP_SNN_OUTPUT_ROOT:-$ROOT/outputs}"
mkdir -p "$OUTPUT_ROOT/modelnet_a100"
: "${MODELNET10_DIR:?Set MODELNET10_DIR before sbatch}"
: "${MODELNET40_DIR:?Set MODELNET40_DIR before sbatch}"

srun python -u experiments/full/train_asp_modelnet_a100.py \
  --datasets "${DATASETS:-ModelNet10,ModelNet40}" \
  --epochs "${EPOCHS:-300}" \
  --batch "${BATCH_SIZE:-64}" \
  --ckpt_dir "$OUTPUT_ROOT/modelnet_a100" \
  --modelnet10_dir "$MODELNET10_DIR" \
  --modelnet40_dir "$MODELNET40_DIR"
