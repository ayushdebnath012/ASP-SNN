# ASP-SNN Optimized Runners

This repo is the final clean home for optimized ASP-SNN code:

- `train_a100.py` - self-contained ModelNet10/ModelNet40 A100 runner.
- `colab_asp_mn40_v4.py` - single-cell Colab/T4 ModelNet40 runner.
- `colab_shapenetpart_t4.py` - single-cell Colab/T4 ShapeNetPart runner.
- `scanobjectnn_full.py` - Kaggle/HPC launcher for ScanObjectNN.
- `shapenetpart_full.py` - Kaggle/HPC launcher for ShapeNetPart.
- `s3dis_full.py` - Kaggle/HPC launcher for S3DIS.

The legacy nested `purdueprj/` tree, old comparison notebooks, generated
reports, zip bundles, checkpoints, and result folders are intentionally not
part of the final repo.

## Clone

```bash
git clone https://github.com/ayush31010/ASP-SNN.git
cd ASP-SNN
```

## Environment

```bash
bash setup.sh
conda activate asp-snn
python smoke_test.py
```

If you already have a CUDA PyTorch environment:

```bash
pip install -r requirements.txt
pip install kagglehub trimesh gdown
```

## ModelNet40 On Colab

Upload or open `colab_asp_mn40_v4.py` in Colab and run it as a single cell:

```python
exec(open("colab_asp_mn40_v4.py").read())
```

It downloads ModelNet40 with `kagglehub`, uses the stronger SPM-style backbone,
keeps checkpoints in Google Drive when mounted, and supports resume.

## ShapeNetPart On Colab

Upload or open `colab_shapenetpart_t4.py` in Colab and run it as a single cell.
It prepares the normal-enabled ShapeNetPart splits, mounts Google Drive when
available, and resumes from saved checkpoints.

## A100 ModelNet Runner

```bash
python train_a100.py --datasets ModelNet10,ModelNet40 --epochs 300
```

Useful overrides:

```bash
python train_a100.py --datasets ModelNet40 --batch 64 --vote 10
python train_a100.py --datasets ModelNet10 --no_kd
```

## Package-Compatible Full Runners

These launchers use the repo's normal training scripts and configs:

```bash
python scanobjectnn_full.py
python shapenetpart_full.py
python s3dis_full.py
```

They auto-detect Kaggle input folders. If the project is not attached in Kaggle,
their fallback clone uses:

```text
https://github.com/ayush31010/ASP-SNN.git
```

## SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=asp_scanobj
#SBATCH --output=logs/scanobj_%j.out
#SBATCH --error=logs/scanobj_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load anaconda/2023
conda activate asp-snn

cd /path/to/ASP-SNN
python scanobjectnn_full.py
```

Create `logs/` before submission if your cluster requires the output directory
to exist:

```bash
mkdir -p logs
sbatch run_scanobj.sh
```

## Generated Files

Training outputs belong in ignored directories such as `checkpoints/`, `logs/`,
`outputs/`, `results/`, or `a100_ckpts/`. Keep large datasets, checkpoints,
archives, and generated reports out of git.
