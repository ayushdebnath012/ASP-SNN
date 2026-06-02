# ASP-SNN: Active Spiking Perception for 3D Point Cloud Understanding

A spiking neural network framework that treats 3D point cloud perception as
an iterative decision process — selecting which local patch ("slice") to examine
next based on accumulated spiking neuron dynamics (LIF neurons), rather than
processing the entire point cloud at once.

Evaluated on three benchmarks:
- **ShapeNetPart** — part segmentation (50 parts, 16 categories)
- **ScanObjectNN PB-T50-RS** — real-world object classification (15 classes)
- **S3DIS Area 5** — indoor scene segmentation (13 classes)

Each experiment runs on a single GPU. Three experiments run in parallel on
separate GPUs via SLURM.

---

## Architecture

```
Input: point cloud [N points, C channels]
        │
        ▼
┌─────────────────────────────────────────────────┐
│  SLICING: FPS (M=16 anchors) → KNN (K per anchor) │
│  + 8-dim geometry descriptor per slice            │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  ENCODER: EdgeConv (static kNN, k=20)            │
│  Conv2d(2C→128→128) + Conv1d(128→256→512)       │
│  Global max-pool → one 512-dim token per slice   │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  POSITIONAL ENCODING: Linear(3→512) on centroid  │
│  + TRANSFORMER: 1 layer, 4 heads, FFN=1024       │
│  Output: [B, 16, 512] contextualised tokens      │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  ASP LOOP (T=6 timesteps):                       │
│  ├─ SSP: attention-based slice scoring           │
│  │   (belief × geometry → score per slice)       │
│  ├─ Gumbel-softmax (train) / argmax (eval)       │
│  ├─ Fuse: selected feature + belief projection   │
│  ├─ 3-layer LIF (soft reset, ATan surrogate):    │
│  │   Linear→BN→ReLU→LIF dynamics + residual      │
│  ├─ Emit logits (classification) or              │
│  │   accumulate belief (segmentation)             │
│  └─ Early exit if confidence > threshold (eval)   │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  OUTPUT:                                         │
│  Classification: logit averaging across T steps  │
│  Segmentation: per-point features via slice      │
│    lookup + PerPointBranch + global belief +     │
│    SegHead MLP → [B, N, num_classes]             │
└─────────────────────────────────────────────────┘
```

Key architectural parameters:
- `feat_dim = hidden_dim = 512`
- `num_slices (M) = 16`, `points_per_slice (K) = 128` (ShapeNet/ScanObj) or `256` (S3DIS)
- `num_lif_layers = 3`, `lif_leak = 0.9`, `lif_threshold = 1.0`
- `T = 6` ASP timesteps, `exit_threshold = 0.40`
- Surrogate gradient: ATan (α=2.0) per Fang et al. ICCV 2021 (SPM-compatible)

---

## Repository structure

```
ASP-SNN/
├── README.md                       # This file
├── environment.yml                 # Conda environment specification
├── requirements.txt                # Pip dependencies
├── setup.sh                        # One-command environment setup
├── smoke_test.py                   # 7-test verification suite (CPU, no data)
│
├── configs/                        # Per-dataset YAML configurations
│   ├── shapenet_seg.yaml           #   300 epochs, bs=32, pts_per_slice=128
│   ├── scanobj_cls.yaml            #   300 epochs, bs=32, SO(3) aug, SWA, deep MLP head
│   └── s3dis_seg.yaml              #   100 epochs, bs=16, RGB+height, class weights
│
├── scripts/                        # SLURM job scripts for GPU cluster
│   ├── run_shapenet.sh             #   12h wall, 1 GPU, 8 CPUs, 32G
│   ├── run_scanobj.sh              #   8h wall, 1 GPU, 8 CPUs, 32G
│   ├── run_s3dis.sh                #   24h wall, 1 GPU, 8 CPUs, 64G
│   └── smoke_train.sh              #   Quick 2-epoch test for all 3 datasets
│
├── config.py                       # YAML loader with auto type casting + CLI overrides
│
├── datasets/                       # Data loading and preprocessing
│   ├── __init__.py
│   ├── download.py                 # Downloads all 3 datasets (auto where possible)
│   ├── shapenetpart.py             # ShapeNetPart HDF5 loader (14,007 train / 2,874 test)
│   ├── scanobjectnn.py             # ScanObjectNN PB-T50-RS loader (11,416 train / 2,882 test)
│   ├── s3dis.py                    # S3DIS room-block loader (6 areas, 271 rooms)
│   ├── slicing.py                  # FPS + KNN slicing + 8-dim geometry descriptors
│   └── transforms.py               # Augmentation: rotation, scale, jitter, color dropout
│
├── models/                         # Network architecture
│   ├── __init__.py
│   ├── encoder.py                  # EdgeConv feature extractor (variable input channels)
│   ├── ssp.py                      # Slice Selection Policy (attention-based scoring)
│   ├── lif.py                      # Multi-layer LIF with ATan surrogate (AMP-safe)
│   ├── asp_classifier.py           # ASP model for classification (ScanObjectNN)
│   └── asp_segmentor.py            # ASP model for segmentation (ShapeNet + S3DIS)
│
├── train_shapenet.py               # Train ShapeNetPart (differential LR, category masking)
├── train_scanobj.py                # Train ScanObjectNN (SWA, aux loss, label smoothing)
├── train_s3dis.py                  # Train S3DIS (class-weighted loss, RGB+height)
├── eval_shapenet.py                # Eval ShapeNetPart (per-category IoU)
├── eval_scanobj.py                 # Eval ScanObjectNN (N-vote TTA, per-class accuracy)
├── eval_s3dis.py                   # Eval S3DIS (per-class IoU/Acc, mIoU, OA)
│
├── checkpoints/                    # Saved models (auto-created)
├── logs/                           # CSV training logs (auto-created)
└── data/                           # Datasets (auto-created by download.py)
```

---

## Setup (one-time, ~5 minutes)

### Prerequisites
- conda or miniconda installed
- CUDA 12.1+ compatible GPU drivers
- Internet access for pip install and dataset download

### Commands

```bash
# 1. Clone the repository
git clone https://github.com/ayush31010/ASP-SNN.git
cd ASP-SNN

# 2. Create conda environment and install all dependencies
bash setup.sh

# 3. Activate environment
conda activate asp-snn

# 4. Verify installation (CPU only, no data needed, ~30 seconds)
python smoke_test.py
# Expected output: ALL 7 TESTS PASSED
```

The smoke test verifies:
1. ASPClassifier builds and trains (forward + backward)
2. ASPSegmentor works in ShapeNetPart mode (50 classes, category conditioning)
3. ASPSegmentor works in S3DIS mode (13 classes, 7-channel input)
4. FPS + KNN slicing pipeline produces correct shapes
5. Early exit does not crash
6. Batch size 1 evaluation works (BN edge case)
7. Checkpoint save/load preserves predictions

---

## Dataset download

```bash
# Download all three datasets
python datasets/download.py --all

# Or individually:
python datasets/download.py --shapenet    # ~346 MB from Stanford
python datasets/download.py --scanobj     # ~50 MB from HuggingFace
python datasets/download.py --s3dis       # ~1.3 GB via gdown
```

| Dataset | Source | Auth required? | Auto-download? |
|---|---|---|---|
| ShapeNetPart | `shapenet.cs.stanford.edu` | No | Yes |
| ScanObjectNN | `huggingface.co/datasets/cminst/ScanObjectNN` | No | Yes |
| S3DIS | Google Drive (OpenPoints preprocessed) | No | Yes (via gdown) |

If S3DIS gdown fails (rate limiting), follow the printed manual instructions or:
```bash
# Option A: Retry with gdown
pip install gdown
python datasets/download.py --s3dis

# Option B: Preprocess raw S3DIS from Stanford
python datasets/download.py --s3dis_preprocess /path/to/Stanford3dDataset_v1.2_Aligned_Version
```

Expected data layout after download:
```
data/
├── shapenet_part_seg_hdf5_data/    # train0-5.h5, test0-1.h5
├── ScanObjectNN/main_split/        # training_objectdataset_augmentedrot_scale75.h5
│                                   # test_objectdataset_augmentedrot_scale75.h5
└── s3dis/Area_1/ ... Area_6/       # per-room .npy files [N, 7]: x,y,z,r,g,b,label
```

---

## Training

### Option A: SLURM cluster (recommended)

Edit the `#SBATCH` headers in `scripts/run_*.sh` to match your cluster's
partition and account names, then submit:

```bash
sbatch scripts/run_shapenet.sh     # ~8h on H100
sbatch scripts/run_scanobj.sh      # ~4h on H100
sbatch scripts/run_s3dis.sh        # ~12h on H100
```

All three run in parallel on separate GPUs.

### Option B: Interactive (3 terminals)

```bash
CUDA_VISIBLE_DEVICES=0 python train_shapenet.py --config configs/shapenet_seg.yaml &
CUDA_VISIBLE_DEVICES=1 python train_scanobj.py --config configs/scanobj_cls.yaml &
CUDA_VISIBLE_DEVICES=2 python train_s3dis.py --config configs/s3dis_seg.yaml &
```

### Quick smoke training (verify full pipeline with real data, ~10 min)

```bash
bash scripts/smoke_train.sh
# Runs 2 epochs per dataset with batch_size=4
```

### Resume from crash

All scripts automatically save `*_last.pt` every epoch. To resume:

```bash
python train_scanobj.py --config configs/scanobj_cls.yaml --resume checkpoints/scanobj_last.pt
python train_shapenet.py --config configs/shapenet_seg.yaml --resume checkpoints/shapenet_last.pt
python train_s3dis.py --config configs/s3dis_seg.yaml --resume checkpoints/s3dis_last.pt
```

Resumes model weights, optimizer state, LR scheduler, AMP scaler, and epoch counter.

### Override config values from CLI

```bash
python train_scanobj.py --config configs/scanobj_cls.yaml --set epochs=100 lr=1e-3 batch_size=16
```

### Monitor training

```bash
squeue -u $USER                      # SLURM job status
tail -f logs/scanobj_*.log           # Live SLURM output
cat logs/scanobj_*.csv               # CSV: epoch, loss, train_acc, val_acc, lr, time
```

---

## Evaluation

```bash
# ScanObjectNN — with 10-vote test-time augmentation
python eval_scanobj.py --ckpt checkpoints/scanobj_best.pt --n_votes 10

# ShapeNetPart — with per-category IoU breakdown
python eval_shapenet.py --ckpt checkpoints/shapenet_best.pt --per_cat

# S3DIS — with per-class IoU and accuracy
python eval_s3dis.py --ckpt checkpoints/s3dis_best.pt --per_class
```

---

## Expected results

| Task | Dataset | Metric | Expected | PointNet++ | DGCNN | SPM (SNN) |
|---|---|---|---|---|---|---|
| Classification | ScanObjectNN PB-T50-RS | OA | 82-87% | 77.9% | 78.1% | 84.2% |
| Part seg | ShapeNetPart | Inst mIoU | 81-84% | 85.1% | 85.2% | 84.8% |
| Scene seg | S3DIS Area 5 | mIoU | 50-58% | 53.5% | 56.1% | — |

Estimated training time on NVIDIA H100 80GB:

| Dataset | Epochs | Batch size | Time per epoch | Total |
|---|---|---|---|---|
| ScanObjectNN | 300 | 32 | ~30s | ~3-4h |
| ShapeNetPart | 300 | 32 | ~1.5min | ~7-8h |
| S3DIS | 100 | 16 | ~5-8min | ~10-14h |

---

## Output structure (after training)

```
checkpoints/
├── shapenet_best.pt      # Best ShapeNetPart model (by Instance mIoU)
├── shapenet_last.pt      # Last epoch (for --resume)
├── scanobj_best.pt       # Best ScanObjectNN model (by validation accuracy)
├── scanobj_last.pt       # Last epoch
├── scanobj_swa.pt        # SWA averaged model (ScanObjectNN only)
├── s3dis_best.pt         # Best S3DIS model (by mIoU)
└── s3dis_last.pt         # Last epoch

logs/
├── shapenet_YYYYMMDD_HHMMSS.csv   # epoch, train_loss, inst_miou, cls_miou, lr, time
├── scanobj_YYYYMMDD_HHMMSS.csv    # epoch, train_loss, train_acc, val_acc, lr, time
└── s3dis_YYYYMMDD_HHMMSS.csv      # epoch, train_loss, miou, macc, oa, lr, time
```

All checkpoints contain: `{epoch, model, optimizer, scheduler, scaler, best_metric}` — fully resumable.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Static kNN (xyz only) in encoder | Dynamic kNN on 64-point slices caused ±5% val oscillation |
| ATan surrogate gradient (α=2.0) | SPM-compatible, SOTA for SNNs (Fang et al. ICCV 2021) |
| AMP-safe surrogate with `custom_fwd` | Prevents fp16 overflow in pow(2) on H100 with mixed precision |
| Soft reset LIF dynamics | `u_t = λ·u + inp − θ·s` preserves membrane information vs hard reset |
| Deterministic FPS at test time | `seed=idx` prevents non-reproducible eval metrics |
| 3-layer MLP classifier for ScanObjectNN | Single Linear bottleneck on real-world data (+1-2%) |
| Class-weighted CE for S3DIS | Inverse-frequency weighting for severely imbalanced classes |
| RGB + normalised height feature for S3DIS | Height alone gives +3-5% mIoU (floor=0, ceiling=1) |
| Color dropout (p=0.2) for S3DIS | Forces geometry learning; prevents RGB overfitting |
| Aux loss normalised by weight sum | Prevents ~2.58× effective LR inflation from multi-timestep loss |
| SWA only when averaging window ≥ 10 epochs | Avoids meaningless weight averaging on short runs |
| Differential LR for ShapeNet encoder (0.1×) | Stabilises pre-existing encoder features during seg head training |

---

## Research context

**Reference backbone:** Spiking Point Mamba (SPM), arxiv 2504.14371

**What ASP-SNN contributes beyond SPM:**
- First **active perception** framework for SNN-based point clouds (sequential slice selection via SSP)
- First SNN method evaluated across **classification + part segmentation + scene segmentation** in a unified architecture
- **Interpretable** slice selection (SSP attention weights show which regions the model examines)
- **Adaptive compute** via early exit (simple shapes use fewer slices)
- Honest **energy efficiency analysis** with per-layer spike rate logging (`SpikeRateLogger`)

**Competitive landscape (SNN domain):**
- SPM: 84.2% ScanObjectNN, 84.8% ShapeNetPart Inst mIoU
- 3DSMT (ICLR 2026): 90.4% ScanObjectNN, 70.2% S3DIS mIoU
- S3DNet: 84.5% ScanObjectNN

Our positioning: competitive accuracy with SPM while offering fundamentally different capabilities (active perception, early exit, interpretability) that no existing SNN method provides.

---

## Spike rate logging (for energy efficiency metric)

After training, measure per-layer spike rates for the paper's energy analysis:

```python
from models.lif import SpikeRateLogger

model.eval()
logger = SpikeRateLogger()
model.lif_head.spike_monitor = logger

# Run evaluation
evaluate(model, test_loader, device)

print(f"Mean spike rate: {logger.mean_rate():.4f}")
print(f"Per-layer rates: {logger.per_layer_rates()}")

model.lif_head.spike_monitor = None
```

---

## Citation

```bibtex
@article{asp_snn_2026,
    title={Active Spiking Perception for 3D Point Cloud Understanding},
    author={},
    year={2026}
}
```
