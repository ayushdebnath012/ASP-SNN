# Full experiments and metrics

## ModelNet classification

| Runner | Dataset | Primary output |
|---|---|---|
| `experiments/full/train_spikegat_modelnet10.py` | ModelNet10 | `final_metrics.json` |
| `experiments/full/train_spikegat_modelnet40.py` | ModelNet40 | `final_metrics.json` |
| `experiments/full/train_asp_modelnet_a100.py` | ModelNet10/40 | aggregate results/history |

The SpikeGAT code preserves Max-First graph aggregation, uses the supplementary
MPR/APTEC equations, initializes attention as identity, transfers ANN teacher
weights, and separates single-pass OA from scale-TTA OA.

ModelNet40 additionally caches one canonical teacher distribution per training
shape, avoiding a second dynamic-graph forward pass during every student batch.

Training jobs require prepared datasets and explicit paths. They do not mount
cloud drives, install packages, or download data at runtime.

## Other point-cloud tasks

| Task | Full training command |
|---|---|
| ScanObjectNN | `python -m tasks.train_scanobjectnn --config configs/scanobj_cls.yaml` |
| ShapeNetPart | `python -m tasks.train_shapenetpart --config configs/shapenet_seg.yaml` |
| S3DIS | `python -m tasks.train_s3dis --config configs/s3dis_seg.yaml` |

## Fair comparison checklist

1. Record the exact train/test split and point count.
2. Report single-pass metrics first.
3. Label voting or TTA metrics explicitly.
4. Keep the random seed and best-checkpoint selection rule in the result file.
5. Do not treat a smoke test or teacher accuracy as a student result.
6. Preserve the complete checkpoint and `final_metrics.json` for auditability.
