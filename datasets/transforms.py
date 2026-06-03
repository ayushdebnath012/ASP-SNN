"""Numpy augmentations used by the point-cloud dataset loaders."""

from __future__ import annotations

import math

import numpy as np


def _rand_z_rotation() -> np.ndarray:
    theta = float(np.random.uniform(0.0, 2.0 * math.pi))
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _rand_so3_rotation() -> np.ndarray:
    # QR gives an orthonormal basis; flip if needed to keep det=+1.
    q, _ = np.linalg.qr(np.random.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def _apply_xyz_aug(xyz: np.ndarray, cfg) -> np.ndarray:
    out = xyz.astype(np.float32, copy=True)

    if getattr(cfg, "aug_rotate_so3", False):
        out = out @ _rand_so3_rotation().T
    elif getattr(cfg, "aug_rotate_z", False):
        out = out @ _rand_z_rotation().T

    lo = float(getattr(cfg, "aug_scale_lo", 1.0))
    hi = float(getattr(cfg, "aug_scale_hi", 1.0))
    if lo != 1.0 or hi != 1.0:
        out *= float(np.random.uniform(lo, hi))

    translate = float(getattr(cfg, "aug_translate", 0.0))
    if translate > 0:
        out += np.random.uniform(-translate, translate, size=(1, 3)).astype(np.float32)

    sigma = float(getattr(cfg, "aug_jitter_sigma", 0.0))
    if sigma > 0:
        clip = float(getattr(cfg, "aug_jitter_clip", 0.05))
        noise = np.clip(np.random.normal(0.0, sigma, out.shape), -clip, clip)
        out += noise.astype(np.float32)

    return out


def _point_dropout(points: np.ndarray, prob: float) -> np.ndarray:
    if prob <= 0 or len(points) == 0:
        return points
    out = points.copy()
    drop = np.random.random(len(out)) < prob
    if np.any(drop):
        keep = np.where(~drop)[0]
        fill = keep[0] if len(keep) else 0
        out[drop] = out[fill]
    return out


def augment_slices(slices: np.ndarray, cfg) -> np.ndarray:
    """Augment classification slices while preserving shape."""
    out = slices.astype(np.float32, copy=True)
    flat_xyz = out[..., :3].reshape(-1, 3)
    out[..., :3] = _apply_xyz_aug(flat_xyz, cfg).reshape(out.shape[0], out.shape[1], 3)

    prob = float(getattr(cfg, "aug_point_dropout", 0.0))
    if prob > 0:
        for i in range(out.shape[0]):
            out[i] = _point_dropout(out[i], prob)

    slice_prob = float(getattr(cfg, "aug_slice_dropout", 0.0))
    if slice_prob > 0 and out.shape[0] > 1:
        drop = np.random.random(out.shape[0]) < slice_prob
        if np.any(drop):
            keep = np.where(~drop)[0]
            fill = keep[0] if len(keep) else 0
            out[drop] = out[fill]

    return out


def augment_seg(slices: np.ndarray, pts_features: np.ndarray, cfg):
    """Apply a shared geometric augmentation to segmentation slices and points."""
    out_slices = slices.astype(np.float32, copy=True)
    out_pts = pts_features.astype(np.float32, copy=True)

    if getattr(cfg, "aug_rotate_so3", False):
        rot = _rand_so3_rotation()
    elif getattr(cfg, "aug_rotate_z", False):
        rot = _rand_z_rotation()
    else:
        rot = np.eye(3, dtype=np.float32)

    lo = float(getattr(cfg, "aug_scale_lo", 1.0))
    hi = float(getattr(cfg, "aug_scale_hi", 1.0))
    scale = float(np.random.uniform(lo, hi)) if (lo != 1.0 or hi != 1.0) else 1.0

    translate = float(getattr(cfg, "aug_translate", 0.0))
    shift = (
        np.random.uniform(-translate, translate, size=(1, 3)).astype(np.float32)
        if translate > 0
        else np.zeros((1, 3), dtype=np.float32)
    )

    def transform(xyz: np.ndarray) -> np.ndarray:
        aug = (xyz @ rot.T) * scale + shift
        sigma = float(getattr(cfg, "aug_jitter_sigma", 0.0))
        if sigma > 0:
            clip = float(getattr(cfg, "aug_jitter_clip", 0.05))
            aug += np.clip(np.random.normal(0.0, sigma, aug.shape), -clip, clip).astype(
                np.float32
            )
        return aug.astype(np.float32)

    out_slices[..., :3] = transform(out_slices[..., :3].reshape(-1, 3)).reshape(
        out_slices.shape[0], out_slices.shape[1], 3
    )
    out_pts[:, :3] = transform(out_pts[:, :3])

    color_drop = float(getattr(cfg, "aug_color_drop", 0.0))
    if out_pts.shape[1] >= 6 and color_drop > 0 and np.random.random() < color_drop:
        out_pts[:, 3:6] = 0.0
        if out_slices.shape[-1] >= 6:
            out_slices[..., 3:6] = 0.0

    color_jitter = float(getattr(cfg, "aug_color_jitter", 0.0))
    if out_pts.shape[1] >= 6 and color_jitter > 0:
        noise = np.random.normal(0.0, color_jitter, out_pts[:, 3:6].shape).astype(
            np.float32
        )
        out_pts[:, 3:6] = np.clip(out_pts[:, 3:6] + noise, 0.0, 1.0)
        if out_slices.shape[-1] >= 6:
            noise_s = np.random.normal(
                0.0, color_jitter, out_slices[..., 3:6].shape
            ).astype(np.float32)
            out_slices[..., 3:6] = np.clip(out_slices[..., 3:6] + noise_s, 0.0, 1.0)

    return out_slices, out_pts

