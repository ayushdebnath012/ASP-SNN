"""
datasets/download.py — Download and prepare all three datasets.

Download methods per dataset:
    ShapeNetPart  : Direct wget from Stanford (no auth needed)
    ScanObjectNN  : HuggingFace mirror (no form needed) → gdown fallback
    S3DIS         : gdown from Google Drive (OpenPoints preprocessed) → manual fallback

All methods tested from US university networks.

Usage:
    python datasets/download.py --all
    python datasets/download.py --shapenet
    python datasets/download.py --scanobj
    python datasets/download.py --s3dis
    python datasets/download.py --s3dis_preprocess /path/to/Stanford3dDataset_v1.2
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
import tarfile
import urllib.request

import numpy as np

DATA_ROOT = "data"


# ═══════════════════════════════════════════════════════════════════════
#  ShapeNetPart — Stanford direct download (no auth)
# ═══════════════════════════════════════════════════════════════════════

SHAPENET_URL = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
SHAPENET_DIR = "shapenet_part_seg_hdf5_data"
SHAPENET_RAW_SYNSETS = {
    "Airplane": "02691156",
    "Bag": "02773838",
    "Cap": "02954340",
    "Car": "02958343",
    "Chair": "03001627",
    "Earphone": "03261776",
    "Guitar": "03467517",
    "Knife": "03624134",
    "Lamp": "03636649",
    "Laptop": "03642806",
    "Motorbike": "03790512",
    "Mug": "03797390",
    "Pistol": "03948459",
    "Rocket": "04099429",
    "Skateboard": "04225987",
    "Table": "04379243",
}
SHAPENET_CATEGORY_NAMES = list(SHAPENET_RAW_SYNSETS.keys())
SHAPENET_PART_RANGES = {
    0: [0, 1, 2, 3],
    1: [4, 5],
    2: [6, 7],
    3: [8, 9, 10, 11],
    4: [12, 13, 14, 15],
    5: [16, 17, 18],
    6: [19, 20, 21],
    7: [22, 23],
    8: [24, 25, 26, 27],
    9: [28, 29],
    10: [30, 31, 32, 33, 34, 35],
    11: [36, 37],
    12: [38, 39, 40],
    13: [41, 42, 43],
    14: [44, 45],
    15: [46, 47, 48, 49],
}


def download_shapenet():
    """
    ShapeNetPart HDF5 — direct from Stanford.
    ~346 MB zip. No authentication required.
    Contains train0-5.h5, test0-1.h5 with 14,007 train / 2,874 test shapes.
    """
    out_dir = os.path.join(DATA_ROOT, SHAPENET_DIR)
    sentinel = os.path.join(out_dir, "all_object_categories.txt")

    if os.path.exists(sentinel):
        print(f"[ShapeNet] Already present at {out_dir}")
        return True

    os.makedirs(DATA_ROOT, exist_ok=True)
    zip_path = os.path.join(DATA_ROOT, "shapenet_part_seg_hdf5_data.zip")

    print(f"[ShapeNet] Downloading from Stanford ({SHAPENET_URL}) ...")
    try:
        _download_with_progress(SHAPENET_URL, zip_path)
    except Exception as e:
        print(f"[ShapeNet] FAILED: {e}")
        print("[ShapeNet] The Stanford server may be temporarily down.")
        print(f"[ShapeNet] Manual: wget {SHAPENET_URL} -O {zip_path}")
        return False

    print(f"[ShapeNet] Extracting ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_ROOT)
    os.remove(zip_path)

    import glob
    n_train = len(glob.glob(os.path.join(out_dir, "train*.h5")))
    n_test = len(glob.glob(os.path.join(out_dir, "test*.h5")))
    print(f"[ShapeNet] Done: {n_train} train + {n_test} test H5 files")
    return True


def _find_shapenet_h5_dir(root: str):
    """Find a directory containing ShapeNetPart HDF5 train/test files."""
    for cur, _, files in os.walk(root):
        has_train = any(f.startswith("train") and f.endswith(".h5") for f in files)
        has_test = any(f.startswith("test") and f.endswith(".h5") for f in files)
        if has_train and has_test:
            return cur
    return None


def _find_shapenet_raw_dir(root: str):
    """Find ShapeNetPart PartAnnotation root with category synset folders."""
    expected = set(SHAPENET_RAW_SYNSETS.values())
    for cur, dirs, files in os.walk(root):
        if "synsetoffset2category.txt" in files:
            return cur
        if len(expected.intersection(dirs)) >= 8:
            return cur
    return None


def _extract_archive(archive_path: str, dest_dir: str):
    """Extract zip/tar archives even when the extension is misleading."""
    os.makedirs(dest_dir, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        return True
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
        return True
    try:
        shutil.unpack_archive(archive_path, dest_dir)
        return True
    except Exception:
        return False


def prepare_shapenet_local(src_path: str):
    """
    Prepare a local/Kaggle ShapeNetPart download.

    This is valid when the extracted data contains the Stanford-style HDF5
    files: train*.h5, test*.h5, data/label/pid keys.
    """
    if not os.path.exists(src_path):
        print(f"[ShapeNet] Local path not found: {src_path}")
        return False

    os.makedirs(DATA_ROOT, exist_ok=True)
    target = os.path.join(DATA_ROOT, SHAPENET_DIR)

    search_root = src_path
    if os.path.isfile(src_path):
        extract_root = os.path.join(DATA_ROOT, "_shapenet_local_extract")
        print(f"[ShapeNet] Extracting local archive: {src_path}")
        if not _extract_archive(src_path, extract_root):
            print("[ShapeNet] Could not extract local archive.")
            return False
        search_root = extract_root

    h5_dir = _find_shapenet_h5_dir(search_root)
    if h5_dir is None:
        print("[ShapeNet] Could not find train*.h5 and test*.h5 files.")
        print("[ShapeNet] Kaggle is OK only if it contains the HDF5 ShapeNetPart format.")
        return False

    if os.path.abspath(h5_dir) != os.path.abspath(target):
        print(f"[ShapeNet] Copying HDF5 files from {h5_dir} -> {target}")
        shutil.copytree(h5_dir, target, dirs_exist_ok=True)

    import glob
    n_train = len(glob.glob(os.path.join(target, "train*.h5")))
    n_test = len(glob.glob(os.path.join(target, "test*.h5")))
    print(f"[ShapeNet] Ready at {target}: {n_train} train + {n_test} test H5 files")
    return n_train > 0 and n_test > 0


def _normalise_shape_id(entry: str):
    parts = entry.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part in SHAPENET_RAW_SYNSETS.values() and i + 1 < len(parts):
            return f"{part}/{parts[i + 1].split('.')[0]}"
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1].split('.')[0]}"
    return entry.split(".")[0]


def _load_shapenet_split_sets(raw_dir: str):
    """Load official ShapeNetPart split JSONs when present."""
    search_roots = [raw_dir, os.path.dirname(raw_dir), os.path.dirname(os.path.dirname(raw_dir))]
    split_dir = None
    for root in search_roots:
        candidate = os.path.join(root, "train_test_split")
        if os.path.isdir(candidate):
            split_dir = candidate
            break
    if split_dir is None:
        return None, None

    def load_json(name):
        path = os.path.join(split_dir, name)
        if not os.path.exists(path):
            return set()
        with open(path) as f:
            return {_normalise_shape_id(x) for x in json.load(f)}

    train_ids = load_json("shuffled_train_file_list.json")
    val_ids = load_json("shuffled_val_file_list.json")
    test_ids = load_json("shuffled_test_file_list.json")
    train_ids = train_ids | val_ids
    if train_ids and test_ids:
        print(f"[ShapeNet] Using official split files from {split_dir}")
        return train_ids, test_ids
    return None, None


def _sample_points(points: np.ndarray, labels: np.ndarray,
                   n_points: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(points)
    if n >= n_points:
        idx = rng.choice(n, n_points, replace=False)
    else:
        idx = rng.choice(n, n_points, replace=True)
    return points[idx].astype(np.float32), labels[idx].astype(np.int64)


def _map_raw_part_labels(raw_labels: np.ndarray, cat_id: int):
    """Map raw ShapeNetPart local labels to global 0-49 part IDs."""
    labels = raw_labels.astype(np.int64)
    parts = SHAPENET_PART_RANGES[cat_id]
    label_min = int(labels.min()) if labels.size else 0
    label_max = int(labels.max()) if labels.size else 0

    if label_min >= 1 and label_max <= len(parts):
        lookup = np.array(parts, dtype=np.int64)
        return lookup[labels - 1]

    if label_min >= 0 and label_max < len(parts):
        lookup = np.array(parts, dtype=np.int64)
        return lookup[labels]

    if set(np.unique(labels).tolist()).issubset(set(parts)):
        return labels

    mapped = labels.copy()
    for raw, global_id in zip(sorted(np.unique(labels)), parts):
        mapped[labels == raw] = global_id
    return mapped


def _collect_shapenet_raw_samples(raw_dir: str, n_points: int):
    train_ids, test_ids = _load_shapenet_split_sets(raw_dir)
    rng = np.random.default_rng(42)
    samples = []

    synset_to_cat = {v: i for i, v in enumerate(SHAPENET_RAW_SYNSETS.values())}
    for synset, cat_id in synset_to_cat.items():
        cat_dir = os.path.join(raw_dir, synset)
        points_dir = os.path.join(cat_dir, "points")
        label_dirs = [
            os.path.join(cat_dir, "expert_verified", "points_label"),
            os.path.join(cat_dir, "points_label"),
        ]
        label_dir = next((d for d in label_dirs if os.path.isdir(d)), None)
        if not os.path.isdir(points_dir) or label_dir is None:
            print(f"[ShapeNet] Skipping {synset}: missing points/labels")
            continue

        point_files = sorted(
            f for f in os.listdir(points_dir)
            if f.endswith(".pts") or f.endswith(".txt")
        )
        for fname in point_files:
            stem = os.path.splitext(fname)[0]
            seg_path = os.path.join(label_dir, f"{stem}.seg")
            if not os.path.exists(seg_path):
                seg_path = os.path.join(label_dir, f"{stem}.txt")
            if not os.path.exists(seg_path):
                continue

            pts_path = os.path.join(points_dir, fname)
            try:
                pts = np.loadtxt(pts_path, dtype=np.float32)
                seg = np.loadtxt(seg_path, dtype=np.int64)
            except Exception as exc:
                print(f"[ShapeNet] Skipping corrupt sample {stem}: {exc}")
                continue
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if seg.ndim > 1:
                seg = seg.reshape(-1)
            if pts.shape[0] != seg.shape[0] or pts.shape[1] < 3:
                print(f"[ShapeNet] Skipping malformed sample {stem}")
                continue

            key = f"{synset}/{stem}"
            if test_ids is not None:
                split = "test" if key in test_ids else "train"
            else:
                split = "test" if rng.random() < 0.17 else "train"

            pid = _map_raw_part_labels(seg, cat_id)
            seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
            pts_sample, pid_sample = _sample_points(
                pts[:, :3], pid, n_points, seed=seed
            )
            samples.append((split, pts_sample, cat_id, pid_sample))

    return samples


def _write_shapenet_h5(samples, split: str, out_path: str):
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py")

    split_samples = [s for s in samples if s[0] == split]
    if not split_samples:
        raise RuntimeError(f"No {split} samples found while preparing ShapeNetPart")

    data = np.stack([s[1] for s in split_samples]).astype(np.float32)
    label = np.array([[s[2]] for s in split_samples], dtype=np.int64)
    pid = np.stack([s[3] for s in split_samples]).astype(np.int64)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip")
        f.create_dataset("label", data=label, compression="gzip")
        f.create_dataset("pid", data=pid, compression="gzip")
    print(f"[ShapeNet] Wrote {out_path}: {len(split_samples)} samples")


def prepare_shapenet_raw(src_path: str, n_points: int = 2048):
    """
    Convert raw/Kaggle ShapeNetPart PartAnnotation layout to HDF5.

    The Kaggle dataset at majdouline20/shapenetpart-dataset exposes the raw
    PartAnnotation tree. This converter writes the Stanford-style HDF5 files
    expected by train_shapenet.py.
    """
    if not os.path.exists(src_path):
        print(f"[ShapeNet] Raw path not found: {src_path}")
        return False

    os.makedirs(DATA_ROOT, exist_ok=True)
    search_root = src_path
    if os.path.isfile(src_path):
        extract_root = os.path.join(DATA_ROOT, "_shapenet_raw_extract")
        print(f"[ShapeNet] Extracting raw archive: {src_path}")
        if not _extract_archive(src_path, extract_root):
            print("[ShapeNet] Could not extract raw archive.")
            return False
        search_root = extract_root

    raw_dir = _find_shapenet_raw_dir(search_root)
    if raw_dir is None:
        print("[ShapeNet] Could not find raw PartAnnotation category folders.")
        return False

    print(f"[ShapeNet] Converting raw PartAnnotation from {raw_dir}")
    samples = _collect_shapenet_raw_samples(raw_dir, n_points)
    if not samples:
        print("[ShapeNet] No usable raw ShapeNetPart samples found.")
        return False

    out_dir = os.path.join(DATA_ROOT, SHAPENET_DIR)
    os.makedirs(out_dir, exist_ok=True)
    _write_shapenet_h5(samples, "train", os.path.join(out_dir, "train0.h5"))
    _write_shapenet_h5(samples, "test", os.path.join(out_dir, "test0.h5"))

    with open(os.path.join(out_dir, "all_object_categories.txt"), "w") as f:
        for name in SHAPENET_CATEGORY_NAMES:
            f.write(f"{name}\n")
    print(f"[ShapeNet] Ready at {out_dir}")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  ScanObjectNN — HuggingFace mirror (no form needed!)
# ═══════════════════════════════════════════════════════════════════════

# This HuggingFace mirror contains the PB_T50_RS variant (hardest)
# and does NOT require filling out any license form.
SCANOBJ_HF_URL = (
    "https://huggingface.co/datasets/cminst/ScanObjectNN/resolve/main/"
    "scanobjectnn_PB_T50_RS_h5.zip"
)
# Google Drive fallback (OpenPoints preprocessed tar)
SCANOBJ_GDRIVE_ID = "1iM3mhMJ_N0x5pytcP831l3ZFwbLmbwzi"


def download_scanobjectnn():
    """
    ScanObjectNN PB_T50_RS — auto-download from HuggingFace mirror.
    No license form required for the HF mirror.
    Falls back to Google Drive if HF fails.

    Expected result:
        data/ScanObjectNN/main_split/
            training_objectdataset_augmentedrot_scale75.h5  (11,416 shapes)
            test_objectdataset_augmentedrot_scale75.h5      (2,882 shapes)
    """
    out_dir = os.path.join(DATA_ROOT, "ScanObjectNN", "main_split")
    train_file = os.path.join(out_dir,
                              "training_objectdataset_augmentedrot_scale75.h5")
    test_file = os.path.join(out_dir,
                             "test_objectdataset_augmentedrot_scale75.h5")

    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"[ScanObjectNN] Already present at {out_dir}")
        return True

    os.makedirs(out_dir, exist_ok=True)

    # ── Method 1: HuggingFace direct download ─────────────────────────
    print("[ScanObjectNN] Downloading from HuggingFace mirror ...")
    hf_zip = os.path.join(DATA_ROOT, "scanobjectnn_hf.zip")
    try:
        _download_with_progress(SCANOBJ_HF_URL, hf_zip)
        print("[ScanObjectNN] Extracting ...")
        with zipfile.ZipFile(hf_zip, 'r') as zf:
            zf.extractall(os.path.join(DATA_ROOT, "ScanObjectNN"))
        os.remove(hf_zip)

        # The HF zip may extract with a slightly different structure
        # Verify the files ended up in the right place
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("[ScanObjectNN] Done (from HuggingFace)")
            return True

        # Check if files extracted to a subfolder
        for root, dirs, files in os.walk(os.path.join(DATA_ROOT, "ScanObjectNN")):
            for f in files:
                if f == "training_objectdataset_augmentedrot_scale75.h5":
                    src = os.path.join(root, f)
                    if src != train_file:
                        os.makedirs(out_dir, exist_ok=True)
                        os.rename(src, train_file)
                if f == "test_objectdataset_augmentedrot_scale75.h5":
                    src = os.path.join(root, f)
                    if src != test_file:
                        os.makedirs(out_dir, exist_ok=True)
                        os.rename(src, test_file)

        if os.path.exists(train_file) and os.path.exists(test_file):
            print("[ScanObjectNN] Done (from HuggingFace, relocated)")
            return True
    except Exception as e:
        print(f"[ScanObjectNN] HuggingFace failed: {e}")
        if os.path.exists(hf_zip):
            os.remove(hf_zip)

    # ── Method 2: Google Drive via gdown ──────────────────────────────
    print("[ScanObjectNN] Trying Google Drive via gdown ...")
    try:
        import gdown
        tar_path = os.path.join(DATA_ROOT, "ScanObjectNN.tar")
        gdown.download(id=SCANOBJ_GDRIVE_ID, output=tar_path, quiet=False)

        print("[ScanObjectNN] Extracting ...")
        with tarfile.open(tar_path, 'r') as tf:
            tf.extractall(DATA_ROOT)
        os.remove(tar_path)

        if os.path.exists(train_file) and os.path.exists(test_file):
            print("[ScanObjectNN] Done (from Google Drive)")
            return True
    except Exception as e:
        print(f"[ScanObjectNN] gdown failed: {e}")

    # ── Method 3: Manual instructions ─────────────────────────────────
    print()
    print("=" * 60)
    print("  ScanObjectNN auto-download failed")
    print("=" * 60)
    print()
    print("  Option 1: Download manually from the official site")
    print("    1. Visit: https://hkust-vgd.github.io/scanobjectnn/")
    print("    2. Fill the license form to get the download link")
    print("    3. Download h5_files.zip")
    print("    4. Extract main_split/ to:")
    print(f"       {out_dir}/")
    print()
    print("  Option 2: Try the direct link")
    print(f"    wget {SCANOBJ_HF_URL}")
    print(f"    unzip scanobjectnn_PB_T50_RS_h5.zip -d {os.path.join(DATA_ROOT, 'ScanObjectNN')}")
    print()
    return False


# ═══════════════════════════════════════════════════════════════════════
#  S3DIS — Google Drive (OpenPoints preprocessed) or manual
# ═══════════════════════════════════════════════════════════════════════

S3DIS_GDRIVE_ID = "1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y"


def _s3dis_area_dirs_ready(out_dir: str):
    found = [
        d for d in os.listdir(out_dir)
        if d.startswith("Area_")
        and os.path.isdir(os.path.join(out_dir, d))
        and any(name.endswith(".npy") for name in os.listdir(os.path.join(out_dir, d)))
    ] if os.path.isdir(out_dir) else []
    return len(found) >= 6, found


def _s3dis_flat_ready(out_dir: str):
    raw_dir = os.path.join(out_dir, "raw")
    if not os.path.isdir(raw_dir):
        return False, []
    files = [f for f in os.listdir(raw_dir) if f.startswith("Area_") and f.endswith(".npy")]
    areas = sorted({f.split("_")[1] for f in files if len(f.split("_")) > 1})
    return len(areas) >= 6, files


def download_s3dis():
    """
    S3DIS preprocessed per-room .npy files.
    Each file: [N_points, 7] = x, y, z, r, g, b, semantic_label

    Primary: gdown from Google Drive (OpenPoints format)
    Fallback: Manual download from Stanford + preprocessing
    """
    out_dir = os.path.join(DATA_ROOT, "s3dis")
    ready, found_areas = _s3dis_area_dirs_ready(out_dir)
    if ready:
        print(f"[S3DIS] Already present at {out_dir} ({len(found_areas)} area folders)")
        return True
    flat_ready, flat_files = _s3dis_flat_ready(out_dir)
    if flat_ready:
        print(f"[S3DIS] Already present at {out_dir}/raw ({len(flat_files)} flat room files)")
        print("[S3DIS] Loader will use the flat raw/Area_*.npy layout directly.")
        return True

    os.makedirs(out_dir, exist_ok=True)

    # ── Method 1: gdown from Google Drive ─────────────────────────────
    print("[S3DIS] Downloading preprocessed data via gdown ...")
    try:
        import gdown
        zip_path = os.path.join(DATA_ROOT, "s3dis_processed.zip")
        if os.path.exists(zip_path):
            print(f"[S3DIS] Using existing archive: {zip_path}")
        else:
            gdown.download(id=S3DIS_GDRIVE_ID, output=zip_path, quiet=False)

        print("[S3DIS] Extracting (this may take a few minutes) ...")
        if not _extract_archive(zip_path, out_dir):
            raise RuntimeError("downloaded file is not a supported archive")
        os.remove(zip_path)

        # Verify structure — look for Area_* directories
        ready, found_areas = _s3dis_area_dirs_ready(out_dir)
        if ready:
            print(f"[S3DIS] Done: {len(found_areas)} areas")
            return True

        flat_ready, flat_files = _s3dis_flat_ready(out_dir)
        if flat_ready:
            print(f"[S3DIS] Done: {len(flat_files)} flat raw room files")
            print("[S3DIS] Loader will use data/s3dis/raw/Area_*.npy directly.")
            return True

        # If extracted into a subdirectory, relocate
        found_areas = []
        for item in os.listdir(out_dir):
            sub = os.path.join(out_dir, item)
            if os.path.isdir(sub) and item not in found_areas:
                for child in os.listdir(sub):
                    if child.startswith("Area_"):
                        os.rename(os.path.join(sub, child),
                                  os.path.join(out_dir, child))
                        found_areas.append(child)

        ready, found_areas = _s3dis_area_dirs_ready(out_dir)
        if ready:
            print(f"[S3DIS] Done: {len(found_areas)} areas (relocated)")
            return True
        flat_ready, flat_files = _s3dis_flat_ready(out_dir)
        if flat_ready:
            print(f"[S3DIS] Done: {len(flat_files)} flat raw room files")
            print("[S3DIS] Loader will use data/s3dis/raw/Area_*.npy directly.")
            return True

    except ImportError:
        print("[S3DIS] gdown not installed. Install with: pip install gdown")
    except Exception as e:
        print(f"[S3DIS] gdown failed: {e}")

    # ── Method 2: Manual instructions ─────────────────────────────────
    print()
    print("=" * 60)
    print("  S3DIS auto-download failed")
    print("=" * 60)
    print()
    print("  Option 1: Install gdown and retry")
    print("    pip install gdown")
    print("    python datasets/download.py --s3dis")
    print()
    print("  Option 2: Download raw S3DIS + preprocess")
    print("    1. Get Stanford3dDataset_v1.2_Aligned_Version.zip from:")
    print("       http://buildingparser.stanford.edu/dataset.html")
    print("    2. Extract it")
    print("    3. Run: python datasets/download.py --s3dis_preprocess /path/to/Stanford3dDataset_v1.2_Aligned_Version")
    print()
    print("  Option 3: Download OpenPoints preprocessed S3DIS")
    print(f"    gdown --id {S3DIS_GDRIVE_ID} -O data/s3dis_processed.zip")
    print(f"    python datasets/download.py --s3dis")
    print("    # The loader accepts either data/s3dis/Area_*/ or data/s3dis/raw/Area_*.npy")
    print()
    return False


def preprocess_s3dis_raw(raw_dir: str):
    """
    Preprocess raw Stanford S3DIS into per-room .npy files.

    Raw structure:
        Stanford3dDataset_v1.2_Aligned_Version/Area_N/room_name/Annotations/*.txt

    Output:
        data/s3dis/Area_N/room_name.npy  — [N_points, 7]: x,y,z,r,g,b,label
    """
    import numpy as np

    out_dir = os.path.join(DATA_ROOT, "s3dis")
    os.makedirs(out_dir, exist_ok=True)

    CLASS_MAP = {name: i for i, name in enumerate([
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window',
        'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter',
    ])}

    total_rooms = 0
    for area_idx in range(1, 7):
        area_name = f"Area_{area_idx}"
        area_raw = os.path.join(raw_dir, area_name)
        area_out = os.path.join(out_dir, area_name)
        os.makedirs(area_out, exist_ok=True)

        if not os.path.isdir(area_raw):
            print(f"[Preprocess] Skipping {area_name} (not found in {raw_dir})")
            continue

        rooms = sorted([d for d in os.listdir(area_raw)
                        if os.path.isdir(os.path.join(area_raw, d))])

        area_count = 0
        for room_name in rooms:
            anno_dir = os.path.join(area_raw, room_name, "Annotations")
            if not os.path.isdir(anno_dir):
                continue

            room_pts = []
            for anno_file in sorted(os.listdir(anno_dir)):
                if not anno_file.endswith('.txt'):
                    continue
                class_name = '_'.join(anno_file.split('_')[:-1])
                label = CLASS_MAP.get(class_name, CLASS_MAP['clutter'])

                fpath = os.path.join(anno_dir, anno_file)
                try:
                    pts = np.loadtxt(fpath)
                except Exception:
                    continue
                if pts.ndim == 1:
                    pts = pts.reshape(1, -1)
                if pts.shape[1] < 6:
                    continue

                labels_col = np.full((len(pts), 1), label, dtype=np.float32)
                room_pts.append(
                    np.concatenate([pts[:, :6].astype(np.float32), labels_col], axis=1)
                )

            if room_pts:
                room_data = np.concatenate(room_pts, axis=0)
                np.save(os.path.join(area_out, f"{room_name}.npy"), room_data)
                area_count += 1

        total_rooms += area_count
        print(f"[Preprocess] {area_name}: {area_count} rooms")

    print(f"[Preprocess] Done: {total_rooms} rooms total in {out_dir}")


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════

def _download_with_progress(url: str, dest: str):
    """Download with progress bar."""
    try:
        from tqdm import tqdm

        class _Hook(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with _Hook(unit='B', unit_scale=True, miniters=1,
                   desc=os.path.basename(dest)) as t:
            urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
    except ImportError:
        print(f"  Downloading {os.path.basename(dest)} (no progress bar) ...")
        urllib.request.urlretrieve(url, dest)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Download ASP-SNN datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python datasets/download.py --all
    python datasets/download.py --scanobj
    python datasets/download.py --shapenet_local /path/to/kaggle/shapenetpart.zip
    python datasets/download.py --shapenet_raw /path/to/PartAnnotation
    python datasets/download.py --s3dis_preprocess /data/Stanford3dDataset_v1.2_Aligned_Version
        """,
    )
    p.add_argument("--all", action="store_true",
                   help="Download all three datasets")
    p.add_argument("--shapenet", action="store_true",
                   help="Download ShapeNetPart HDF5")
    p.add_argument("--shapenet_local", type=str, default=None,
                   help="Prepare local/Kaggle ShapeNetPart HDF5 zip or folder")
    p.add_argument("--shapenet_raw", type=str, default=None,
                   help="Convert raw/Kaggle ShapeNetPart PartAnnotation to HDF5")
    p.add_argument("--scanobj", action="store_true",
                   help="Download ScanObjectNN PB_T50_RS")
    p.add_argument("--s3dis", action="store_true",
                   help="Download S3DIS preprocessed")
    p.add_argument("--s3dis_preprocess", type=str, default=None,
                   help="Preprocess raw S3DIS from Stanford directory")
    args = p.parse_args()

    if args.shapenet_local:
        prepare_shapenet_local(args.shapenet_local)
        return

    if args.shapenet_raw:
        prepare_shapenet_raw(args.shapenet_raw)
        return

    if args.s3dis_preprocess:
        import numpy as np
        preprocess_s3dis_raw(args.s3dis_preprocess)
        return

    results = {}

    if args.all or args.shapenet:
        results['ShapeNetPart'] = download_shapenet()

    if args.all or args.scanobj:
        results['ScanObjectNN'] = download_scanobjectnn()

    if args.all or args.s3dis:
        results['S3DIS'] = download_s3dis()

    if not any([args.all, args.shapenet, args.scanobj, args.s3dis]):
        p.print_help()
        return

    # Summary
    print()
    print("=" * 60)
    print("  Download Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "READY" if ok else "NEEDS ATTENTION"
        print(f"  {name:<15} {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
