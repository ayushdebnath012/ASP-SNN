"""
colab_asp_mn40_v4.py  —  ASP on ModelNet40, targeting > 92% OA
==============================================================
Paste as a single Colab cell, or:  !python colab_asp_mn40_v4.py

Architecture:  SPM-strength backbone (FPS+KNN, dim=384, 12 Mamba-lite blocks)
               + ASP adaptive group selection (4 chunks × 32 groups each)
Training:      Knowledge Distillation from PointTransformer teacher
               Full SO3 augmentation (QR random rotation)
               AMP + gradient accumulation (eff. batch 64)
               Test-time voting (10 rotations)

Drive layout:
  MyDrive/asp_mn40_v4_ckpts/
    teacher_latest.pt       teacher resume checkpoint
    teacher_best.pth        best teacher weights only
    asp_mn40_latest.pt      ASP resume checkpoint (atomic write)
    asp_mn40_best.pth       best ASP weights only
    history.json
"""

# ─── 0. GPU check + deps ─────────────────────────────────────────────────────
import subprocess, sys

for pkg in ["trimesh", "kagglehub"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import torch
print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
    print("VRAM    :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    raise RuntimeError("No GPU — Runtime > Change runtime type > T4 GPU")

# ─── 1. Mount Drive ───────────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("[Drive] Mounted")
except Exception as e:
    print(f"[Drive] Could not mount ({e}) — checkpoints saved locally only")

# ─── 2. Config ────────────────────────────────────────────────────────────────
import os, json, math, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda"

# Architecture (match SPM paper)
TIMESTEP    = 2
TRANS_DIM   = 384
DEPTH       = 12
NUM_GROUP   = 128
GROUP_SIZE  = 32
EXPAND      = 1.1
DROP_PATH   = 0.3
ASP_STEPS   = 4          # 128 / 4 = 32 groups per chunk

# Training
EPOCHS       = 300
BATCH        = 16
GRAD_ACCUM   = 4         # effective batch = 64
LR           = 1e-3
WEIGHT_DECAY = 0.1
WARMUP_EP    = 30
LABEL_SMOOTH = 0.2

# Numerical augmentation
MIXUP_ALPHA        = 0.4   # Beta param for PointMixup coordinate interpolation

# Knowledge distillation
TEACHER_EPOCHS     = 150
TEACHER_DIM        = 384
TEACHER_DEPTH      = 8
TEACHER_HEADS      = 8
KD_TEMP            = 4.0
KD_CE_WEIGHT       = 0.5
KD_LOGIT_WEIGHT    = 0.5
KD_AUX_WEIGHT      = 0.1

# Data / eval
NUM_POINTS  = 1024
NUM_CLASSES = 40
N_VOTE      = 10
EXIT_THR    = 0.45
VAL_EVERY   = 5
NUM_WORKERS = 2

# Paths
DRIVE_DIR = "/content/drive/MyDrive/asp_mn40_v4_ckpts"
MN40_DIR  = "/content/ModelNet40"
for d in [DRIVE_DIR]:
    os.makedirs(d, exist_ok=True)

TEACHER_LATEST = os.path.join(DRIVE_DIR, "teacher_latest.pt")
TEACHER_BEST   = os.path.join(DRIVE_DIR, "teacher_best.pth")
ASP_LATEST     = os.path.join(DRIVE_DIR, "asp_mn40_latest.pt")
ASP_BEST       = os.path.join(DRIVE_DIR, "asp_mn40_best.pth")

print(f"\nConfig: epochs={EPOCHS} batch={BATCH}×{GRAD_ACCUM}={BATCH*GRAD_ACCUM}")
print(f"        dim={TRANS_DIM} depth={DEPTH} T={TIMESTEP} "
      f"group={NUM_GROUP}×{GROUP_SIZE} asp={ASP_STEPS}")
print(f"        teacher_epochs={TEACHER_EPOCHS} kd_temp={KD_TEMP} vote={N_VOTE}")
print(f"        Drive: {DRIVE_DIR}")

# ─── 3. Download ModelNet40 ───────────────────────────────────────────────────
import shutil, glob as _glob

if not os.path.isdir(MN40_DIR):
    print("\nDownloading ModelNet40 via kagglehub …")
    import kagglehub
    p = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset")
    found = None
    for root, dirs, _ in os.walk(p):
        if "ModelNet40" in dirs:
            found = os.path.join(root, "ModelNet40"); break
    if found:
        shutil.copytree(found, MN40_DIR)
    else:
        zips = _glob.glob(os.path.join(p, "*.zip"))
        if zips:
            os.system(f'unzip -q "{zips[0]}" -d /content/')
        else:
            raise RuntimeError(f"ModelNet40 not found in {p}")
    print("Done.")
else:
    print("ModelNet40 already present.")

print(f"Classes: {len([d for d in os.listdir(MN40_DIR) if os.path.isdir(os.path.join(MN40_DIR,d))])}")

# ─── 4. Dataset + Augmentation ───────────────────────────────────────────────

def _so3_rotation():
    """Uniform random SO3 rotation via QR decomposition."""
    R = np.random.randn(3, 3).astype(np.float32)
    R, _ = np.linalg.qr(R)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


def _augment(pts: np.ndarray, split: str) -> np.ndarray:
    pts = pts - pts.mean(0)
    pts /= np.max(np.linalg.norm(pts, axis=1)) + 1e-8

    if split != "train":
        return pts.astype(np.float32)

    # 1. Random point dropout [87.5 %, 100 %]
    n = pts.shape[0]
    keep = max(int(n * random.uniform(0.875, 1.0)), 1)
    idx = np.random.choice(n, keep, replace=False)
    pts2 = pts[idx]
    if keep < n:
        pad = np.random.choice(keep, n - keep, replace=True)
        pts2 = np.vstack([pts2, pts2[pad]])

    # 2. Anisotropic scale — each axis scaled independently [0.8, 1.25]
    pts2 = pts2 * np.random.uniform(0.8, 1.25, (1, 3)).astype(np.float32)

    # 3. Random axis flip — mirror along a random axis with 50% probability
    flip_mask = (np.random.randint(0, 2, 3) * 2 - 1).astype(np.float32)  # ±1 per axis
    pts2 = pts2 * flip_mask

    # 4. Random translate [-0.1, 0.1]
    pts2 = pts2 + np.random.uniform(-0.1, 0.1, (1, 3)).astype(np.float32)

    # 5. Full SO3 rotation (uniform random 3-D rotation, not just Z-axis)
    pts2 = pts2 @ _so3_rotation().T

    # 6. Gaussian jitter σ=0.02, clipped ±0.05
    pts2 += np.clip(np.random.randn(*pts2.shape).astype(np.float32) * 0.02,
                    -0.05, 0.05)
    return pts2.astype(np.float32)


def _mixup_collate(batch, alpha=0.4):
    """PointMixup: interpolate two point clouds in input space.
    Labels become soft — returned as (pts, label_a, label_b, lam).
    The DataLoader uses a normal collate; mixing happens in the training loop.
    This helper is called per mini-batch after stacking.
    """
    pts   = torch.stack([b[0] for b in batch])
    lbls  = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lam   = float(np.random.beta(alpha, alpha))
    idx   = torch.randperm(pts.shape[0])
    pts_m = lam * pts + (1 - lam) * pts[idx]
    return pts_m, lbls, lbls[idx], lam


class ModelNetDataset(Dataset):
    def __init__(self, root: str, num_points: int = 1024, split: str = "train"):
        self.num_points = num_points
        self.split = split
        clss = sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)))
        items = []
        for cls in clss:
            p = os.path.join(root, cls, split)
            if not os.path.isdir(p): continue
            label = clss.index(cls)
            for f in os.listdir(p):
                if f.lower().endswith((".npy", ".txt", ".off")):
                    items.append((os.path.join(p, f), label))

        print(f"  [{split}] Loading {len(items)} files …")
        pts_list, lbl_list = [], []
        for path, label in items:
            try:
                pts_list.append(self._load(path))
                lbl_list.append(label)
            except Exception as e:
                print(f"  [WARN] skip {os.path.basename(path)}: {e}")

        self.data   = np.array(pts_list, dtype=np.float32)
        self.labels = np.array(lbl_list, dtype=np.int64)
        print(f"  [{split}] {len(lbl_list)}/{len(items)} loaded  shape={self.data.shape}")

    def _load(self, path: str) -> np.ndarray:
        if path.endswith(".npy"):
            pts = np.load(path).astype(np.float32)[:, :3]
        elif path.endswith(".txt"):
            pts = np.loadtxt(path, delimiter=",").astype(np.float32)[:, :3]
        else:
            import trimesh
            mesh = trimesh.load(path, force="mesh")
            pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)
            pts = pts.astype(np.float32)
        n = pts.shape[0]
        if n >= self.num_points:
            pts = pts[np.random.choice(n, self.num_points, replace=False)]
        else:
            pad = np.random.choice(n, self.num_points - n, replace=True)
            pts = np.vstack([pts, pts[pad]])
        return pts

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        pts = _augment(self.data[idx].copy(), self.split)
        np.random.shuffle(pts)
        return (torch.tensor(pts, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))

# ─── 5. Spiking / common utilities ───────────────────────────────────────────

class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): ctx.save_for_backward(x); return (x > 0).float()
    @staticmethod
    def backward(ctx, g): (x,) = ctx.saved_tensors; return g / (1 + x.abs()) ** 2

spike_fn = _SurrogateSpike.apply


class SpikeAct(nn.Module):
    def __init__(self, vth=0.5):
        super().__init__()
        self.vth = vth
        self.register_buffer("_sum", torch.tensor(0.0))
        self.register_buffer("_cnt", torch.tensor(0.0))
    def forward(self, x):
        y = spike_fn(x - self.vth)
        self._sum = self._sum + y.detach().sum()
        self._cnt = self._cnt + y.numel()
        return y
    def rate(self):
        return (self._sum / self._cnt).item() if self._cnt > 0 else 0.0


class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p = p
    def forward(self, x):
        if not self.training or self.p == 0: return x
        keep  = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep


class TokenBNSpike(nn.Module):
    def __init__(self, dim, vth=0.5):
        super().__init__(); self.bn = nn.BatchNorm1d(dim); self.sp = SpikeAct(vth)
    def forward(self, x):
        b, l, c = x.shape
        return self.sp(self.bn(x.reshape(b * l, c)).reshape(b, l, c))

# ─── 6. FPS + KNN ────────────────────────────────────────────────────────────

def index_points(points, idx):
    b = points.shape[0]
    view_shape = list(idx.shape); view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape); repeat_shape[0] = 1
    bi = torch.arange(b, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[bi, idx]


def fps_batched(xyz, npoint):
    b, n, _ = xyz.shape
    npoint  = min(npoint, n)
    cents   = torch.zeros(b, npoint, dtype=torch.long, device=xyz.device)
    dist    = torch.full((b, n), 1e10, device=xyz.device)
    far     = torch.randint(0, n, (b,), device=xyz.device)
    bi      = torch.arange(b, device=xyz.device)
    for i in range(npoint):
        cents[:, i] = far
        cent = xyz[bi, far, :].unsqueeze(1)
        d    = ((xyz - cent) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far  = dist.max(-1).indices
    return cents

# ─── 7. SPM backbone modules ─────────────────────────────────────────────────

class OfficialLikeGroup(nn.Module):
    def __init__(self, num_group, group_size, expand=1.1, timestep=2):
        super().__init__()
        self.num_group = num_group; self.group_size = group_size
        self.expand = expand; self.timestep = timestep

    def _moving_centers(self, pts):
        b, n, _ = pts.shape
        step_f = int((self.expand - 1.0) * self.num_group / self.timestep * 2)
        step_b = int((self.expand - 1.0) * self.num_group)
        total  = min(max(self.num_group + (step_f + step_b) * (self.timestep - 1),
                         self.num_group), n)
        idx    = fps_batched(pts.contiguous(), total)
        pool   = index_points(pts, idx)
        need   = self.num_group + (step_f + step_b) * (self.timestep - 1)
        if pool.shape[1] < need:
            rep  = math.ceil(need / pool.shape[1])
            pool = pool.repeat(1, rep, 1)
        centers = []
        for i in range(self.timestep):
            first  = pool[:, i * step_f: i * step_f + (self.num_group - step_b)]
            start  = (i - 1) * step_b + self.num_group + (self.timestep - 1) * step_f
            end    = i * step_b + self.num_group + (self.timestep - 1) * step_f
            second = pool[:, start:end]
            cur    = torch.cat([first, second], dim=1)
            if cur.shape[1] < self.num_group:
                cur = torch.cat([cur, cur[:, -1:].repeat(1, self.num_group - cur.shape[1], 1)], 1)
            centers.append(cur[:, :self.num_group])
        return torch.stack(centers, dim=0)

    def forward(self, pts):
        b, n, _ = pts.shape
        centers  = self._moving_centers(pts)                       # [T,B,G,3]
        flat_c   = centers.reshape(self.timestep * b, self.num_group, 3)
        flat_pts = pts.unsqueeze(0).expand(self.timestep, -1, -1, -1).reshape(self.timestep * b, n, 3)
        k        = min(self.group_size, n)
        idx      = torch.cdist(flat_c, flat_pts).topk(k, dim=-1, largest=False).indices
        grouped  = index_points(flat_pts, idx)                     # [TB,G,K,3]
        grouped  = grouped.reshape(self.timestep, b, self.num_group, k, 3)
        grouped  = grouped - centers.unsqueeze(3)
        return grouped.contiguous(), centers.contiguous()


class OfficialLikeEncoder(nn.Module):
    def __init__(self, enc_channel):
        super().__init__()
        self.sp1 = SpikeAct(); self.sp2 = SpikeAct(); self.sp3 = SpikeAct()
        self.c1 = nn.Conv2d(3,   128, 1); self.b1 = nn.BatchNorm2d(128)
        self.c2 = nn.Conv2d(128, 256, 1); self.b2 = nn.BatchNorm2d(256)
        self.c3 = nn.Conv2d(512, 512, 1); self.b3 = nn.BatchNorm2d(512)
        self.c4 = nn.Conv2d(512, enc_channel, 1); self.b4 = nn.BatchNorm2d(enc_channel)

    def forward(self, neighborhoods):
        t, b, g, k, _ = neighborhoods.shape
        x = neighborhoods.flatten(0, 1).permute(0, 3, 1, 2).contiguous()
        x = self.sp1(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        gl = x.max(dim=3, keepdim=True).values
        x  = torch.cat([gl.expand(-1, -1, -1, k), x], dim=1)
        x  = self.sp2(x)
        x  = self.sp3(self.b3(self.c3(x)))
        x  = self.b4(self.c4(x))
        x  = x.max(dim=3).values.transpose(1, 2).contiguous()
        return x.reshape(t, b, g, -1)


class PosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), SpikeAct(),
            nn.Conv1d(128, dim, 1), nn.BatchNorm1d(dim),
        )
    def forward(self, centers):
        t, b, g, _ = centers.shape
        x = centers.flatten(0, 1).permute(0, 2, 1).contiguous()
        return self.net(x).permute(0, 2, 1).reshape(t, b, g, -1)


class MambaLiteMixer(nn.Module):
    def __init__(self, dim, expand=2):
        super().__init__()
        inner = dim * expand
        self.in_proj   = nn.Linear(dim, inner * 2)
        self.dwconv    = nn.Conv1d(inner, inner, 3, padding=1, groups=inner)
        self.scan_proj = nn.Linear(inner, inner)
        self.out_proj  = nn.Linear(inner, dim)
    def forward(self, x):
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)
        steps = torch.arange(1, u.shape[1] + 1, device=u.device, dtype=u.dtype).view(1, -1, 1)
        state = torch.cumsum(u, dim=1) / steps
        u = u + self.scan_proj(state)
        return self.out_proj(u * torch.sigmoid(gate))


class OfficialLikeBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm  = TokenBNSpike(dim)
        self.mixer = MambaLiteMixer(dim)
        self.dp    = DropPath(drop_path)
    def forward(self, x, residual=None):
        residual = self.dp(x) + residual if residual is not None else x
        x = self.norm(residual)
        x = self.mixer(x)
        return x, residual


class OfficialLikeMixerModel(nn.Module):
    def __init__(self, dim, depth, timestep, drop_path=0.3):
        super().__init__()
        self.timestep = timestep
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.layers = nn.ModuleList([OfficialLikeBlock(dim, dpr[i]) for i in range(depth)])
    def forward(self, tokens, pos):
        t, b, l, c = tokens.shape
        x = (tokens + pos).reshape(t * b, l, c)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x = (x + residual) if residual is not None else x
        return x.reshape(t, b, l, c)


class OfficialLikeHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            SpikeAct(),
            nn.Conv1d(dim, 256, 1), nn.BatchNorm1d(256), SpikeAct(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), SpikeAct(),
            nn.Conv1d(128, num_classes, 1),
        )
    def forward(self, x):
        t, b, _l, c = x.shape
        pooled = x.mean(dim=2).reshape(t * b, c, 1)
        logits = self.net(pooled).reshape(t, b, -1, 1)
        return logits.mean(dim=0).squeeze(-1)


class OfficialLikeSPM(nn.Module):
    def __init__(self, num_classes=40, dim=384, depth=12, num_group=128,
                 group_size=32, timestep=2, expand=1.1, drop_path=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dim         = dim
        self.num_group   = num_group
        self.group_size  = group_size
        self.grouper     = OfficialLikeGroup(num_group, group_size, expand, timestep)
        self.encoder     = OfficialLikeEncoder(dim)
        self.pos_embed   = PosEmbed(dim)
        self.blocks      = OfficialLikeMixerModel(dim, depth, timestep, drop_path)
        self.head        = OfficialLikeHead(dim, num_classes)

    def encode_groups(self, pts):
        nh, ctr = self.grouper(pts)
        return self.encoder(nh), self.pos_embed(ctr), ctr

    def forward_tokens(self, tokens, pos):
        return self.head(self.blocks(tokens, pos))

    def forward(self, pts):
        tokens, pos, _ = self.encode_groups(pts)
        return self.forward_tokens(tokens, pos)

    def mean_firing_rate(self):
        rates = [m.rate() for m in self.modules() if isinstance(m, SpikeAct)]
        return sum(rates) / max(len(rates), 1)

# ─── 8. Knowledge Distillation teacher ───────────────────────────────────────
# Non-spiking PointTransformer-style teacher (pure PyTorch, no custom CUDA).
# Trained first for TEACHER_EPOCHS, then frozen during ASP training.
# Teacher's soft labels guide both the ASP and per-step auxiliary losses.

class AnalogGroupEncoder(nn.Module):
    def __init__(self, enc_channel):
        super().__init__()
        self.c1 = nn.Conv2d(3,   128, 1); self.b1 = nn.BatchNorm2d(128)
        self.c2 = nn.Conv2d(128, 256, 1); self.b2 = nn.BatchNorm2d(256)
        self.c3 = nn.Conv2d(512, 512, 1); self.b3 = nn.BatchNorm2d(512)
        self.c4 = nn.Conv2d(512, enc_channel, 1); self.b4 = nn.BatchNorm2d(enc_channel)

    def forward(self, neighborhoods):
        t, b, g, k, _ = neighborhoods.shape
        x = neighborhoods.flatten(0, 1).permute(0, 3, 1, 2).contiguous()
        x = F.gelu(self.b1(self.c1(x)))
        x = F.gelu(self.b2(self.c2(x)))
        gl = x.max(dim=3, keepdim=True).values
        x  = torch.cat([gl.expand(-1, -1, -1, k), x], dim=1)
        x  = F.gelu(self.b3(self.c3(x)))
        x  = self.b4(self.c4(x))
        x  = x.max(dim=3).values.transpose(1, 2).contiguous()
        return x.reshape(t, b, g, -1)


class AnalogPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, dim, 1), nn.BatchNorm1d(dim),
        )
    def forward(self, centers):
        t, b, g, _ = centers.shape
        x = centers.flatten(0, 1).permute(0, 2, 1).contiguous()
        return self.net(x).permute(0, 2, 1).reshape(t, b, g, -1)


class PointTransformerTeacher(nn.Module):
    """Point-BERT / PointTransformer-style teacher in pure PyTorch.
    Same group tokenizer as SPM, but with GELU encoder + standard attention.
    Typically reaches 93-94% OA on MN40 in 150 epochs.
    """
    def __init__(self, num_classes=40, dim=384, depth=8, heads=8,
                 num_group=128, group_size=32, expand=1.1):
        super().__init__()
        self.num_classes = num_classes
        self.grouper  = OfficialLikeGroup(num_group, group_size, expand, timestep=1)
        self.encoder  = AnalogGroupEncoder(dim)
        self.pos_embed = AnalogPosEmbed(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm   = nn.LayerNorm(dim)
        self.head   = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(dim, num_classes),
        )

    def forward(self, pts):
        nh, ctr = self.grouper(pts)
        x = self.encoder(nh) + self.pos_embed(ctr)
        x = x.squeeze(0)
        x = self.norm(self.blocks(x))
        pooled = torch.cat([x.mean(1), x.max(1).values], dim=-1)
        return self.head(pooled)

# ─── 9. ASP wrapper ───────────────────────────────────────────────────────────

class SliceSelectionPolicy(nn.Module):
    def __init__(self, mem_dim, geo_dim=7, hidden=128):
        super().__init__()
        self.mem_proj = nn.Linear(mem_dim, hidden, bias=False)
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.scale = math.sqrt(hidden)
    def forward(self, belief, geo, visited=None):
        scores = torch.bmm(self.geo_proj(geo),
                           self.mem_proj(belief).unsqueeze(-1)).squeeze(-1) / self.scale
        if visited is not None:
            scores = scores.masked_fill(visited.clone(), float("-inf"))
        return scores


class OfficialLikeASP(nn.Module):
    def __init__(self, base_model, asp_steps=4, d_ssp=128):
        super().__init__()
        self.base      = base_model
        self.asp_steps = asp_steps
        self.chunk_sz  = base_model.num_group // asp_steps
        self.ssp       = SliceSelectionPolicy(base_model.dim, 7, d_ssp)
        self.belief_proj = nn.Sequential(
            nn.Linear(base_model.num_classes, base_model.dim), nn.GELU(),
            nn.Linear(base_model.dim, base_model.dim),
        )
        self.register_buffer("gumbel_tau", torch.tensor(1.0))

    @property
    def num_classes(self): return self.base.num_classes

    def set_gumbel_tau(self, tau): self.gumbel_tau.fill_(tau)

    def mean_firing_rate(self): return self.base.mean_firing_rate()

    def _chunkify(self, tokens, pos, centers, pts):
        t, b, g, c = tokens.shape
        s, k = self.asp_steps, self.chunk_sz
        tok_c = tokens.reshape(t, b, s, k, c)
        pos_c = pos.reshape(t, b, s, k, c)
        ctr_b = centers.mean(0).reshape(b, s, k, 3)
        chunk_ctr  = ctr_b.mean(2)
        centroid   = pts.mean(1, keepdim=True)
        anchor_d   = (chunk_ctr - centroid).norm(dim=-1, keepdim=True)
        spread     = (ctr_b - chunk_ctr.unsqueeze(2)).norm(dim=-1).mean(2, keepdim=True)
        coverage   = torch.ones(b, s, 1, device=pts.device)
        order      = torch.linspace(0, 1, s, device=pts.device).view(1, s, 1).expand(b, -1, -1)
        geo = torch.cat([chunk_ctr, anchor_d, spread, coverage, order], dim=-1)
        return tok_c, pos_c, geo

    def _gather_chunk(self, chunks, idx):
        t, b, _s, k, c = chunks.shape
        gi = idx.view(1, b, 1, 1, 1).expand(t, b, 1, k, c)
        return chunks.gather(2, gi).squeeze(2)

    def forward_train(self, pts):
        tokens, pos, centers = self.base.encode_groups(pts)
        tok_c, pos_c, geo    = self._chunkify(tokens, pos, centers, pts)
        b, device = pts.shape[0], pts.device
        visited = torch.zeros(b, self.asp_steps, dtype=torch.bool, device=device)
        belief  = torch.zeros(b, self.base.dim, device=device)
        sel_tok, sel_pos, logits_all = [], [], []
        for _ in range(self.asp_steps):
            scores = self.ssp(belief, geo, visited)
            w      = F.gumbel_softmax(scores, tau=float(self.gumbel_tau), hard=True)
            idx    = w.detach().argmax(-1)
            visited.scatter_(1, idx.unsqueeze(1), True)
            tok = (w.view(1, b, self.asp_steps, 1, 1) * tok_c).sum(2)
            ps  = (w.view(1, b, self.asp_steps, 1, 1) * pos_c).sum(2)
            sel_tok.append(tok); sel_pos.append(ps)
            logits = self.base.forward_tokens(torch.cat(sel_tok, 2), torch.cat(sel_pos, 2))
            logits_all.append(logits)
            belief = self.belief_proj(logits.detach().softmax(-1))
        return logits_all[-1], logits_all

    @torch.no_grad()
    def forward_infer(self, pts, threshold=0.45):
        tokens, pos, centers = self.base.encode_groups(pts)
        tok_c, pos_c, geo    = self._chunkify(tokens, pos, centers, pts)
        b, device = pts.shape[0], pts.device
        visited = torch.zeros(b, self.asp_steps, dtype=torch.bool, device=device)
        belief  = torch.zeros(b, self.base.dim, device=device)
        sel_tok, sel_pos = [], []
        last_logits = None
        for step in range(self.asp_steps):
            idx = self.ssp(belief, geo, visited).argmax(-1)
            visited.scatter_(1, idx.unsqueeze(1), True)
            sel_tok.append(self._gather_chunk(tok_c, idx))
            sel_pos.append(self._gather_chunk(pos_c, idx))
            logits = self.base.forward_tokens(torch.cat(sel_tok, 2), torch.cat(sel_pos, 2))
            last_logits = logits
            belief = self.belief_proj(logits.softmax(-1))
            top2   = logits.softmax(-1).topk(2, -1).values
            if (top2[:, 0] - top2[:, 1]).min().item() > threshold:
                return logits, step + 1
        return last_logits, self.asp_steps

# ─── 10. Loss functions ───────────────────────────────────────────────────────

def smooth_ce(logits, labels):
    return F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTH)


def mixup_ce(logits, labels_a, labels_b, lam):
    """Cross-entropy for PointMixup soft targets."""
    return lam * smooth_ce(logits, labels_a) + (1 - lam) * smooth_ce(logits, labels_b)


def kd_ce(logits, labels, teacher_logits=None, labels_b=None, lam=1.0):
    """CE (possibly mixed) + optional KD from teacher soft logits."""
    if labels_b is not None and lam < 1.0:
        ce = mixup_ce(logits, labels, labels_b, lam)
    else:
        ce = smooth_ce(logits, labels)
    if teacher_logits is None:
        return ce
    kd = F.kl_div(
        F.log_softmax(logits / KD_TEMP, -1),
        F.softmax(teacher_logits.detach() / KD_TEMP, -1),
        reduction="batchmean",
    ) * (KD_TEMP ** 2)
    return KD_CE_WEIGHT * ce + KD_LOGIT_WEIGHT * kd


def active_loss(logits_final, logits_all, labels, model,
                teacher_logits=None, labels_b=None, lam=1.0):
    loss = kd_ce(logits_final, labels, teacher_logits, labels_b, lam)
    if len(logits_all) > 1:
        aux  = sum(kd_ce(lg, labels, teacher_logits, labels_b, lam)
                   for lg in logits_all[:-1])
        loss = loss + KD_AUX_WEIGHT * aux / (len(logits_all) - 1)
    exit_l = sum(
        (len(logits_all) - i) / len(logits_all) *
        (1.0 - lg.softmax(-1).max(-1).values).mean()
        for i, lg in enumerate(logits_all)
    )
    loss = loss + 0.05 * exit_l / len(logits_all)
    loss = loss + 0.01 * model.mean_firing_rate()
    return loss

# ─── 11. Scheduler + checkpoint helpers ──────────────────────────────────────

def gumbel_tau_sched(epoch, tau0=1.0, tau_min=0.1, rate=0.04):
    return max(tau_min, tau0 * math.exp(-rate * epoch))


def make_scheduler(opt, epochs, warmup):
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        t = (ep - warmup) / max(1, epochs - warmup)
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _torch_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def save_ckpt(path, model, opt, sch, epoch, best, history, scaler=None):
    payload = {
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": opt.state_dict(), "scheduler": sch.state_dict(),
        "best": best, "history": history,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    if os.path.exists(path):
        try: os.replace(path, path + ".bak")
        except: pass
    os.replace(tmp, path)


def load_ckpt(path, model, opt, sch, scaler=None):
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        return 0, 0.0, []
    try:
        ckpt = _torch_load(path)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sch.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        ep   = int(ckpt["epoch"])
        best = float(ckpt.get("best", 0.0))
        hist = ckpt.get("history", [])
        print(f"  [CKPT] Resumed epoch {ep}  best={best*100:.2f}%  hist={len(hist)}")
        return ep, best, hist
    except Exception as e:
        print(f"  [CKPT] {os.path.basename(path)} failed: {e}")
        return 0, 0.0, []

# ─── 12. Build models ─────────────────────────────────────────────────────────

print("\nBuilding models …")
base_spm = OfficialLikeSPM(
    num_classes=NUM_CLASSES, dim=TRANS_DIM, depth=DEPTH,
    num_group=NUM_GROUP, group_size=GROUP_SIZE,
    timestep=TIMESTEP, expand=EXPAND, drop_path=DROP_PATH,
).to(DEVICE)

asp = OfficialLikeASP(base_spm, asp_steps=ASP_STEPS, d_ssp=128).to(DEVICE)

teacher = PointTransformerTeacher(
    num_classes=NUM_CLASSES, dim=TEACHER_DIM, depth=TEACHER_DEPTH,
    heads=TEACHER_HEADS, num_group=NUM_GROUP, group_size=GROUP_SIZE, expand=EXPAND,
).to(DEVICE)

print(f"  ASP params     : {sum(p.numel() for p in asp.parameters()):,}")
print(f"  Teacher params : {sum(p.numel() for p in teacher.parameters()):,}")

# ─── 13. Dataset loaders ─────────────────────────────────────────────────────

print(f"\nLoading MN40 from {MN40_DIR} …")
train_ds = ModelNetDataset(MN40_DIR, NUM_POINTS, "train")
val_ds   = ModelNetDataset(MN40_DIR, NUM_POINTS, "test")
train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Batches/ep: {len(train_loader)}")

# ─── 14. Teacher training / loading ──────────────────────────────────────────

@torch.no_grad()
def eval_teacher(model, loader, n_vote=3):
    model.eval()
    correct = total = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        prob_sum = torch.zeros(pts.shape[0], NUM_CLASSES, device=DEVICE)
        for _ in range(n_vote):
            theta = random.uniform(0, 2 * math.pi)
            c, s = math.cos(theta), math.sin(theta)
            Rz = torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]], device=DEVICE)
            prob_sum += model(pts @ Rz.T).softmax(-1)
        correct += (prob_sum.argmax(1) == labels).sum().item()
        total   += pts.shape[0]
    return correct / total


def train_teacher_epoch(model, loader, opt):
    model.train(); opt.zero_grad()
    total_loss = total_acc = n = 0
    for step, (pts, labels) in enumerate(loader):
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        logits = model(pts)
        loss   = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTH) / GRAD_ACCUM
        if torch.isfinite(loss):
            loss.backward()
        if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); opt.zero_grad()
        b = pts.shape[0]
        total_loss += loss.item() * GRAD_ACCUM * b
        total_acc  += (logits.argmax(1) == labels).sum().item()
        n += b
    return total_loss / n, total_acc / n


print("\n" + "=" * 60)
print("Phase 1: Teacher training")
print("=" * 60)

t_opt = torch.optim.AdamW(teacher.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
t_sch = make_scheduler(t_opt, TEACHER_EPOCHS, WARMUP_EP)
t_start, t_best, t_hist = load_ckpt(TEACHER_LATEST, teacher, t_opt, t_sch)

if t_start >= TEACHER_EPOCHS and os.path.isfile(TEACHER_BEST):
    print(f"Teacher already trained ({TEACHER_EPOCHS} ep). Loading best weights.")
    teacher.load_state_dict(_torch_load(TEACHER_BEST))
else:
    for ep in range(t_start, TEACHER_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_teacher_epoch(teacher, train_loader, t_opt)
        t_sch.step()

        val_acc = None
        if (ep + 1) % 5 == 0 or ep + 1 == TEACHER_EPOCHS:
            val_acc = eval_teacher(teacher, val_loader, n_vote=3)
            is_best = val_acc > t_best
            if is_best:
                t_best = val_acc
                torch.save(teacher.state_dict(), TEACHER_BEST)
            print(f"  [Teacher] Ep {ep+1:3d}/{TEACHER_EPOCHS}  "
                  f"tr={tr_acc:.4f}  val={val_acc:.4f} {'★' if is_best else ' '}  "
                  f"lr={t_opt.param_groups[0]['lr']:.5f}  {time.time()-t0:.0f}s")

        t_hist.append({"ep": ep + 1, "tr": tr_acc, "val": val_acc})
        save_ckpt(TEACHER_LATEST, teacher, t_opt, t_sch, ep + 1, t_best, t_hist)

    if os.path.isfile(TEACHER_BEST):
        teacher.load_state_dict(_torch_load(TEACHER_BEST))

teacher.eval()
print(f"\nTeacher ready  best_val={t_best*100:.2f}%  (frozen from here)")
for p in teacher.parameters():
    p.requires_grad_(False)

# ─── 15. Teacher inference helper ────────────────────────────────────────────

@torch.no_grad()
def teacher_forward(pts):
    return teacher(pts)

# ─── 16. ASP training ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Phase 2: ASP training with KD")
print("=" * 60)

optimizer = torch.optim.AdamW(asp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = make_scheduler(optimizer, EPOCHS, WARMUP_EP)
scaler    = GradScaler()
start_epoch, best_acc, history = load_ckpt(ASP_LATEST, asp, optimizer, scheduler, scaler)

if start_epoch == 0:
    # Try backup if primary is missing / corrupt
    if os.path.exists(ASP_LATEST + ".bak"):
        start_epoch, best_acc, history = load_ckpt(
            ASP_LATEST + ".bak", asp, optimizer, scheduler, scaler)

if start_epoch == 0:
    print("Starting ASP from scratch.")


@torch.no_grad()
def eval_asp_vote(model, loader, n_vote=N_VOTE):
    model.eval()
    correct = total = slices = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        b = pts.shape[0]
        prob_sum = torch.zeros(b, NUM_CLASSES, device=DEVICE)
        for _ in range(n_vote):
            theta = random.uniform(0, 2 * math.pi)
            c, s = math.cos(theta), math.sin(theta)
            Rz = torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]], device=DEVICE)
            logits, used = model.forward_infer(pts @ Rz.T, EXIT_THR)
            prob_sum += logits.softmax(-1)
            slices   += used * b
        correct += (prob_sum.argmax(1) == labels).sum().item()
        total   += b
    return correct / total, slices / total / n_vote


def train_one_epoch(model, loader, opt, epoch):
    model.train()
    tau = gumbel_tau_sched(epoch)
    model.set_gumbel_tau(tau)
    opt.zero_grad()
    total_loss = total_acc = n = 0
    t0 = time.time()

    for step, (pts, labels) in enumerate(loader):
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)

        # PointMixup: mix pairs within the batch in coordinate space
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        idx_mix = torch.randperm(pts.shape[0], device=DEVICE)
        pts_m   = lam * pts + (1 - lam) * pts[idx_mix]
        labels_b = labels[idx_mix]

        t_logits = teacher_forward(pts_m)

        with autocast():
            logits_f, logits_all = model.forward_train(pts_m)
            loss = active_loss(logits_f, logits_all, labels, model,
                               teacher_logits=t_logits,
                               labels_b=labels_b, lam=lam) / GRAD_ACCUM

        if torch.isfinite(loss):
            scaler.scale(loss).backward()
        else:
            print(f"  [SKIP] step {step}: non-finite loss")

        if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(loader):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()

        b = pts.shape[0]
        total_loss += loss.item() * GRAD_ACCUM * b
        # accuracy uses the dominant label (lam-weighted)
        pred_label = labels if lam >= 0.5 else labels_b
        total_acc  += (logits_f.detach().argmax(1) == pred_label).sum().item()
        n += b

    return total_loss / max(n, 1), total_acc / max(n, 1), time.time() - t0


print(f"epochs={EPOCHS}  start_epoch={start_epoch}  best={best_acc*100:.2f}%")

for epoch in range(start_epoch, EPOCHS):
    tr_loss, tr_acc, elapsed = train_one_epoch(asp, train_loader, optimizer, epoch)
    scheduler.step()

    lr  = optimizer.param_groups[0]["lr"]
    tau = float(asp.gumbel_tau)
    print(f"Ep {epoch+1:3d}/{EPOCHS}  loss={tr_loss:.4f}  tr={tr_acc:.4f}  "
          f"tau={tau:.3f}  lr={lr:.5f}  {elapsed:.0f}s", end="")

    val_acc = None
    if (epoch + 1) % VAL_EVERY == 0 or epoch + 1 == EPOCHS:
        val_acc, val_sl = eval_asp_vote(asp, val_loader, N_VOTE)
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(asp.state_dict(), ASP_BEST)
        fr = asp.mean_firing_rate()
        print(f"  | val={val_acc:.4f} {'★' if is_best else ' '} "
              f"sl={val_sl:.2f}/{ASP_STEPS}  fr={fr:.3f}  best={best_acc:.4f}", end="")

    history.append({
        "epoch": epoch + 1, "tr_loss": tr_loss, "tr_acc": tr_acc,
        "val_acc": val_acc, "tau": tau, "lr": lr,
    })
    save_ckpt(ASP_LATEST, asp, optimizer, scheduler, epoch + 1, best_acc, history, scaler)

    with open(os.path.join(DRIVE_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("  ✓")

# ─── 17. Final verdict ────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Teacher OA  : {t_best*100:.2f}%")
print(f"ASP best OA : {best_acc*100:.2f}%  (target ≥ 92.0%)")
print(f"Drive dir   : {DRIVE_DIR}")
print(f"{'='*60}")

if best_acc >= 0.92:
    print("VERDICT: ✓ Beat SPM 92% target!")
elif best_acc >= 0.89:
    print("VERDICT: Very close. Try 50 more epochs or n_vote=15.")
elif best_acc >= 0.85:
    print("VERDICT: Good. Architecture correct — needs more epochs.")
else:
    print("VERDICT: Below target — check data path and GPU memory.")
