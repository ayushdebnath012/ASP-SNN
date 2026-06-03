"""ASP segmentor for ShapeNetPart and S3DIS training scripts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ssp import SSP


def _head_count(hidden_dim: int, requested: int) -> int:
    requested = max(1, int(requested))
    return requested if hidden_dim % requested == 0 else 1


class ASPSegmentor(nn.Module):
    """Slice encoder with SSP-guided context and per-point logits."""

    def __init__(self, cfg):
        super().__init__()
        in_channels = int(getattr(cfg, "in_channels", 6))
        hidden_dim = int(getattr(cfg, "hidden_dim", getattr(cfg, "feat_dim", 512)))
        geo_dim = int(getattr(cfg, "geo_dim", 8))
        num_classes = int(getattr(cfg, "num_classes", 50))
        heads = _head_count(hidden_dim, getattr(cfg, "transformer_heads", 4))

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_category = bool(getattr(cfg, "use_category", False))
        self.T = int(getattr(cfg, "T", 6))
        self.register_buffer("gumbel_tau", torch.tensor(1.0))

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pos_proj = nn.Linear(geo_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=int(getattr(cfg, "transformer_ffn_dim", hidden_dim * 2)),
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.slice_transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.ssp = SSP(
            belief_dim=hidden_dim,
            geo_dim=geo_dim,
            d_ssp=int(getattr(cfg, "d_ssp", 128)),
        )
        self.temporal = nn.GRUCell(hidden_dim, hidden_dim)

        num_categories = int(getattr(cfg, "num_categories", 0))
        self.category_embed = (
            nn.Embedding(max(num_categories, 1), hidden_dim)
            if self.use_category
            else None
        )
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def _encode_slices(self, slices: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        bsz, n_slices, pts_per_slice, channels = slices.shape
        x = slices.reshape(bsz * n_slices * pts_per_slice, channels)
        x = self.feature_extractor(x)
        x = x.reshape(bsz, n_slices, pts_per_slice, self.hidden_dim).mean(dim=2)
        x = x + self.pos_proj(geo)
        return self.slice_transformer(x)

    def _active_context(
        self,
        slice_feats: torch.Tensor,
        geo: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        bsz, n_slices, _ = slice_feats.shape
        steps = min(n_slices, max(1, int(getattr(self, "T", n_slices))))
        device = slice_feats.device
        belief = torch.zeros(bsz, self.hidden_dim, device=device)
        visited = torch.zeros(bsz, n_slices, dtype=torch.bool, device=device)
        states = []

        for _ in range(steps):
            scores = self.ssp(belief, geo, visited.clone())
            if training:
                weights = F.gumbel_softmax(
                    scores, tau=float(self.gumbel_tau.item()), hard=True, dim=-1
                )
            else:
                idx = scores.argmax(dim=-1)
                weights = F.one_hot(idx, num_classes=n_slices).float()
            selected = weights.argmax(dim=-1)
            visited.scatter_(1, selected.unsqueeze(1), True)
            chosen = (weights.unsqueeze(-1) * slice_feats).sum(dim=1)
            belief = self.temporal(chosen, belief)
            states.append(belief)

        return belief, states

    def forward(
        self,
        slices: torch.Tensor,
        geo: torch.Tensor,
        sid_arr: torch.Tensor,
        cat_ids: torch.Tensor,
        pts_features: torch.Tensor,
        training: bool = True,
    ):
        slice_feats = self._encode_slices(slices, geo)
        global_ctx, states = self._active_context(slice_feats, geo, training)

        bsz, n_points = sid_arr.shape
        idx = sid_arr.clamp(0, slice_feats.size(1) - 1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        point_ctx = slice_feats.gather(1, gather_idx)

        global_ctx = global_ctx.unsqueeze(1).expand(-1, n_points, -1)
        context = point_ctx + global_ctx
        if self.category_embed is not None:
            cat_ctx = self.category_embed(cat_ids.long().clamp_min(0))
            context = context + cat_ctx.unsqueeze(1)

        logits = self.point_head(torch.cat([point_ctx, context], dim=-1))
        return logits, states
