"""Minimal one-query 3D mask decoder for frozen RS2-Net features."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class OneQueryMaskDecoder(nn.Module):
    """Turn one learned object query into one volumetric mask.

    ``level4`` supplies compact semantic tokens to cross-attention. ``level1``
    supplies the higher-resolution voxel grid used for mask prediction. Both are
    projected to ``embedding_dim`` before their dot product.
    """

    def __init__(self, embedding_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.normal_(self.query, std=0.02)

        self.semantic_projection = nn.Conv3d(384, embedding_dim, kernel_size=1)
        self.voxel_projection = nn.Conv3d(48, embedding_dim, kernel_size=1)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.query_norm1 = nn.LayerNorm(embedding_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.query_norm2 = nn.LayerNorm(embedding_dim)
        self.mask_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.mask_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        output_size: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        semantic = self.semantic_projection(features["level4"])
        batch, channels, depth, height, width = semantic.shape
        semantic_tokens = semantic.flatten(2).transpose(1, 2)  # [B, D4*H4*W4, E]

        query = self.query.expand(batch, -1, -1)  # exactly one query: [B, 1, E]
        attended, _ = self.cross_attention(query, semantic_tokens, semantic_tokens, need_weights=False)
        query = self.query_norm1(query + attended)
        query = self.query_norm2(query + self.query_ffn(query))
        mask_embedding = self.mask_embedding(query).squeeze(1)  # [B, E]

        voxel_features = self.voxel_projection(features["level1"])  # [B, E, D1, H1, W1]
        logits = torch.einsum("bc,bcdhw->bdhw", mask_embedding, voxel_features)
        logits = logits.unsqueeze(1) + self.mask_bias.view(1, 1, 1, 1, 1)
        if output_size is not None and tuple(logits.shape[-3:]) != tuple(output_size):
            logits = F.interpolate(logits, size=tuple(output_size), mode="trilinear", align_corners=False)
        return logits


class QueryUpdateBlock(nn.Module):
    """Update one query from one scale using cross-attention and a small FFN."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, query: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        tokens = feature.flatten(2).transpose(1, 2)  # [B, D*H*W, E]
        attended, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
        query = self.norm1(query + attended)
        return self.norm2(query + self.ffn(query))


class MultiScaleOneQueryMaskDecoder(nn.Module):
    """One-query decoder with a compact four-level 3D feature pyramid.

    All RS2 scales level4, level3, level2, and level1 are projected to the same
    channel width. A top-down path adds coarse semantic context into progressively
    finer grids. The same single learned query then attends from coarse to fine.
    The final query is compared with the fused level1 grid to retain boundaries.
    """

    CHANNELS = {"level1": 48, "level2": 96, "level3": 192, "level4": 384}
    COARSE_TO_FINE = ("level4", "level3", "level2", "level1")

    def __init__(self, embedding_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.normal_(self.query, std=0.02)
        self.projections = nn.ModuleDict({
            name: nn.Conv3d(channels, embedding_dim, kernel_size=1)
            for name, channels in self.CHANNELS.items()
        })
        # One 3x3 refinement per fusion step removes interpolation artifacts while
        # remaining far smaller than the frozen RS2-Net decoder.
        self.refinements = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv3d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
                nn.InstanceNorm3d(embedding_dim, affine=True),
                nn.GELU(),
            )
            for name in ("level3", "level2", "level1")
        })
        self.query_updates = nn.ModuleDict({
            name: QueryUpdateBlock(embedding_dim, num_heads) for name in self.COARSE_TO_FINE
        })
        self.mask_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.mask_refinement = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.mask_bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: Mapping[str, torch.Tensor], output_size=None) -> torch.Tensor:
        projected = {name: self.projections[name](features[name]) for name in self.CHANNELS}

        # Coarse-to-fine additive FPN. Each output retains its native grid size.
        fused = {"level4": projected["level4"]}
        previous = fused["level4"]
        for name in ("level3", "level2", "level1"):
            previous = F.interpolate(previous, size=projected[name].shape[-3:], mode="trilinear", align_corners=False)
            previous = self.refinements[name](projected[name] + previous)
            fused[name] = previous

        batch = features["level1"].shape[0]
        query = self.query.expand(batch, -1, -1)  # always [B, 1, E]
        for name in self.COARSE_TO_FINE:
            query = self.query_updates[name](query, fused[name])

        mask_embedding = self.mask_embedding(query).squeeze(1)
        voxel_features = self.mask_refinement(fused["level1"])
        logits = torch.einsum("bc,bcdhw->bdhw", mask_embedding, voxel_features).unsqueeze(1)
        logits = logits + self.mask_bias.view(1, 1, 1, 1, 1)
        if output_size is not None and tuple(logits.shape[-3:]) != tuple(output_size):
            logits = F.interpolate(logits, size=tuple(output_size), mode="trilinear", align_corners=False)
        return logits


class MultiScaleAttentionOneQueryMaskDecoder(nn.Module):
    """Ablation: query attends to every scale, but voxel features are not fused."""

    CHANNELS = MultiScaleOneQueryMaskDecoder.CHANNELS
    COARSE_TO_FINE = MultiScaleOneQueryMaskDecoder.COARSE_TO_FINE

    def __init__(self, embedding_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.normal_(self.query, std=0.02)
        self.projections = nn.ModuleDict({
            name: nn.Conv3d(channels, embedding_dim, 1) for name, channels in self.CHANNELS.items()
        })
        self.query_updates = nn.ModuleDict({
            name: QueryUpdateBlock(embedding_dim, num_heads) for name in self.COARSE_TO_FINE
        })
        self.mask_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.mask_bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: Mapping[str, torch.Tensor], output_size=None) -> torch.Tensor:
        projected = {name: self.projections[name](features[name]) for name in self.CHANNELS}
        query = self.query.expand(features["level1"].shape[0], -1, -1)
        for name in self.COARSE_TO_FINE:
            query = self.query_updates[name](query, projected[name])
        mask_embedding = self.mask_embedding(query).squeeze(1)
        logits = torch.einsum("bc,bcdhw->bdhw", mask_embedding, projected["level1"]).unsqueeze(1)
        logits = logits + self.mask_bias.view(1, 1, 1, 1, 1)
        if output_size is not None and tuple(logits.shape[-3:]) != tuple(output_size):
            logits = F.interpolate(logits, size=tuple(output_size), mode="trilinear", align_corners=False)
        return logits


class FrozenEncoderQueryModel(nn.Module):
    """Compose an immutable encoder with the trainable one-query decoder."""

    def __init__(self, encoder: nn.Module, decoder: OneQueryMaskDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Calling model.train() must never enable stochastic encoder behavior.
        self.encoder.eval()
        return self

    def forward(self, volume: torch.Tensor, output_size: Optional[Sequence[int]] = None) -> torch.Tensor:
        features = self.encode(volume)
        return self.decode(features, output_size)

    def encode(self, volume: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the frozen encoder without building an autograd graph."""
        with torch.no_grad():
            features: Dict[str, torch.Tensor] = self.encoder(volume)
        return features

    def decode(self, features: Mapping[str, torch.Tensor], output_size=None) -> torch.Tensor:
        """Run only trainable components; useful for a tiny in-memory feature set."""
        return self.decoder(features, output_size=output_size)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5):
    """Equal-weight soft Dice and voxelwise BCE-with-logits loss."""
    target = target.float()
    probabilities = logits.sigmoid()
    reduce_dims = tuple(range(1, logits.ndim))
    intersection = (probabilities * target).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice_loss = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()
    bce_loss = F.binary_cross_entropy_with_logits(logits, target)
    return dice_loss + bce_loss, {"dice_loss": dice_loss.detach(), "bce_loss": bce_loss.detach()}


@torch.no_grad()
def volumetric_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    prediction = logits.sigmoid() >= threshold
    target_binary = target >= 0.5
    intersection = (prediction & target_binary).sum().float()
    denominator = prediction.sum().float() + target_binary.sum().float()
    return float(((2.0 * intersection + 1e-5) / (denominator + 1e-5)).cpu())
