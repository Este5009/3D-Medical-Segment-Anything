"""Lightweight 3D grouping refinement for a frozen one-query segmenter.

The module deliberately does not replace or alter the established decoder.  It
predicts a small, bounded residual correction to that decoder's logits at the
RS2-Net level-2 grid.  Consequently a zero correction exactly reproduces the
validated mixed-domain model.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def normalized_coordinate_grid(
    spatial_shape: Sequence[int], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return dataset-independent coordinates in ``[-1, 1]`` as ``[1,3,D,H,W]``."""
    axes = [torch.linspace(-1.0, 1.0, int(n), device=device, dtype=dtype) for n in spatial_shape]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((z, y, x), dim=0).unsqueeze(0)


def largest_component_3d(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 26-connected foreground component.

    This inference-only rule depends solely on the predicted binary mask.  It
    never sees the image, ground truth, subject identity, or validation metric.
    """
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return binary.astype(np.uint8)
    labels, count = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return (labels == int(sizes.argmax())).astype(np.uint8)


class QueryFiLMBlock(nn.Module):
    """Depthwise-separable 3D residual block conditioned by the object query."""

    def __init__(self, channels: int, query_dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.GroupNorm(4, channels)
        self.pointwise = nn.Conv3d(channels, channels, 1)
        self.film = nn.Linear(query_dim, channels * 2)

    def forward(self, x: torch.Tensor, query: torch.Tensor, use_query: bool = True) -> torch.Tensor:
        residual = x
        x = self.pointwise(F.silu(self.norm(self.depthwise(x))))
        if use_query:
            gamma, beta = self.film(query).chunk(2, dim=-1)
            x = x * (1.0 + gamma[..., None, None, None]) + beta[..., None, None, None]
        return F.silu(x + residual)


class QueryConditionedGroupingDecoder(nn.Module):
    """Correct frozen decoder logits using level-2 anatomy and one fixed query.

    Inputs
    ------
    level2:
        Frozen RS2-Net feature map ``[B,96,32,32,40]``.
    initial_logits:
        Frozen decoder mask logits, normally ``[B,1,64,64,80]``.
    query:
        The existing decoder query ``[1,1,32]``.  No second query is introduced.
    """

    def __init__(
        self,
        *,
        feature_channels: int = 16,
        hidden_channels: int = 24,
        query_dim: int = 32,
        use_coordinates: bool = True,
        max_correction: float = 2.0,
    ) -> None:
        super().__init__()
        self.use_coordinates = use_coordinates
        self.max_correction = float(max_correction)
        self.feature_projection = nn.Conv3d(96, feature_channels, 1)
        # Logit, probability, and Bernoulli uncertainty supply complementary
        # descriptions of the frozen prediction.
        scalar_channels = 3 + (3 if use_coordinates else 0)
        self.input_projection = nn.Conv3d(feature_channels + scalar_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [QueryFiLMBlock(hidden_channels, query_dim), QueryFiLMBlock(hidden_channels, query_dim)]
        )
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )
        self.correction_head = nn.Conv3d(hidden_channels, 1, 1)
        # Identity initialization is an important safety property.
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(
        self,
        level2: torch.Tensor,
        initial_logits: torch.Tensor,
        query: torch.Tensor,
        *,
        use_query: bool = True,
        output_size: Optional[Sequence[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_size = level2.shape[-3:]
        initial_low = F.interpolate(initial_logits, size=low_size, mode="trilinear", align_corners=False)
        probability = initial_low.sigmoid()
        uncertainty = 4.0 * probability * (1.0 - probability)
        inputs = [self.feature_projection(level2), initial_low.clamp(-12, 12), probability, uncertainty]
        if self.use_coordinates:
            coordinates = normalized_coordinate_grid(low_size, device=level2.device, dtype=level2.dtype)
            inputs.append(coordinates.expand(level2.shape[0], -1, -1, -1, -1))
        x = self.input_projection(torch.cat(inputs, dim=1))
        fixed_query = query.expand(level2.shape[0], -1, -1).squeeze(1)
        for block in self.blocks:
            x = block(x, fixed_query, use_query=use_query)
        x = x * self.context_gate(x)[..., None, None, None]
        correction_low = self.max_correction * torch.tanh(self.correction_head(x))
        correction = F.interpolate(
            correction_low, size=initial_logits.shape[-3:], mode="trilinear", align_corners=False
        )
        final_logits = initial_logits + correction
        if output_size is not None and tuple(final_logits.shape[-3:]) != tuple(output_size):
            final_logits = F.interpolate(final_logits, size=tuple(output_size), mode="trilinear", align_corners=False)
            correction = F.interpolate(correction, size=tuple(output_size), mode="trilinear", align_corners=False)
        return final_logits, correction


class FrozenBaselineWithGrouping(nn.Module):
    """Inference composition that freezes both established model components."""

    def __init__(
        self, baseline: nn.Module, grouping: QueryConditionedGroupingDecoder, *, use_query: bool = True
    ) -> None:
        super().__init__()
        self.baseline = baseline
        self.grouping = grouping
        self.use_query = use_query
        self.baseline.eval()
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def encode(self, volume: torch.Tensor):
        return self.baseline.encode(volume)

    def decode(self, features, output_size=None):
        with torch.no_grad():
            initial = self.baseline.decode(features, output_size=None)
        final, _ = self.grouping(
            features["level2"], initial, self.baseline.decoder.query.detach(),
            use_query=self.use_query, output_size=output_size
        )
        return final
