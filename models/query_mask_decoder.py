"""Minimal one-query 3D mask decoder for frozen RS2-Net features."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetrUpBlock


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


class FullResolutionLevel0OneQueryMaskDecoder(nn.Module):
    """Controlled level0 extension of :class:`MultiScaleOneQueryMaskDecoder`.

    The level4→level1 implementation is intentionally identical. One additional
    lateral projection, additive top-down fusion/refinement, and query update
    operate on full-grid ``level0``. The mask dot product is formed directly on
    that `[D,H,W]` grid; interpolation is only a defensive fallback for callers
    requesting a genuinely different geometry.
    """

    CHANNELS = {"level0": 48, **MultiScaleOneQueryMaskDecoder.CHANNELS}
    COARSE_TO_FINE = ("level4", "level3", "level2", "level1", "level0")

    def __init__(self, embedding_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.normal_(self.query, std=0.02)
        self.projections = nn.ModuleDict({
            name: nn.Conv3d(channels, embedding_dim, kernel_size=1)
            for name, channels in MultiScaleOneQueryMaskDecoder.CHANNELS.items()
        })
        self.level0_width = 8
        self.level0_projection = nn.Conv3d(48, self.level0_width, 1)
        self.refinements = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv3d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
                nn.InstanceNorm3d(embedding_dim, affine=True),
                nn.GELU(),
            ) for name in ("level3", "level2", "level1")
        })
        # A dense 32x32x3x3x3 convolution over 2.6M voxels is unnecessary for
        # the one added lateral stage. Depthwise spatial refinement followed by
        # pointwise channel mixing is the minimum full-grid analogue.
        self.level1_to_level0 = nn.Conv3d(embedding_dim, self.level0_width, 1)
        self.level0_refinement = nn.Sequential(
            nn.Conv3d(self.level0_width, self.level0_width, 3, padding=1,
                      groups=self.level0_width),
            nn.Conv3d(self.level0_width, self.level0_width, 1),
            nn.InstanceNorm3d(self.level0_width, affine=True), nn.GELU())
        self.query_updates = nn.ModuleDict({
            name: QueryUpdateBlock(embedding_dim, num_heads)
            for name in MultiScaleOneQueryMaskDecoder.COARSE_TO_FINE
        })
        self.level0_query_projection = nn.Conv3d(
            self.level0_width, embedding_dim, 1)
        self.query_updates["level0"] = QueryUpdateBlock(embedding_dim, num_heads)
        self.mask_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim))
        # Preserve the baseline mask refinement at level1 exactly. The added
        # lightweight module injects projected level0 after its 2x upsampling.
        self.mask_refinement = nn.Conv3d(
            embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.level0_mask_embedding = nn.Linear(embedding_dim, self.level0_width)
        # Zero makes the untrained architecture exactly reproduce the corrected
        # half-resolution baseline. Training can then introduce only a learned
        # full-grid residual, avoiding a random level0 branch destroying the
        # transferred solution on epoch one.
        self.level0_residual_scale = nn.Parameter(torch.zeros(1))
        self.mask_bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: Mapping[str, torch.Tensor], output_size=None):
        projected = {name: self.projections[name](features[name])
                     for name in MultiScaleOneQueryMaskDecoder.CHANNELS}
        fused = {"level4": projected["level4"]}
        previous = fused["level4"]
        for name in ("level3", "level2", "level1"):
            previous = F.interpolate(previous, size=projected[name].shape[-3:],
                                     mode="trilinear", align_corners=False)
            previous = self.refinements[name](projected[name] + previous)
            fused[name] = previous
        level0 = self.level0_projection(features["level0"])
        level1_up = F.interpolate(
            self.level1_to_level0(fused["level1"]), size=level0.shape[-3:],
            mode="trilinear", align_corners=False)
        fused["level0"] = self.level0_refinement(level0 + level1_up)
        query = self.query.expand(features["level0"].shape[0], -1, -1)
        for name in MultiScaleOneQueryMaskDecoder.COARSE_TO_FINE:
            query = self.query_updates[name](query, fused[name])
        baseline_query = query
        # The query is global, so its level0 attention tokens are 2x average-
        # pooled to bound memory. The mask branch itself remains full-grid.
        level0_query_feature = self.level0_query_projection(
            F.avg_pool3d(fused["level0"], kernel_size=2, stride=2))
        query = self.query_updates["level0"](query, level0_query_feature)
        baseline_embedding = self.mask_embedding(baseline_query).squeeze(1)
        mask_embedding = self.mask_embedding(query).squeeze(1)
        # The preserved baseline head supplies the initialization. Its coarse
        # logits are interpolated as a skip, not used as the final prediction:
        # the trainable level0 residual below is formed directly on full voxels.
        baseline_voxels = self.mask_refinement(fused["level1"])
        baseline_logits = torch.einsum(
            "bc,bcdhw->bdhw", baseline_embedding, baseline_voxels).unsqueeze(1)
        baseline_logits = baseline_logits + self.mask_bias.view(1,1,1,1,1)
        baseline_logits = F.interpolate(
            baseline_logits, size=level0.shape[-3:], mode="trilinear",
            align_corners=False)
        voxel_features = F.interpolate(
            self.level1_to_level0(baseline_voxels), size=level0.shape[-3:],
            mode="trilinear", align_corners=False)
        voxel_features = self.level0_refinement(voxel_features + level0)
        residual_embedding = self.level0_mask_embedding(mask_embedding)
        residual = torch.einsum("bc,bcdhw->bdhw", residual_embedding,
                                voxel_features).unsqueeze(1)
        logits = baseline_logits + self.level0_residual_scale * residual
        if output_size is not None and tuple(logits.shape[-3:]) != tuple(output_size):
            logits = F.interpolate(logits, size=tuple(output_size),
                                   mode="trilinear", align_corners=False)
        return logits


class TrueFullResolutionLevel0OneQueryMaskDecoder(nn.Module):
    """Genuine full-resolution level0 decoder.

    This differs from :class:`FullResolutionLevel0OneQueryMaskDecoder` in one
    deliberate, audited way: it contains **no** half-resolution mask-logit
    branch and **no** logit-space interpolation or residual skip. The final
    voxel/query dot product that forms the mask logits is evaluated exactly
    once, directly on a fused ``level0`` (full model-grid) feature field.
    Levels4→3→2→1 are only ever used to produce *features* that are fused
    into that field; they never independently form logits of their own.

    levels4→3→2→1 reuse the identical module names/shapes as
    :class:`MultiScaleOneQueryMaskDecoder` (``projections``, ``refinements``,
    ``query_updates.level4/level3/level2/level1``, ``mask_embedding``,
    ``mask_refinement``, ``mask_bias``, ``query``) so a converged corrected-
    label checkpoint's decoder weights transfer into this architecture
    unchanged and un-renamed.

    The level0 stage is new and deliberately narrow: features are projected
    to ``level0_width`` channels (much smaller than ``embedding_dim``) and
    fused with one depthwise 3x3x3 + pointwise 1x1x1 convolution pair, so its
    cost scales linearly with ``level0_width`` rather than quadratically.
    Only the level0 *query-attention* tokens are 2x average-pooled (a memory
    control for the attention step alone); the mask field itself is never
    pooled, interpolated, or otherwise reduced before the final dot product.
    """

    CHANNELS = MultiScaleOneQueryMaskDecoder.CHANNELS
    COARSE_TO_FINE = MultiScaleOneQueryMaskDecoder.COARSE_TO_FINE

    def __init__(self, embedding_dim: int = 32, num_heads: int = 4, level0_width: int = 16) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.level0_width = level0_width

        # --- Identical to MultiScaleOneQueryMaskDecoder: same names/shapes so
        # a corrected-label checkpoint's decoder_state_dict transfers exactly. ---
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.normal_(self.query, std=0.02)
        self.projections = nn.ModuleDict({
            name: nn.Conv3d(channels, embedding_dim, kernel_size=1)
            for name, channels in self.CHANNELS.items()
        })
        self.refinements = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv3d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
                nn.InstanceNorm3d(embedding_dim, affine=True),
                nn.GELU(),
            ) for name in ("level3", "level2", "level1")
        })
        self.query_updates = nn.ModuleDict({
            name: QueryUpdateBlock(embedding_dim, num_heads) for name in self.COARSE_TO_FINE
        })
        self.mask_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim))
        # Reused as a level1 feature refinement stage that feeds level0 fusion
        # below -- not as a stand-alone mask head. It never forms logits here.
        self.mask_refinement = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.mask_bias = nn.Parameter(torch.zeros(1))

        # --- New: genuine full-grid level0 fusion. All-new parameters. ---
        self.level0_projection = nn.Conv3d(48, level0_width, 1)
        self.level1_to_level0 = nn.Conv3d(embedding_dim, level0_width, 1)
        self.level0_refinement = nn.Sequential(
            nn.Conv3d(level0_width, level0_width, 3, padding=1, groups=level0_width),
            nn.Conv3d(level0_width, level0_width, 1),
            nn.InstanceNorm3d(level0_width, affine=True), nn.GELU())
        self.query_updates["level0"] = QueryUpdateBlock(embedding_dim, num_heads)
        # Global query attention over level0 pools 2x for memory only; the
        # mask field computed below is never pooled.
        self.level0_query_projection = nn.Conv3d(level0_width, embedding_dim, 1)
        # Projects the (transferred) mask embedding down into level0's native
        # width for the final dot product. This is the only new head param.
        self.level0_embedding_projection = nn.Linear(embedding_dim, level0_width)

    def forward(self, features: Mapping[str, torch.Tensor], output_size=None) -> torch.Tensor:
        projected = {name: self.projections[name](features[name]) for name in self.CHANNELS}

        # Identical levels4->3->2->1 additive top-down FPN.
        fused = {"level4": projected["level4"]}
        previous = fused["level4"]
        for name in ("level3", "level2", "level1"):
            previous = F.interpolate(previous, size=projected[name].shape[-3:], mode="trilinear", align_corners=False)
            previous = self.refinements[name](projected[name] + previous)
            fused[name] = previous

        # Genuine full-resolution fusion: refine level1 FEATURES (reusing the
        # transferred mask_refinement conv), upsample those FEATURES (not
        # logits) to the level0 grid, and additively fuse with the projected
        # encoder level0 feature. This is the sole place level0 information
        # enters the network, and it happens before any mask is formed.
        refined_level1 = self.mask_refinement(fused["level1"])
        level0_feature = self.level0_projection(features["level0"])
        upsampled_level1 = F.interpolate(
            self.level1_to_level0(refined_level1), size=level0_feature.shape[-3:],
            mode="trilinear", align_corners=False)
        fused_level0 = self.level0_refinement(level0_feature + upsampled_level1)  # [B, W, 128,128,160]

        batch = features["level1"].shape[0]
        query = self.query.expand(batch, -1, -1)
        for name in self.COARSE_TO_FINE:
            query = self.query_updates[name](query, fused[name])
        # One additional attention step over level0. Pooled 2x for memory
        # only -- this affects what the *query* sees, not the mask field.
        level0_query_tokens = self.level0_query_projection(F.avg_pool3d(fused_level0, kernel_size=2, stride=2))
        query = self.query_updates["level0"](query, level0_query_tokens)

        # Single mask head, evaluated once, directly on the full-grid fused
        # field. No half-resolution logits are ever computed or interpolated.
        mask_embedding = self.mask_embedding(query).squeeze(1)  # [B, embedding_dim]
        level0_embedding = self.level0_embedding_projection(mask_embedding)  # [B, level0_width]
        logits = torch.einsum("bc,bcdhw->bdhw", level0_embedding, fused_level0).unsqueeze(1)
        logits = logits + self.mask_bias.view(1, 1, 1, 1, 1)
        if output_size is not None and tuple(logits.shape[-3:]) != tuple(output_size):
            logits = F.interpolate(logits, size=tuple(output_size), mode="trilinear", align_corners=False)
        return logits


class UnetrStyleLevel0OneQueryMaskDecoder(nn.Module):
    """Full-depth, literature-matched full-resolution level0 decoder.

    Motivation (see outputs/level0_depth_diagnostic/): the residual boundary
    error remaining after the 192x128x160 resolution fix is small, scattered,
    1-4 native-voxel clusters, not axis-locked quantization on an unrefined
    axis -- a signature consistent with the *decoder* giving each voxel too
    little local context before its decision is finalized, not with a
    remaining resolution shortfall. TrueFullResolutionLevel0OneQueryMaskDecoder
    reaches full resolution with a *single* depthwise+pointwise conv pass at
    level0 (effective receptive field ~3 model-grid voxels). This class
    replaces that single pass with a real four-stage skip-connected residual
    upsampling chain, level4->3->2->1->0, using ``monai.networks.blocks.
    UnetrUpBlock`` unmodified at every stage -- the exact, published block
    class the original RS2-Net decoder itself is built from (confirmed from
    the authors' own RSSNet.forward(); each stage: transpose-conv upsample,
    concatenate with the frozen encoder's same-resolution skip connection,
    residual conv block, res_block=True). This is deliberately not a novel
    design: it is the original paper's own decoder depth, reused unmodified,
    wrapped around our existing query-conditioning mechanism.

    Channel width is tapered from embedding_dim (32) down to level0_width
    (8) for the single largest-grid stage (level1->level0, 192x128x160).
    This is not an aesthetic choice -- it is a measured hardware constraint,
    profiled directly on this machine before committing to any training:
    Conv3d forward+backward cost on this CPU build is a sharp, non-linear
    function of channel count at this grid size, not a smooth one -- an
    isolated UnetrUpBlock(in=32, out=W) stage at the level0 grid measured
    3.05s (W=6), 4.51s (W=8), 8.99s (W=10), 13.89s (W=12) forward+backward,
    and a *single* forward pass did not complete in over 8 minutes at W=32.
    Whatever the underlying cause (likely a CPU conv-algorithm selection
    threshold), the practical consequence is that level0_width=16 -- fine
    for the single depthwise+pointwise pass in
    TrueFullResolutionLevel0OneQueryMaskDecoder -- makes a real, denser
    UnetrUpBlock residual stage there take >200s per training sample
    (measured: one full decoder training step timed at 206.71s), which would
    put a single 39-sample training epoch at multiple hours and the full
    20-epoch schedule at many days. level0_width=8 was chosen as the
    narrowest safely-fast point on that measured curve. Tapering channel
    width toward the decoder's finest, largest-grid stage is itself standard
    practice in efficient U-Net-style decoders (and matches this project's
    own prior, already-validated level0_width precedent, just narrower here
    because a real residual UnetrUpBlock costs strictly more than a single
    depthwise+pointwise pass at the same width). Levels 4->3->2->1 stay at
    the full embedding_dim=32 width throughout -- also measured, not
    assumed: an isolated level2->level1-scale UnetrUpBlock(32,32) stage
    (the most expensive of these three) cost 0.63s forward + 1.59s backward,
    comfortably fast at that width.

    What is unchanged from TrueFullResolutionLevel0OneQueryMaskDecoder (so
    this is a single, isolated architectural variable, not a re-run of the
    resolution experiment): the one learned query, the per-level
    QueryUpdateBlock cross-attention update (it only ever reads the finished
    canvas at each scale, never the skip connections or upsampling
    directly), and the final decision rule -- one dot product between the
    query's mask embedding and the finished full-resolution canvas, with no
    logit-space interpolation. That final rule is what keeps this a
    *query-conditioned* decoder (capable, in principle, of segmenting
    different structures given different queries) rather than the original's
    fixed, single-purpose 1x1x1 output head. The level0_query_projection /
    level0_embedding_projection bridge (embedding_dim <-> level0_width for
    the attention step and final dot product) is reused unchanged from
    TrueFullResolutionLevel0OneQueryMaskDecoder.

    ``query``, ``mask_embedding``, ``mask_bias``, ``projections.level1/2/3/4``,
    and ``query_updates.level4/level3/level2/level1`` keep the exact
    names/shapes of TrueFullResolutionLevel0OneQueryMaskDecoder, so a
    converged checkpoint's decoder_state_dict transfers wherever the
    architecture is genuinely unchanged. The four ``up_blocks`` (real
    UnetrUpBlock residual convolutions), ``projections.level0``,
    ``query_updates.level0``, ``level0_query_projection``, and
    ``level0_embedding_projection`` have no prior-checkpoint counterpart with
    a matching shape (level0_width=16 here vs. the old decoder's level0
    fusion) and start at random initialization, same as any newly added
    capacity.
    """

    CHANNELS = {"level0": 48, "level1": 48, "level2": 96, "level3": 192, "level4": 384}
    COARSE_TO_FINE = ("level4", "level3", "level2", "level1", "level0")
    UP_STAGES = ("level3", "level2", "level1")  # embedding_dim-width stages; level0 handled separately below

    def __init__(self, embedding_dim: int = 32, num_heads: int = 4, level0_width: int = 8) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.level0_width = level0_width
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.normal_(self.query, std=0.02)
        self.projections = nn.ModuleDict({
            name: nn.Conv3d(channels, embedding_dim, kernel_size=1)
            for name, channels in self.CHANNELS.items() if name != "level0"
        })
        # level0's skip projects to the narrower, measured-tractable width.
        self.projections["level0"] = nn.Conv3d(self.CHANNELS["level0"], level0_width, kernel_size=1)

        # Real UNETR-style skip-connected residual upsampling -- the original
        # decoder's own block (monai.networks.blocks.UnetrUpBlock), reused
        # unmodified, at every stage including the final one (only its
        # channel width differs there; the block itself is identical).
        self.up_blocks = nn.ModuleDict({
            name: UnetrUpBlock(
                spatial_dims=3, in_channels=embedding_dim, out_channels=embedding_dim,
                kernel_size=3, upsample_kernel_size=2, norm_name="instance", res_block=True,
            ) for name in self.UP_STAGES
        })
        self.up_blocks["level0"] = UnetrUpBlock(
            spatial_dims=3, in_channels=embedding_dim, out_channels=level0_width,
            kernel_size=3, upsample_kernel_size=2, norm_name="instance", res_block=True,
        )
        self.query_updates = nn.ModuleDict({
            name: QueryUpdateBlock(embedding_dim, num_heads) for name in self.COARSE_TO_FINE
        })
        self.mask_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim))
        self.mask_bias = nn.Parameter(torch.zeros(1))

        # Bridge between embedding_dim (query space) and level0_width (mask
        # canvas space) for the final attention step and dot product --
        # reused unchanged from TrueFullResolutionLevel0OneQueryMaskDecoder.
        self.level0_query_projection = nn.Conv3d(level0_width, embedding_dim, 1)
        self.level0_embedding_projection = nn.Linear(embedding_dim, level0_width)

    def forward(self, features: Mapping[str, torch.Tensor], output_size=None) -> torch.Tensor:
        projected = {name: self.projections[name](features[name]) for name in self.CHANNELS}

        # Real skip-connected residual upsampling chain, coarsest to finest.
        # UnetrUpBlock.forward(inp, skip) upsamples `inp` to `skip`'s grid,
        # concatenates, and applies a residual conv block -- exactly the
        # original decoder's decoder4->3->2->1 pattern, extended one stage
        # further to reach level0 (the original's decoder1 already does this
        # for level0; ours simply keeps the chain going to the same place).
        canvas = projected["level4"]
        fused = {"level4": canvas}
        for name in (*self.UP_STAGES, "level0"):
            canvas = self.up_blocks[name](canvas, projected[name])
            fused[name] = canvas
        # fused["level0"]: [B, level0_width, *tile_size] -- the full native
        # model-grid canvas the final mask decision reads from directly.

        batch = features["level1"].shape[0]
        query = self.query.expand(batch, -1, -1)
        for name in ("level4", "level3", "level2", "level1"):
            query = self.query_updates[name](query, fused[name])
        # One additional attention step over level0. Pooled 2x for memory
        # only -- this affects what the *query* sees, not the mask field.
        level0_query_tokens = self.level0_query_projection(F.avg_pool3d(fused["level0"], kernel_size=2, stride=2))
        query = self.query_updates["level0"](query, level0_query_tokens)

        # Single mask head, evaluated once, directly on the full-grid fused
        # field. No half-resolution logits are ever computed or interpolated.
        mask_embedding = self.mask_embedding(query).squeeze(1)  # [B, embedding_dim]
        level0_embedding = self.level0_embedding_projection(mask_embedding)  # [B, level0_width]
        logits = torch.einsum("bc,bcdhw->bdhw", level0_embedding, fused["level0"]).unsqueeze(1)
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


def dice_bce_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float = 0.25,
    boundary_width: int = 1,
    smooth: float = 1e-5,
):
    """Dice + BCE plus one symmetric ground-truth boundary loss.

    The boundary band is the difference between a dilation and erosion of the
    target. BCE is averaged only inside that band, so false-positive and
    false-negative boundary voxels receive exactly the same treatment. This is
    deliberately different from an FP-heavy Tversky term: it emphasizes hard
    contours without rewarding conservative mask shrinkage.

    This deliberately supports the immediate one-voxel neighborhood only. It
    is built from adjacent-voxel label transitions instead of an expensive
    full-volume morphology kernel. No prediction-derived weighting or
    post-processing is used, keeping supervision deterministic.
    """
    if boundary_width != 1:
        raise ValueError("Only the controlled one-voxel boundary band is supported")
    target = target.float()
    probabilities = logits.sigmoid()
    reduce_dims = tuple(range(1, logits.ndim))
    intersection = (probabilities * target).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice_loss = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()
    bce_loss = F.binary_cross_entropy_with_logits(logits, target)

    # Mark both sides of every 6-connected foreground/background transition.
    # Targets do not require gradients, so direct boolean indexing is safe.
    binary_target = target >= 0.5
    boundary_band = torch.zeros_like(binary_target)
    for axis in range(2, target.ndim):
        lower = [slice(None)] * target.ndim
        upper = [slice(None)] * target.ndim
        lower[axis] = slice(None, -1)
        upper[axis] = slice(1, None)
        transition = binary_target[tuple(lower)] != binary_target[tuple(upper)]
        boundary_band[tuple(lower)] |= transition
        boundary_band[tuple(upper)] |= transition
    boundary_band = boundary_band.to(target.dtype)
    voxel_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    boundary_loss = (voxel_bce * boundary_band).sum() / boundary_band.sum().clamp_min(1.0)

    total = dice_loss + bce_loss + boundary_weight * boundary_loss
    return total, {
        "dice_loss": dice_loss.detach(),
        "bce_loss": bce_loss.detach(),
        "boundary_loss": boundary_loss.detach(),
        "boundary_fraction": boundary_band.mean().detach(),
    }


@torch.no_grad()
def volumetric_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    prediction = logits.sigmoid() >= threshold
    target_binary = target >= 0.5
    intersection = (prediction & target_binary).sum().float()
    denominator = prediction.sum().float() + target_binary.sum().float()
    return float(((2.0 * intersection + 1e-5) / (denominator + 1e-5)).cpu())
