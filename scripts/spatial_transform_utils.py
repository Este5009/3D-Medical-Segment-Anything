"""Shared rigid-transform utilities for the orientation- and translation-
invariance tests. Used identically by both, and by the per-stage
architecture-localization probe, so every part of both experiments moves
volumes with exactly the same, once-verified math.

Correctness note: this project's model grid is ANISOTROPIC (0.167 x 0.200 x
0.160 mm per voxel, unequal voxel counts per axis: 192 x 128 x 160). A
rotation matrix applied directly in normalized [-1,1] tensor-index space
(PyTorch's affine_grid/grid_sample convention) would NOT correspond to a
true anatomical rotation on this grid -- it would be sheared, because one
normalized unit covers a different physical distance on each axis. Every
rotation here is built by explicitly converting normalized coordinates to
real physical millimeters, applying a standard 3x3 rotation matrix there,
then converting back -- so a "30 degree rotation" here really is 30 degrees
in real anatomical space, not in raw voxel-index space. Translations use
the same physical-mm framing for consistent, comparable reporting (mm
shifts, not an ambiguous voxel count that means something different on each
axis), even though a pure shift is not itself distorted by anisotropic
spacing the way rotation is.

All transforms are built about the volume's own center and are exactly
invertible (rotation inverse = transpose; translation inverse = negation),
so a transformed prediction can be mapped back into the original reference
frame for a direct, apples-to-apples comparison against the untouched
expert mask.
"""
from __future__ import annotations
import math
from typing import Sequence
import torch
import torch.nn.functional as F


def _physical_extent(shape: Sequence[int], spacing: Sequence[float]) -> torch.Tensor:
    """Full physical size of the volume along each axis, in mm."""
    return torch.tensor([s * sp for s, sp in zip(shape, spacing)], dtype=torch.float64)


def rotation_matrix(axis: str, angle_deg: float) -> torch.Tensor:
    """3x3 rotation matrix in physical (D, H, W) = (axis0, axis1, axis2)
    space, rotating about the named axis by angle_deg degrees."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "axis0":  # rotate the H-W (axis1-axis2) plane
        R = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == "axis1":  # rotate the D-W (axis0-axis2) plane
        R = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    elif axis == "axis2":  # rotate the D-H (axis0-axis1) plane
        R = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return torch.tensor(R, dtype=torch.float64)


def _grid_from_physical_matrix(shape, spacing, M3x3: torch.Tensor, translation_mm: torch.Tensor, device):
    """Builds the (1, D, H, W, 3) sampling grid for grid_sample from a 3x3
    physical-space transform matrix M3x3 (applied to OUTPUT physical
    coordinates to find the corresponding INPUT physical coordinates,
    i.e. this is the mapping grid_sample itself expects) plus a physical
    translation in mm. Internally converts through mm exactly once, using
    each axis's own real extent -- this is the piece that makes rotations
    correct under anisotropic spacing."""
    extent = _physical_extent(shape, spacing)  # (3,), mm, order (axis0,axis1,axis2)
    half_extent = extent / 2.0
    # normalized-output -> physical-output -> physical-input (via M, translation) -> normalized-input
    scale_out = torch.diag(half_extent)                     # normalized -> mm
    scale_in_inv = torch.diag(1.0 / half_extent)             # mm -> normalized
    full = scale_in_inv @ M3x3.double() @ scale_out           # normalized-out -> normalized-in (linear part)
    trans_normalized = (scale_in_inv @ translation_mm.double())

    theta = torch.zeros(1, 3, 4, dtype=torch.float64)
    theta[0, :, :3] = full
    theta[0, :, 3] = trans_normalized
    theta = theta.to(device=device, dtype=torch.float32)
    grid = F.affine_grid(theta, [1, 1, *shape], align_corners=False)
    return grid


def rotation_grid(shape, spacing, axis: str, angle_deg: float, device, invert: bool = False):
    R = rotation_matrix(axis, angle_deg)
    if invert:
        R = R.T  # inverse of a rotation matrix is its transpose
    return _grid_from_physical_matrix(shape, spacing, R, torch.zeros(3, dtype=torch.float64), device)


def translation_grid(shape, spacing, axis: str, shift_mm: float, device, invert: bool = False):
    idx = {"axis0": 0, "axis1": 1, "axis2": 2}[axis]
    t = torch.zeros(3, dtype=torch.float64)
    t[idx] = -shift_mm if not invert else shift_mm
    # Note the sign: grid_sample's grid maps OUTPUT coords to INPUT coords
    # to sample from, so to make the OUTPUT appear shifted by +shift_mm
    # relative to the input, we sample the input at output_coord - shift.
    I3 = torch.eye(3, dtype=torch.float64)
    return _grid_from_physical_matrix(shape, spacing, I3, t, device)


def apply_grid(tensor: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """tensor: (B, C, D, H, W). grid: (1, D, H, W, 3) as built above."""
    b = tensor.shape[0]
    g = grid.expand(b, -1, -1, -1, -1) if b > 1 else grid
    return F.grid_sample(tensor, g, mode=mode, padding_mode="zeros", align_corners=False)
