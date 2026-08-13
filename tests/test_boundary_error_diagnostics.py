"""Focused mathematical tests for the post-filter boundary diagnostic."""

import numpy as np

from scripts.analyze_boundary_error_diagnostics import (
    align_probability_to_raw, binned_counts, binary_metrics,
    distance_to_surface, expert_surface,
)


def test_error_distance_is_to_expert_surface():
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[2:7, 2:7, 2:7] = True
    surface = expert_surface(mask)
    voxel, millimetres, _ = distance_to_surface(surface, (0.1, 0.2, 0.4))
    assert voxel[7, 4, 4] == 1.0
    assert np.isclose(millimetres[7, 4, 4], 0.1)
    assert voxel[4, 4, 4] == 2.0
    assert np.isclose(millimetres[4, 4, 4], 0.2)


def test_distance_bins_are_exclusive_and_exhaustive():
    values = np.array([1, 1.1, 2, 2.1, 3, 3.1, 5, 5.1])
    result = binned_counts(values)
    assert [result[f"{label}_count"] for label in
            ("<=1 voxel", ">1 to 2 voxels", ">2 to 3 voxels",
             ">3 to 5 voxels", ">5 voxels")] == [1, 2, 2, 2, 1]


def test_identical_masks_have_perfect_surface_metrics():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    metrics = binary_metrics(mask, mask, (0.2, 0.2, 0.5))
    assert metrics["dice"] == 1.0
    assert metrics["hd95_mm"] == 0.0
    assert metrics["assd_mm"] == 0.0
    assert all(metrics[f"surface_dice_{t:g}mm"] == 1.0
               for t in (0.1, 0.2, 0.5, 1.0))


def test_anisotropic_physical_distance_is_not_index_distance():
    surface = np.zeros((7, 7, 7), dtype=bool)
    surface[3, 3, 3] = True
    voxel, millimetres, _ = distance_to_surface(surface, (0.1, 0.1, 1.0))
    assert voxel[3, 3, 4] == voxel[4, 3, 3] == 1.0
    assert np.isclose(millimetres[3, 3, 4], 1.0)
    assert np.isclose(millimetres[4, 3, 3], 0.1)


def test_probability_axis_alignment_requires_exact_raw_reproduction():
    raw = np.zeros((5, 5, 3), dtype=bool)
    raw[1:4, 2:4, 1] = True
    probability = np.where(raw, .9, .1).transpose(1, 0, 2)
    aligned, permutation = align_probability_to_raw(probability, raw)
    assert permutation == (1, 0, 2)
    assert np.array_equal(aligned >= .5, raw)
