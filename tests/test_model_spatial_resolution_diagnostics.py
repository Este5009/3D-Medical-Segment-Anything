"""Focused tests for spatial-resolution diagnostic primitives."""

import numpy as np
import torch

from scripts.diagnose_model_spatial_resolution import (
    digital_contour_statistics, effective_native_footprint, tile_layout,
    transition_coordinates,
)


def test_tile_layout_reproduces_half_overlap_and_center_padding():
    image = torch.zeros((1, 1, 72, 70, 113))
    padded, original, spatial, starts, selected = tile_layout(image)
    assert original == (72, 70, 113)
    assert spatial == (128, 128, 160)
    assert starts == [[0], [0], [0]]
    assert selected == (0, 0, 0)
    assert tuple(padded.shape[-3:]) == spatial


def test_level1_native_footprint_respects_axis_mapping():
    pixels, millimetres = effective_native_footprint((0.1, 0.1, 0.4))
    assert np.allclose(millimetres, (0.32, 0.50, 0.40))
    assert np.allclose(pixels, (3.2, 5.0, 1.0))


def test_transition_coordinates_detect_known_lattice():
    mask = np.zeros((12, 12), dtype=bool)
    mask[:, 4:8] = True
    x = transition_coordinates(mask, axis=1)
    assert set(x) == {4, 8}


def test_long_rectangle_has_axis_aligned_runs():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 3:17] = True
    metrics = digital_contour_statistics(mask)
    assert metrics["axis_run_max"] >= 10
    assert metrics["direction_changes"] > 0
