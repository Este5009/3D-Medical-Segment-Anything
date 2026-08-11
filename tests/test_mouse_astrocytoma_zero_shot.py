"""Synthetic tests for pathology-series QC and qualitative mask analysis."""
from __future__ import annotations
from pathlib import Path

import numpy as np

from scripts.prepare_mouse_astrocytoma_zero_shot import (
    first_spatial_volume, initial_qc_class, robust_normalize,
)
from scripts.run_mouse_astrocytoma_zero_shot import (
    align_probability_to_raw, cavity_analysis, indentation_analysis,
)
from models.query_conditioned_grouping import largest_component_3d


def test_dynamic_repeated_positions_are_detected():
    records = [
        {"position": z, "temporal": t, "acquisition": 0, "instance": t * 2 + z}
        for t in range(3) for z in range(2)
    ]
    selected, repetitions = first_spatial_volume(records)
    assert len(selected) == 2 and repetitions == 3


def test_anatomical_and_dynamic_classification():
    assert initial_qc_class("TSE_100_Cor_P4", "", "", 30, 1)[0] == "primary anatomical"
    assert initial_qc_class("3D Dyn22s", "", "", 12, 50)[0] == "dynamic"
    assert initial_qc_class("sT1_Subtr", "", "", 48, 1)[0] == "subtraction"


def test_normalization_is_finite():
    output = robust_normalize(np.ones((4, 4)))
    assert np.isfinite(output).all()


def test_largest_component_is_the_only_prediction_cleanup():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[1:5, 1:5, 1:5] = 1
    mask[8, 8, 8] = 1
    filtered = largest_component_3d(mask)
    assert filtered.sum() == 64
    assert filtered[8, 8, 8] == 0


def test_cavity_analysis_does_not_modify_prediction():
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[1:8, 1:8, 1:8] = True
    mask[4, 4, 4] = False
    before = mask.copy()
    cavity, rows = cavity_analysis(mask, np.eye(4))
    assert np.array_equal(mask, before)
    assert cavity.sum() == 1
    assert rows[0]["voxels"] == 1


def test_indentation_flags_are_analysis_only():
    mask = np.zeros((8, 8, 5), dtype=bool)
    mask[1:7, 1:7, :] = True
    mask[3:7, :, 2] = False
    before = mask.copy()
    probability = np.full(mask.shape, .9, dtype=np.float32)
    rows = indentation_analysis(mask, probability)
    assert np.array_equal(mask, before)
    assert any(row["slice"] == 2 for row in rows)


def test_probability_axis_alignment_requires_exact_threshold_identity():
    probability = np.zeros((5, 4, 3), dtype=np.float32)
    probability[1:4, 1:3, :] = .8
    raw = probability.transpose(1, 0, 2) > .5
    aligned, audit = align_probability_to_raw(probability, raw)
    assert np.array_equal(aligned > .5, raw)
    assert audit["threshold_mismatched_voxels"] == 0
