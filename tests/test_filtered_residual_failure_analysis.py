"""Controlled checks for deterministic filtered residual analysis."""
from __future__ import annotations

import numpy as np

from models.query_conditioned_grouping import largest_component_3d
from scripts.analyze_filtered_residual_failures import (
    aggregate_category_comparison,
    empty_atlas_message,
    prediction_slices,
    select_ranked_cases,
    validate_prediction_pairing,
)
from scripts.analyze_residual_failures import CATEGORIES


def test_cross_slice_connection_is_preserved():
    mask = np.zeros((5, 5, 4), dtype=np.uint8)
    mask[1:4, 1:4, 0] = 1
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1  # 26-connected through adjacent slices
    assert largest_component_3d(mask).sum() == mask.sum()


def test_floating_island_is_removed_and_largest_retained():
    mask = np.zeros((6, 6, 6), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    mask[5, 5, 5] = 1
    result = largest_component_3d(mask)
    assert result.sum() == 27
    assert not result[5, 5, 5]


def test_prediction_only_slice_selection():
    prediction = np.zeros((4, 4, 12), dtype=bool)
    prediction[:, :, 2:10] = True
    selected = prediction_slices(prediction)
    assert selected == [4, 6, 8]


def test_visual_ranking_uses_filtered_dice():
    rows = [
        {"domain": "Mouse", "subject": "a", "filtered_dice": 0.7},
        {"domain": "Mouse", "subject": "b", "filtered_dice": 0.8},
        {"domain": "Mouse", "subject": "c", "filtered_dice": 0.9},
    ]
    selected = select_ranked_cases(rows, "Mouse")
    assert [row["subject"] for _, row in selected] == ["a", "b", "c"]


class FakeAnalysis:
    @staticmethod
    def make(domain, total, values, affected=None):
        affected = affected or {}
        return {
            "summary": {"domain": domain, "total_error_voxels": total},
            "attributed": {category: values.get(category, 0) for category in CATEGORIES},
            "category_affected": {category: affected.get(category, False) for category in CATEGORIES},
        }


def test_absolute_and_percentage_calculation_is_exclusive():
    before = [FakeAnalysis.make("CAMRI", 100, {"Boundary Error": 80, "Leakage": 20})]
    after = [FakeAnalysis.make("CAMRI", 80, {"Boundary Error": 80})]
    rows = aggregate_category_comparison(before, after, "Combined")
    boundary = next(row for row in rows if row["failure_category"] == "Boundary Error")
    leakage = next(row for row in rows if row["failure_category"] == "Leakage")
    assert boundary["absolute_difference_filtered_minus_baseline"] == 0
    assert boundary["baseline_percentage_of_total_error"] == 80
    assert boundary["filtered_percentage_of_total_error"] == 100
    assert leakage["absolute_difference_filtered_minus_baseline"] == -20
    assert next(row for row in rows if row["failure_category"] == "Unclassified")["filtered_voxels"] == 0


def test_expert_overlap_is_preserved_when_expert_is_in_primary_component():
    prediction = np.zeros((7, 7, 7), dtype=np.uint8)
    prediction[1:5, 1:5, 1:5] = 1
    prediction[6, 6, 6] = 1
    expert = np.zeros_like(prediction)
    expert[2:4, 2:4, 2:4] = 1
    filtered = largest_component_3d(prediction)
    assert np.all(filtered[expert.astype(bool)])


def test_prediction_pairing_accepts_unique_matching_saved_files(tmp_path):
    baseline = tmp_path / "mouse-a_baseline.nii.gz"
    filtered = tmp_path / "mouse-a_filtered.nii.gz"
    baseline.touch(); filtered.touch()
    assert validate_prediction_pairing([{
        "domain": "Mouse", "subject": "mouse-a",
        "baseline_prediction_path": str(baseline), "filter_prediction_path": str(filtered),
    }]) == 1


def test_empty_detached_atlas_uses_required_statement():
    message = empty_atlas_message("Detached False Positive Island")
    assert message == (
        "No detached false-positive islands remained after deterministic "
        "full-volume connected-component filtering."
    )
