"""Focused safety and shape tests for the post-decoder grouping experiment."""
from __future__ import annotations

import numpy as np
import torch

from models.query_conditioned_grouping import (
    QueryConditionedGroupingDecoder,
    largest_component_3d,
    normalized_coordinate_grid,
)


def small_inputs():
    level2 = torch.randn(1, 96, 4, 5, 6)
    logits = torch.randn(1, 1, 8, 10, 12)
    query = torch.randn(1, 1, 32)
    return level2, logits, query


def test_shape_and_identity_initialization():
    model = QueryConditionedGroupingDecoder()
    level2, logits, query = small_inputs()
    final, correction = model(level2, logits, query)
    assert final.shape == logits.shape
    assert correction.shape == logits.shape
    assert torch.equal(final, logits)


def test_residual_is_bounded():
    model = QueryConditionedGroupingDecoder(max_correction=1.25)
    torch.nn.init.constant_(model.correction_head.bias, 100)
    _, correction = model(*small_inputs())
    assert correction.abs().max() <= 1.25 + 1e-6


def test_query_conditioning_changes_result_after_nonzero_head():
    model = QueryConditionedGroupingDecoder()
    torch.nn.init.normal_(model.correction_head.weight)
    level2, logits, query = small_inputs()
    enabled, _ = model(level2, logits, query, use_query=True)
    disabled, _ = model(level2, logits, query, use_query=False)
    assert not torch.allclose(enabled, disabled)


def test_coordinate_grid_range_and_shape():
    grid = normalized_coordinate_grid((3, 4, 5), device=torch.device("cpu"), dtype=torch.float32)
    assert grid.shape == (1, 3, 3, 4, 5)
    assert float(grid.min()) == -1.0
    assert float(grid.max()) == 1.0


def test_largest_component_uses_full_3d_26_connectivity():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, 1, 1] = 1  # diagonally connected only under 26-connectivity
    mask[4, 4, 4] = 1
    filtered = largest_component_3d(mask)
    assert filtered.sum() == 2
    assert filtered[0, 0, 0] and filtered[1, 1, 1]


def test_component_filter_does_not_accept_ground_truth():
    # Its API has one prediction argument, preventing accidental label access.
    import inspect
    assert list(inspect.signature(largest_component_3d).parameters) == ["mask"]


def test_backward_reaches_grouping_only():
    model = QueryConditionedGroupingDecoder()
    level2, logits, query = small_inputs()
    level2.requires_grad_(False)
    logits.requires_grad_(False)
    query.requires_grad_(False)
    final, _ = model(level2, logits, query)
    final.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert level2.grad is None and logits.grad is None and query.grad is None


def test_checkpoint_round_trip(tmp_path):
    model = QueryConditionedGroupingDecoder(use_coordinates=False)
    path = tmp_path / "grouping.pt"
    torch.save({"grouping_state_dict": model.state_dict(), "options": {"use_coordinates": False}}, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = QueryConditionedGroupingDecoder(use_coordinates=payload["options"]["use_coordinates"])
    restored.load_state_dict(payload["grouping_state_dict"], strict=True)


def test_binary_metrics_known_example():
    from scripts.train_mouse_boundary_adaptation import metric
    logits = torch.tensor([[[[[10.0, 10.0, -10.0, -10.0]]]]])
    target = torch.tensor([[[[[1.0, 0.0, 1.0, 0.0]]]]])
    result = metric(logits, target)
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["dice"] == 0.5
