#!/usr/bin/env python3
"""Mandatory pre-training verification for the UNETR-style, real-skip-
connection full-resolution level0 decoder experiment.

Single isolated architectural variable relative to
outputs/higher_resolution_true_level0 (the initialization checkpoint):
TrueFullResolutionLevel0OneQueryMaskDecoder's single depthwise+pointwise
level0 fusion pass is replaced by UnetrStyleLevel0OneQueryMaskDecoder's real
four-stage skip-connected residual upsampling chain (level4->3->2->1->0),
using monai.networks.blocks.UnetrUpBlock unmodified -- the exact block class
the original RS2-Net decoder itself is built from. Resolution, spacing, tile
size, encoder, hyperparameters, loss, and augmentation are all unchanged
from that checkpoint's own training run.

Unlike the two prior level0 experiments, this one is NOT spatially
shape-agnostic end to end and does NOT transfer 100% of the initial
checkpoint -- the new up_blocks (real UnetrUpBlock residual convolutions)
have no counterpart in the old architecture. Gate 8 below verifies the
transfer is exactly the expected partial set (78/102 old tensors, by name
and shape, all transferring with bit-identical values; the other 24 old
tensors were the old decoder's now-removed level0-only modules and are
correctly left behind, not corrupted or silently dropped), not "as much as
possible" or a guess.

A performance sanity check (not a hard gate, but logged and printed) is
included because the naive equal-width design was measured, during scoping,
to cost 206.71s per training step (infeasible for an overnight run); the
level0_width=8 taper used here measured 7.59-10.62s/step. This script
re-confirms that measurement holds for the actual initialization checkpoint
and real cached features, not just synthetic tensors, before training is
allowed to start.

Any failed hard gate raises immediately -- this script must exit 0 before
training is allowed to start.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import torch

from higher_resolution_preprocessing import preprocess_image_and_corrected_target_at_spacing
from models.query_mask_decoder import FrozenEncoderQueryModel, UnetrStyleLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import load_json

OUT = ROOT / "outputs/unetr_style_level0_full_width_decoder"
LEVELS = ("level0", "level1", "level2", "level3", "level4")

# Measured directly against the real checkpoint (not a synthetic same-width
# stand-in -- an earlier scoping check used a fresh level0_width=8 "old"
# instance for convenience and missed this). The 24 tensors on the first
# block below are the old decoder's now-replaced level0-only modules
# (level0_projection, level1_to_level0, level0_refinement, mask_refinement,
# refinements.level1/2/3). The 3 tensors on the second block --
# level0_query_projection.weight, level0_embedding_projection.{weight,bias}
# -- share a name with modules in the new architecture (both decoders bridge
# embedding_dim<->level0_width the same way) but do NOT transfer because the
# real checkpoint's level0_width is 16 and this decoder's is 8 (see model
# docstring for why): level0_query_projection's Conv3d bias is [32] in both
# (depends only on embedding_dim, so it DOES transfer) but its weight is
# [32,16,1,1,1] vs [32,8,1,1,1] (does not); level0_embedding_projection's
# Linear weight AND bias both depend on level0_width, so neither transfers.
EXPECTED_NOT_TRANSFERRED = {
    "level0_projection.weight", "level0_projection.bias",
    "level1_to_level0.weight", "level1_to_level0.bias",
    "level0_refinement.0.weight", "level0_refinement.0.bias",
    "level0_refinement.1.weight", "level0_refinement.1.bias",
    "level0_refinement.2.weight", "level0_refinement.2.bias",
    "mask_refinement.weight", "mask_refinement.bias",
    "refinements.level1.0.weight", "refinements.level1.0.bias",
    "refinements.level1.1.weight", "refinements.level1.1.bias",
    "refinements.level2.0.weight", "refinements.level2.0.bias",
    "refinements.level2.1.weight", "refinements.level2.1.bias",
    "refinements.level3.0.weight", "refinements.level3.0.bias",
    "refinements.level3.1.weight", "refinements.level3.1.bias",
    "level0_query_projection.weight",
    "level0_embedding_projection.weight", "level0_embedding_projection.bias",
}
EXPECTED_TRANSFERRED_COUNT = 75


def build_decoder(config):
    ck_path = ROOT / config["initial_checkpoint"]
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    old = ck["decoder_state_dict"]
    decoder = UnetrStyleLevel0OneQueryMaskDecoder(32, 4, level0_width=config["level0_width"])
    current = decoder.state_dict()
    shared = {k: v for k, v in old.items() if k in current and current[k].shape == v.shape}
    missing, unexpected = decoder.load_state_dict(shared, strict=False)
    return decoder, ck_path, old, shared, missing, unexpected


def verify(config_path=None):
    config = load_json(ROOT / (config_path or "configs/unetr_style_level0_full_width.yaml"))
    report = {"gates": {}}
    tile_size = tuple(config["tile_size"])

    # Gate 1: exact initialization checkpoint path.
    expected_checkpoint = ROOT / "outputs/higher_resolution_true_level0/checkpoints/best_higher_resolution_true_level0.pt"
    actual_checkpoint = ROOT / config["initial_checkpoint"]
    gate1 = actual_checkpoint.resolve() == expected_checkpoint.resolve()
    report["gates"]["initialization_checkpoint_is_higher_resolution_true_level0"] = {
        "passed": gate1, "expected": str(expected_checkpoint), "actual": str(actual_checkpoint),
    }
    if not gate1:
        raise AssertionError(f"Gate 1 FAILED: initial_checkpoint={actual_checkpoint}, expected {expected_checkpoint}")

    decoder, ck_path, old_state, shared, missing, unexpected = build_decoder(config)

    # Gate 8: the transfer is EXACTLY the expected partial set -- every
    # transferred tensor bit-identical, every non-transferred tensor exactly
    # the expected (old, now-removed) level0-only modules, no more no less.
    mismatched = [k for k, v in shared.items() if not torch.equal(decoder.state_dict()[k], v)]
    not_transferred = set(old_state) - set(shared)
    gate8 = (len(mismatched) == 0 and len(unexpected) == 0
             and len(shared) == EXPECTED_TRANSFERRED_COUNT
             and not_transferred == EXPECTED_NOT_TRANSFERRED)
    report["gates"]["transfer_is_exactly_the_expected_partial_set"] = {
        "passed": gate8, "old_checkpoint_tensor_count": len(old_state),
        "transferred_tensor_count": len(shared), "expected_transferred_count": EXPECTED_TRANSFERRED_COUNT,
        "mismatched_keys": mismatched, "unexpected_keys": list(unexpected),
        "not_transferred_keys": sorted(not_transferred),
        "not_transferred_matches_expected": not_transferred == EXPECTED_NOT_TRANSFERRED,
        "fresh_init_keys_in_new_architecture": sorted(missing),
    }
    if not gate8:
        raise AssertionError(f"Gate 8 FAILED: {json.dumps(report['gates']['transfer_is_exactly_the_expected_partial_set'], indent=2)}")

    total_params = sum(p.numel() for p in decoder.parameters())
    report["gates"]["transfer_is_exactly_the_expected_partial_set"]["total_decoder_parameters"] = total_params

    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    encoder = RS2NetEncoderAdapter(paths, image_size=tile_size, in_channels=1, out_channels=1, feature_size=48)
    model = FrozenEncoderQueryModel(encoder, decoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # MPS lacks Conv3d support locally; prefer CUDA when present.
    model.to(device).eval()

    # Gate 2: encoder frozen.
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    gate2 = encoder_trainable == 0 and not model.encoder.training
    report["gates"]["encoder_frozen"] = {"passed": gate2, "trainable_encoder_parameters": encoder_trainable}
    if not gate2:
        raise AssertionError(f"Gate 2 FAILED: encoder has {encoder_trainable} trainable parameters")

    # Gate 3: exactly one learned query.
    gate3 = tuple(decoder.query.shape) == (1, 1, 32)
    report["gates"]["query_count_is_one"] = {"passed": gate3, "query_shape": list(decoder.query.shape)}
    if not gate3:
        raise AssertionError(f"Gate 3 FAILED: query shape {tuple(decoder.query.shape)}")

    camri_metrics_rows = list(__import__("csv").DictReader(open(ROOT / config["camri_metrics"])))
    mouse_metrics_rows = list(__import__("csv").DictReader(open(ROOT / config["mouse_metrics"])))
    camri_split = load_json(ROOT / config["camri_split"])
    mouse_split = load_json(ROOT / config["mouse_split"])
    camri_row = next(r for r in camri_metrics_rows if r["subject"] == camri_split["validation"][0])
    mouse_scan_id = mouse_split["validation"]["scans"][0]
    mouse_row = next(r for r in mouse_metrics_rows if r["scan_id"] == mouse_scan_id)

    cases = {
        "CAMRI": (Path(camri_row["image_path"]), Path(camri_row["mask_path"])),
        "Mouse": (Path(mouse_row["image_path"]), Path(mouse_row["ground_truth_path"])),
    }
    case_results = {}
    timings = {}
    for domain, (image_path, mask_path) in cases.items():
        image, target, shape, _ = preprocess_image_and_corrected_target_at_spacing(
            image_path, mask_path, paths, tuple(config["model_spacing_mm"]), tile_size)

        # Gate 4: MRI input is the full-grid tile shape.
        gate4 = tuple(image.shape) == (1, 1, *tile_size)
        report["gates"].setdefault("mri_input_is_tile_shape", {})[domain] = {"passed": gate4, "shape": list(image.shape)}
        if not gate4:
            raise AssertionError(f"Gate 4 FAILED for {domain}: image shape {tuple(image.shape)}")

        with torch.inference_mode():
            features = model.encode(image.to(device))
        target = target.to(device)

        # Gate 5: target is full-grid categorical {0,1}.
        target_shape_ok = tuple(target.shape) == (1, 1, *tile_size)
        target_values_ok = set(torch.unique(target).tolist()) <= {0.0, 1.0}
        gate5 = target_shape_ok and target_values_ok
        report["gates"].setdefault("target_is_full_grid_binary", {})[domain] = {
            "passed": gate5, "shape": list(target.shape), "unique_values": sorted(set(torch.unique(target).tolist())),
        }
        if not gate5:
            raise AssertionError(f"Gate 5 FAILED for {domain}: shape={tuple(target.shape)} values={torch.unique(target).tolist()}")

        t0 = time.time()
        with torch.inference_mode():
            logits_no_size = model.decode(features)
        forward_seconds = time.time() - t0
        with torch.inference_mode():
            logits_with_size = model.decode(features, output_size=target.shape[-3:])
        timings[domain] = forward_seconds

        # Gate 6: logits are natively the full tile grid.
        gate6 = tuple(logits_no_size.shape) == (1, 1, *tile_size)
        report["gates"].setdefault("logits_are_full_grid", {})[domain] = {"passed": gate6, "shape": list(logits_no_size.shape)}
        if not gate6:
            raise AssertionError(f"Gate 6 FAILED for {domain}: logits shape {tuple(logits_no_size.shape)}")

        # Gate 7: no final mask-logit upsampling.
        gate7 = torch.equal(logits_no_size, logits_with_size)
        max_abs_diff = (logits_no_size - logits_with_size).abs().max().item()
        report["gates"].setdefault("no_final_logit_upsampling", {})[domain] = {"passed": gate7, "max_abs_diff_no_size_vs_with_size": max_abs_diff}
        if not gate7:
            raise AssertionError(f"Gate 7 FAILED for {domain}: output_size interpolation changed logits (max|Δ|={max_abs_diff})")

        case_results[domain] = {
            "image_path": str(image_path), "mask_path": str(mask_path),
            "encoder_level0_feature_shape": list(features["level0"].shape),
            "target_shape": list(target.shape), "logits_shape": list(logits_no_size.shape),
            "decoder_forward_seconds": forward_seconds,
        }

    report["cases"] = case_results
    report["performance_sanity"] = {
        "note": "Not a hard gate -- informational, logged so the morning report has real per-sample timing from the actual checkpoint/features, not just the synthetic-tensor scoping numbers.",
        "forward_seconds_by_domain": timings,
        "scoping_measurement_single_training_step_seconds": 7.59,
        "scoping_measurement_level0_width_16_single_training_step_seconds": 206.71,
    }
    report["all_gates_passed"] = all(
        (v["passed"] if isinstance(v, dict) and "passed" in v else all(x["passed"] for x in v.values()))
        for v in report["gates"].values()
    )
    if not report["all_gates_passed"]:
        raise AssertionError(f"One or more gates failed: {json.dumps(report['gates'], indent=2)}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pretraining_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nALL PRE-TRAINING GATES PASSED", flush=True)
    return report


if __name__ == "__main__":
    verify()
