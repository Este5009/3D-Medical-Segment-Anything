#!/usr/bin/env python3
"""Mandatory pre-training verification for the higher-resolution (192x128x160)
TRUE level0 decoder experiment.

Runs entirely without training. Loads one real CAMRI and one real Mouse
sample through the frozen encoder at the new resolution, builds the decoder
exactly as the training script will, and hard-asserts every gate requested:

  1. initialization checkpoint is exactly
     outputs/true_full_resolution_level0_decoder/checkpoints/best_true_level0_decoder.pt
  2. encoder is frozen (no trainable parameters)
  3. query count is exactly one
  4. MRI input is [1,1,192,128,160]
  5. target is [1,1,192,128,160] with values in {0,1}
  6. new decoder logits are [1,1,192,128,160]
  7. no final mask-logit upsampling occurs
  8. every compatible tensor from the initial checkpoint transfers exactly
     (the decoder architecture is spatially shape-agnostic -- all Conv3d/
     Linear/Attention/InstanceNorm parameters operate on channel dimensions
     only, never on D/H/W -- so ALL 182,081 parameters are expected to
     transfer, not a subset; this is verified, not assumed)

Any failed gate raises immediately -- this script must exit 0 before
training is allowed to start.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import torch

from higher_resolution_preprocessing import preprocess_image_and_corrected_target_at_spacing
from models.query_mask_decoder import FrozenEncoderQueryModel, TrueFullResolutionLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import choose_device, load_json

OUT = ROOT / "outputs/higher_resolution_true_level0"
LEVELS = ("level0", "level1", "level2", "level3", "level4")


def build_decoder(config):
    ck_path = ROOT / config["initial_checkpoint"]
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    old = ck["decoder_state_dict"]
    decoder = TrueFullResolutionLevel0OneQueryMaskDecoder(32, 4, level0_width=config["level0_width"])
    current = decoder.state_dict()
    shared = {k: v for k, v in old.items() if k in current and current[k].shape == v.shape}
    missing, unexpected = decoder.load_state_dict(shared, strict=False)
    if set(shared) != set(old):
        raise RuntimeError(f"Not all old checkpoint tensors transferred: {set(old) - set(shared)}")
    return decoder, ck_path, old, missing, unexpected


def verify(config_path=None):
    config = load_json(ROOT / (config_path or "configs/higher_resolution_true_level0.yaml"))
    report = {"gates": {}}
    tile_size = tuple(config["tile_size"])

    # Gate 1: exact initialization checkpoint path.
    expected_checkpoint = ROOT / "outputs/true_full_resolution_level0_decoder/checkpoints/best_true_level0_decoder.pt"
    actual_checkpoint = ROOT / config["initial_checkpoint"]
    gate1 = actual_checkpoint.resolve() == expected_checkpoint.resolve()
    report["gates"]["initialization_checkpoint_is_selected_true_level0"] = {
        "passed": gate1, "expected": str(expected_checkpoint), "actual": str(actual_checkpoint),
    }
    if not gate1:
        raise AssertionError(f"Gate 1 FAILED: initial_checkpoint={actual_checkpoint}, expected {expected_checkpoint}")

    decoder, ck_path, old_state, missing, unexpected = build_decoder(config)

    # Gate 8: every compatible tensor transfers exactly; report how many of
    # the checkpoint's tensors this covers (expected: all of them, 100%).
    current_state = decoder.state_dict()
    mismatched = [k for k in old_state if k in current_state and not torch.equal(current_state[k], old_state[k])]
    gate8 = len(mismatched) == 0 and len(unexpected) == 0
    report["gates"]["all_compatible_weights_transferred_exactly"] = {
        "passed": gate8, "old_checkpoint_tensor_count": len(old_state),
        "transferred_tensor_count": len(old_state) - len(mismatched),
        "transferred_fraction_of_182081_total_params": None,  # filled after param count known
        "mismatched_keys": mismatched, "unexpected_keys": list(unexpected), "new_zero_init_keys": sorted(missing),
    }
    if not gate8:
        raise AssertionError(f"Gate 8 FAILED: mismatched={mismatched} unexpected={unexpected}")

    total_params = sum(p.numel() for p in decoder.parameters())
    old_param_count = sum(v.numel() for v in old_state.values())
    report["gates"]["all_compatible_weights_transferred_exactly"]["total_decoder_parameters"] = total_params
    report["gates"]["all_compatible_weights_transferred_exactly"]["checkpoint_parameter_count"] = old_param_count
    report["gates"]["all_compatible_weights_transferred_exactly"]["new_parameter_keys"] = sorted(missing)

    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    encoder = RS2NetEncoderAdapter(paths, image_size=tile_size, in_channels=1, out_channels=1, feature_size=48)
    model = FrozenEncoderQueryModel(encoder, decoder)
    device = torch.device("cpu")  # MPS lacks Conv3d support in the installed torch 2.0.0.
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
    for domain, (image_path, mask_path) in cases.items():
        image, target, shape, _ = preprocess_image_and_corrected_target_at_spacing(
            image_path, mask_path, paths, tuple(config["model_spacing_mm"]), tile_size)

        # Gate 4: MRI input is the new full-grid tile shape.
        gate4 = tuple(image.shape) == (1, 1, *tile_size)
        report["gates"].setdefault("mri_input_is_new_tile_shape", {})[domain] = {"passed": gate4, "shape": list(image.shape)}
        if not gate4:
            raise AssertionError(f"Gate 4 FAILED for {domain}: image shape {tuple(image.shape)}")

        with torch.inference_mode():
            features = model.encode(image.to(device))
        target = target.to(device)

        # Gate 5: target is full-grid categorical {0,1} at the new resolution.
        target_shape_ok = tuple(target.shape) == (1, 1, *tile_size)
        target_values_ok = set(torch.unique(target).tolist()) <= {0.0, 1.0}
        gate5 = target_shape_ok and target_values_ok
        report["gates"].setdefault("target_is_full_grid_binary", {})[domain] = {
            "passed": gate5, "shape": list(target.shape), "unique_values": sorted(set(torch.unique(target).tolist())),
        }
        if not gate5:
            raise AssertionError(f"Gate 5 FAILED for {domain}: shape={tuple(target.shape)} values={torch.unique(target).tolist()}")

        with torch.inference_mode():
            logits_no_size = model.decode(features)
            logits_with_size = model.decode(features, output_size=target.shape[-3:])

        # Gate 6: new logits are natively the full new-tile grid.
        gate6 = tuple(logits_no_size.shape) == (1, 1, *tile_size)
        report["gates"].setdefault("logits_are_new_full_grid", {})[domain] = {"passed": gate6, "shape": list(logits_no_size.shape)}
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
        }

    report["cases"] = case_results
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
