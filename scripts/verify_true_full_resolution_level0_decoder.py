#!/usr/bin/env python3
"""Mandatory pre-training verification for the TRUE level0 decoder ablation.

Runs entirely without training. Loads one real CAMRI and one real Mouse
sample through the frozen encoder, builds the new decoder exactly as the
training script will, and hard-asserts every gate the audit requested:

  1. initialization checkpoint is exactly best_corrected_labels.pt
  2. encoder is frozen (no trainable parameters)
  3. query count is exactly one
  4. target is [1,1,128,128,160] with values in {0,1}
  5. new decoder logits are [1,1,128,128,160]
  6. no final mask-logit upsampling occurs (forward with no output_size
     already returns full grid; forward with output_size == that grid is
     bit-identical, proving the interpolation branch is never exercised)
  7. every inherited levels1-4/query/mask-head tensor is bit-identical to
     best_corrected_labels.pt's decoder_state_dict

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

from corrected_label_preprocessing import preprocess_image_and_corrected_target
from models.query_mask_decoder import FrozenEncoderQueryModel, TrueFullResolutionLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import choose_device, load_json

OUT = ROOT / "outputs/true_full_resolution_level0_decoder"
LEVELS = ("level0", "level1", "level2", "level3", "level4")


def build_decoder(config):
    """Mirror scripts/train_true_full_resolution_level0_decoder.py::initialize()."""
    ck_path = ROOT / config["initial_checkpoint"]
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    old = ck["decoder_state_dict"]
    decoder = TrueFullResolutionLevel0OneQueryMaskDecoder(32, 4, level0_width=config["level0_width"])
    current = decoder.state_dict()
    shared = {k: v for k, v in old.items() if k in current and current[k].shape == v.shape}
    missing, unexpected = decoder.load_state_dict(shared, strict=False)
    if set(shared) != set(old):
        raise RuntimeError(
            f"Not all baseline decoder tensors transferred: missing from shared "
            f"{set(old) - set(shared)}"
        )
    return decoder, ck_path, old, missing, unexpected


def verify(config_path=None):
    config = load_json(ROOT / (config_path or "configs/true_full_resolution_level0_decoder.yaml"))
    report = {"gates": {}}

    # Gate 1: exact initialization checkpoint path.
    expected_checkpoint = ROOT / "outputs/corrected_label_retraining/checkpoints/best_corrected_labels.pt"
    actual_checkpoint = ROOT / config["initial_checkpoint"]
    gate1 = actual_checkpoint.resolve() == expected_checkpoint.resolve()
    report["gates"]["initialization_checkpoint_is_corrected_label_baseline"] = {
        "passed": gate1, "expected": str(expected_checkpoint), "actual": str(actual_checkpoint),
    }
    if not gate1:
        raise AssertionError(f"Gate 1 FAILED: initial_checkpoint={actual_checkpoint}, expected {expected_checkpoint}")

    decoder, ck_path, old_state, missing, unexpected = build_decoder(config)

    # Gate 7: every inherited tensor is bit-identical to the source checkpoint.
    current_state = decoder.state_dict()
    mismatched = [k for k in old_state if k in current_state and not torch.equal(current_state[k], old_state[k])]
    gate7 = len(mismatched) == 0 and len(unexpected) == 0
    report["gates"]["inherited_levels1_4_weights_match_corrected_label_checkpoint"] = {
        "passed": gate7, "old_tensor_count": len(old_state),
        "transferred_tensor_count": len(old_state) - len(mismatched),
        "mismatched_keys": mismatched, "unexpected_keys": list(unexpected),
        "new_parameter_keys": sorted(missing),
    }
    if not gate7:
        raise AssertionError(f"Gate 7 FAILED: mismatched={mismatched} unexpected={unexpected}")

    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    encoder = RS2NetEncoderAdapter(paths, image_size=tuple(config["tile_size"]), in_channels=1, out_channels=1, feature_size=48)
    model = FrozenEncoderQueryModel(encoder, decoder)
    device = torch.device("cpu")  # MPS lacks Conv3d support in the installed torch 2.0.0; forced CPU, matching the baseline's recorded execution device.
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
        image, target, shape, _ = preprocess_image_and_corrected_target(image_path, mask_path, paths, tuple(config["tile_size"]))
        with torch.inference_mode():
            features = model.encode(image.to(device))
        target = target.to(device)

        # Gate 4: target is full-grid categorical {0,1}.
        target_shape_ok = tuple(target.shape) == (1, 1, 128, 128, 160)
        target_values_ok = set(torch.unique(target).tolist()) <= {0.0, 1.0}
        gate4 = target_shape_ok and target_values_ok
        report["gates"].setdefault("target_is_full_grid_binary", {})[domain] = {
            "passed": gate4, "shape": list(target.shape), "unique_values": sorted(set(torch.unique(target).tolist())),
        }
        if not gate4:
            raise AssertionError(f"Gate 4 FAILED for {domain}: shape={tuple(target.shape)} values={torch.unique(target).tolist()}")

        with torch.inference_mode():
            logits_no_size = model.decode(features)
            logits_with_size = model.decode(features, output_size=target.shape[-3:])

        # Gate 5: new logits are natively [1,1,128,128,160].
        gate5 = tuple(logits_no_size.shape) == (1, 1, 128, 128, 160)
        report["gates"].setdefault("logits_are_full_grid", {})[domain] = {
            "passed": gate5, "shape": list(logits_no_size.shape),
        }
        if not gate5:
            raise AssertionError(f"Gate 5 FAILED for {domain}: logits shape {tuple(logits_no_size.shape)}")

        # Gate 6: no final mask-logit upsampling -- passing output_size equal
        # to the already-native shape must be a bitwise no-op.
        gate6 = torch.equal(logits_no_size, logits_with_size)
        max_abs_diff = (logits_no_size - logits_with_size).abs().max().item()
        report["gates"].setdefault("no_final_logit_upsampling", {})[domain] = {
            "passed": gate6, "max_abs_diff_no_size_vs_with_size": max_abs_diff,
        }
        if not gate6:
            raise AssertionError(f"Gate 6 FAILED for {domain}: output_size interpolation changed logits (max|Δ|={max_abs_diff})")

        case_results[domain] = {
            "image_path": str(image_path), "mask_path": str(mask_path),
            "encoder_level0_feature_shape": list(features["level0"].shape),
            "target_shape": list(target.shape),
            "logits_shape": list(logits_no_size.shape),
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
