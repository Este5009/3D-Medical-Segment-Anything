#!/usr/bin/env python3
"""FAST diagnostic: where does fine Mouse boundary detail disappear?

Read-only / inference-only. No training, no optimizer, no full-cohort run.
Three already-characterized Mouse test subjects (pixelated / median / best
TRUE-level0 case, reused from the visual atlas) are each pushed through ONE
direct (non-tiled) encoder+decoder forward pass -- all three subjects'
preprocessed volumes fit inside a single 128x128x160 tile, verified below, so
no sliding-window inference is needed. Native-space probability/prediction
panels are loaded directly from already-saved files
(outputs/true_full_resolution_level0_decoder/), not recomputed.

Reused, unmodified, from scripts/diagnose_model_spatial_resolution.py (the
prior, already-validated spatial-resolution diagnostic):
  - model_axis_point           (native x/y/z -> model tile coordinate)
  - choose_native_boundary_point (picks the native slice/point of maximal
                                   expert/prediction disagreement)
  - crop_plane                 (crops a model-space volume around a point)
  - digital_contour_statistics (direction-change / axis-run statistics)
  - effective_native_footprint (physical footprint -> native-pixel units)
  - quantization_rows          (factor-N grid-alignment alignment fractions)
  - probability_detail_statistics (native probability isocontour subvoxel %)
  - feature_image              (RMS-across-channels activation map)
  - TARGET_SPACING
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure
import torch

from analyze_boundary_error_diagnostics import align_probability_to_raw
from corrected_label_preprocessing import preprocess_image_and_corrected_target
from diagnose_model_spatial_resolution import (
    model_axis_point, choose_native_boundary_point, crop_plane,
    digital_contour_statistics, effective_native_footprint, quantization_rows,
    probability_detail_statistics, feature_image, TARGET_SPACING,
)
from models.query_mask_decoder import FrozenEncoderQueryModel, TrueFullResolutionLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import load_json

OUT = ROOT / "outputs/boundary_stage_localization"
LEVEL0 = ROOT / "outputs/true_full_resolution_level0_decoder"
TILE_SIZE = (128, 128, 160)

# Reused directly from the visual-comparison atlas (manifest.csv), not
# re-derived, to keep this run fast and consistent with already-reported cases.
CASES = [
    {"role": "strongly_pixelated", "subject": "POLYIC_20190517_mouse39__E12_P1"},
    {"role": "median", "subject": "POLYIC_20190510_mouse37__E2_P1"},
    {"role": "best_true_level0", "subject": "POLYIC_20190517_mouse37__E3_P1"},
]


def isocontour_subvoxel_percent(probability2d):
    """Mirrors probability_detail_statistics's isocontour test, for one slice."""
    contours = measure.find_contours(probability2d.astype(np.float32), 0.5)
    if not contours:
        return float("nan"), 0
    points = np.concatenate(contours)
    fractional = np.abs(points - np.rint(points))
    return 100 * float(np.mean(fractional > 1e-3)), int(len(points))


def crop_native(array2d, xy, radius=28):
    x, y = int(xy[0]), int(xy[1])
    xs = slice(max(x - radius, 0), min(x + radius + 1, array2d.shape[0]))
    ys = slice(max(y - radius, 0), min(y + radius + 1, array2d.shape[1]))
    return array2d[xs, ys]


def load_model(config):
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    encoder = RS2NetEncoderAdapter(paths, image_size=TILE_SIZE, in_channels=1, out_channels=1, feature_size=48)
    ck = torch.load(LEVEL0 / "checkpoints/best_true_level0_decoder.pt", map_location="cpu", weights_only=False)
    decoder = TrueFullResolutionLevel0OneQueryMaskDecoder(32, 4, level0_width=config["level0_width"])
    decoder.load_state_dict(ck["decoder_state_dict"], strict=True)
    model = FrozenEncoderQueryModel(encoder, decoder).eval()
    return model, paths


def run_case(model, paths, row, metrics_rows):
    subject = row["subject"]
    image_path, gt_path = row["image_path"], row["ground_truth_path"]
    native_pred_path = row["level0_filtered_prediction_path"]
    native_raw_pred_path = row["level0_raw_prediction_path"]
    native_prob_path = row["level0_probability_path"]

    # --- Native arrays (already-saved outputs reused, not recomputed) ---
    native_image = np.asarray(nib.load(image_path).dataobj, dtype=np.float32)
    native_expert = np.asarray(nib.load(gt_path).dataobj) > 0
    native_pred = np.asarray(nib.load(native_pred_path).dataobj) > 0
    native_raw_pred = np.asarray(nib.load(native_raw_pred_path).dataobj) > 0
    native_prob_saved = np.asarray(nib.load(native_prob_path).dataobj, dtype=np.float32)
    # Per MEMORY.md: the saved probability exporter stores square in-plane
    # arrays in a possibly-transposed axis order. Recover the correct order
    # by finding the permutation whose unchanged 0.5 threshold exactly
    # reproduces the canonical RAW (unfiltered) native prediction.
    native_prob, prob_permutation = align_probability_to_raw(native_prob_saved, native_raw_pred)
    native_spacing = tuple(float(x) for x in nib.load(gt_path).header.get_zooms()[:3])

    # --- ONE direct (non-tiled) forward pass: preprocess -> encode -> decode ---
    with torch.inference_mode():
        image, target, shape, properties = preprocess_image_and_corrected_target(
            Path(image_path), Path(gt_path), paths, TILE_SIZE)
        fits_single_tile = all(s <= t for s, t in zip(shape[1:], TILE_SIZE))
        features = model.encode(image)
        logits = model.decode(features)
    model_prob = logits.sigmoid()[0, 0].numpy()
    model_expert = (target[0, 0].numpy() > 0.5)
    model_mri = image[0, 0].numpy()
    level0_activation = feature_image(features["level0"])  # RMS across 48 channels

    # --- Native boundary point (max expert/prediction disagreement) + mapping ---
    native_point = choose_native_boundary_point(native_expert, native_pred)
    x, y, z = int(native_point[0]), int(native_point[1]), int(native_point[2])
    model_point = model_axis_point(properties, native_point, TILE_SIZE, (0, 0, 0))

    # --- Cropped, same-region 2D panels ---
    r_native, r_model = 30, 30
    p_native_mri = crop_native(native_image[:, :, z], (x, y), r_native)
    p_native_expert = crop_native(native_expert[:, :, z], (x, y), r_native)
    p_native_pred = crop_native(native_pred[:, :, z], (x, y), r_native)
    p_native_prob = crop_native(native_prob[:, :, z], (x, y), r_native)
    p_model_mri = crop_plane(model_mri, model_point, r_model)
    p_model_expert = crop_plane(model_expert, model_point, r_model)
    p_model_prob = crop_plane(model_prob, model_point, r_model)
    p_level0_activation = crop_plane(level0_activation, model_point, r_model)

    # --- Figure: 8 stage panels, same anatomical region throughout ---
    fig, ax = plt.subplots(2, 4, figsize=(18, 9.5), constrained_layout=True)
    panels = [
        (p_native_mri, "1. Native MRI", "gray"),
        (p_native_expert, "2. Native expert mask", "gray"),
        (p_model_mri, "3. Model-space MRI", "gray"),
        (p_model_expert, "4. Model-space expert (corrected NN)", "gray"),
        (p_level0_activation, "5. Encoder level0 |activation| (RMS/48ch)", "magma"),
        (p_model_prob, "6. TRUE-level0 probability (model space)", "magma"),
        (p_native_prob, "7. Probability mapped to native space", "magma"),
        (p_native_pred, "8. Final thresholded native prediction", "gray"),
    ]
    for a, (arr, title, cmap) in zip(ax.flat, panels):
        a.imshow(arr.T, cmap=cmap, origin="lower", interpolation="nearest")
        a.set_title(title, fontsize=10)
        a.axis("off")
    fig.suptitle(
        f"Mouse {subject}  |  {row['role']}  |  native slice {z}, point ({x},{y})  |  "
        f"native spacing {native_spacing[0]:.4f}x{native_spacing[1]:.4f}x{native_spacing[2]:.4f} mm  |  "
        f"single-tile forward pass (no sliding window; fits: {fits_single_tile})",
        fontsize=11,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig_path = OUT / f"{row['role']}_{subject}.png"
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)

    # --- Quantitative metrics ---
    footprint_pixels, footprint_mm = effective_native_footprint(native_spacing)
    # effective_native_footprint reports the LEVEL1 (2x-coarser) footprint;
    # level0 is the full model grid, so halve the physical footprint back down
    # (level1_model = TARGET_SPACING*2 internally -> level0_model = TARGET_SPACING).
    level0_footprint_pixels = footprint_pixels / 2.0
    level0_footprint_mm = footprint_mm / 2.0

    contour_native_expert = digital_contour_statistics(native_expert[:, :, z])
    # model_expert/model_prob are 3D at TILE_SIZE; slice at the mapped
    # through-plane index (model_point[1], per model_axis_point's convention).
    k = int(np.clip(round(model_point[1]), 0, model_expert.shape[1] - 1))
    contour_model_expert = digital_contour_statistics(model_expert[:, k, :])
    contour_native_pred = digital_contour_statistics(native_pred[:, :, z])
    contour_model_pred = digital_contour_statistics(model_prob[:, k, :] >= 0.5)

    native_iso_pct, native_iso_n = isocontour_subvoxel_percent(native_prob[:, :, z])
    model_iso_pct, model_iso_n = isocontour_subvoxel_percent(model_prob[:, k, :])
    prob_stats_native = probability_detail_statistics(native_prob, native_pred, native_expert)

    quant = quantization_rows("Mouse", subject, native_expert, native_pred, footprint_pixels)
    factor5_y = next(r for r in quant if r["candidate_factor_pixels"] == 5 and r["native_coordinate_axis"] == "y")

    row_metrics = {
        "subject": subject, "role": row["role"],
        "native_spacing_x_mm": native_spacing[0], "native_spacing_y_mm": native_spacing[1], "native_spacing_z_mm": native_spacing[2],
        "model_spacing_mm": f"{TARGET_SPACING[0]:.4f}x{TARGET_SPACING[1]:.4f}x{TARGET_SPACING[2]:.4f}",
        "level0_footprint_x_native_px": level0_footprint_pixels[0], "level0_footprint_y_native_px": level0_footprint_pixels[1], "level0_footprint_z_native_px": level0_footprint_pixels[2],
        "downsample_factor_x": level0_footprint_pixels[0], "downsample_factor_y": level0_footprint_pixels[1], "downsample_factor_z": level0_footprint_pixels[2],
        "native_expert_direction_changes_per_100": contour_native_expert["direction_changes_per_100_steps"] if contour_native_expert else None,
        "model_expert_direction_changes_per_100": contour_model_expert["direction_changes_per_100_steps"] if contour_model_expert else None,
        "native_pred_direction_changes_per_100": contour_native_pred["direction_changes_per_100_steps"] if contour_native_pred else None,
        "model_pred_direction_changes_per_100": contour_model_pred["direction_changes_per_100_steps"] if contour_model_pred else None,
        "native_expert_axis_run_max": contour_native_expert["axis_run_max"] if contour_native_expert else None,
        "model_expert_axis_run_max": contour_model_expert["axis_run_max"] if contour_model_expert else None,
        "native_pred_axis_run_max": contour_native_pred["axis_run_max"] if contour_native_pred else None,
        "model_pred_axis_run_max": contour_model_pred["axis_run_max"] if contour_model_pred else None,
        "factor5_native_y_prediction_minus_expert_alignment_pp": 100 * factor5_y["prediction_minus_expert_alignment"],
        "native_probability_isocontour_subvoxel_percent": native_iso_pct,
        "native_probability_isocontour_points": native_iso_n,
        "model_probability_isocontour_subvoxel_percent": model_iso_pct,
        "model_probability_isocontour_points": model_iso_n,
        "native_boundary_band_intermediate_0.1_0.9_percent": prob_stats_native["boundary_band_intermediate_0.1_0.9_percent"],
        "fits_single_tile": fits_single_tile,
        "probability_axis_permutation": str(prob_permutation),
        "figure_path": str(fig_path),
    }
    metrics_rows.append(row_metrics)
    print(f"{subject} [{row['role']}] slice={z} point=({x},{y}) "
          f"native_iso_subvoxel%={native_iso_pct:.1f} model_iso_subvoxel%={model_iso_pct:.1f} "
          f"native_expert_dc/100={row_metrics['native_expert_direction_changes_per_100']:.1f} "
          f"model_expert_dc/100={row_metrics['model_expert_direction_changes_per_100']:.1f} "
          f"native_pred_dc/100={row_metrics['native_pred_direction_changes_per_100']:.1f} "
          f"model_pred_dc/100={row_metrics['model_pred_direction_changes_per_100']:.1f}", flush=True)


def main():
    start = time.time()
    config = load_json(ROOT / "configs/true_full_resolution_level0_decoder.yaml")
    model, paths = load_model(config)
    per_subject = {r["subject"]: r for r in csv.DictReader(open(LEVEL0 / "per_subject_comparison.csv")) if r["domain"] == "Mouse"}

    metrics_rows = []
    for case in CASES:
        row = dict(per_subject[case["subject"]])
        row["role"] = case["role"]
        run_case(model, paths, row, metrics_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "stage_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0])); w.writeheader(); w.writerows(metrics_rows)

    elapsed = time.time() - start
    print(f"\nTotal elapsed: {elapsed:.1f} s")
    (OUT / "run_meta.json").write_text(json.dumps({"elapsed_seconds": elapsed, "cases": len(metrics_rows)}, indent=2))


if __name__ == "__main__":
    main()
