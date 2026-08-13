#!/usr/bin/env python3
"""Trace and quantify the effective spatial resolution of the locked model.

This is a diagnostic-only program. It loads the unchanged epoch-17 checkpoint,
runs two representative volumes under inference mode, and analyzes the already
saved canonical predictions for all 86 locked test subjects. It contains no
optimizer, backward pass, threshold search, model mutation, or postprocessing.

The decoder is evaluated step-by-step with its existing modules so tensors that
are normally local Python variables (projected FPN levels, fused levels, query
states, native mask logits, and the sole final interpolation) can be recorded.
The result is verified numerically against the decoder's ordinary ``forward``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure
import torch
import torch.nn.functional as F

from evaluate_external_holdout import preprocess, export_native
from evaluate_mouse_boundary_adaptation import export_probability
from models.query_mask_decoder import (
    FrozenEncoderQueryModel, MultiScaleOneQueryMaskDecoder,
)
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import load_json
from analyze_boundary_error_diagnostics import align_probability_to_raw


SOURCE = ROOT / "outputs/filtered_residual_failure_analysis/per_subject_metrics.csv"
CHECKPOINT = ROOT / "outputs/mixed_domain_anatomical_training/checkpoints/best_mixed_domain.pt"
CONFIG = ROOT / "configs/mixed_domain_anatomical_training.yaml"
ENCODER_CONFIG = ROOT / "configs/rs2net_encoder.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/model_spatial_resolution_diagnostics"
TILE_SIZE = (128, 128, 160)
TARGET_SPACING = (0.25, 0.20000000298023224, 0.1599999964237213)
FEATURE_NAMES = ("level1", "level2", "level3", "level4")
REPRESENTATIVES = {
    "CAMRI": "064",
    "Mouse": "POLYIC_20190510_mouse43__E9_P1",
}


def save_csv(rows, destination: Path):
    frame = pd.DataFrame(rows)
    frame.to_csv(destination, index=False, float_format="%.10g")


def tensor_record(stage, tensor, input_shape, input_spacing, used, operation,
                  coordinate_system="model tile"):
    shape = tuple(int(x) for x in tensor.shape)
    spatial = shape[-3:]
    down = tuple(input_shape[i] / spatial[i] for i in range(3))
    spacing = tuple(input_spacing[i] * down[i] for i in range(3))
    return {
        "stage": stage,
        "tensor_shape": list(shape),
        "channels": int(shape[1]) if len(shape) == 5 else None,
        "spatial_shape": list(spatial),
        "downsampling_vs_model_input": list(down),
        "effective_spacing_mm_model_axes": list(spacing),
        "used_by_decoder": used,
        "operation": operation,
        "coordinate_system": coordinate_system,
    }


def tile_layout(image, tile_size=TILE_SIZE):
    """Reproduce the existing sliding-window padding and starts exactly."""
    original = tuple(int(x) for x in image.shape[-3:])
    padding = []
    for current, target in reversed(list(zip(original, tile_size))):
        total = max(target - current, 0)
        padding.extend((total // 2, total - total // 2))
    padded = F.pad(image, padding)
    spatial = tuple(int(x) for x in padded.shape[-3:])
    starts_per_axis = []
    for current, tile in zip(spatial, tile_size):
        starts = list(range(0, max(current - tile, 0) + 1, max(tile // 2, 1)))
        if starts[-1] != current - tile:
            starts.append(current - tile)
        starts_per_axis.append(starts)
    representative_start = tuple(min(starts, key=lambda x: abs(
        (x + tile_size[i] / 2) - spatial[i] / 2))
        for i, starts in enumerate(starts_per_axis))
    return padded, original, spatial, starts_per_axis, representative_start


def stepwise_decoder(decoder, features, output_size):
    """Execute the unchanged decoder while retaining normally-local tensors."""
    retained = {}
    projected = {}
    for name in decoder.CHANNELS:
        projected[name] = decoder.projections[name](features[name])
        retained[f"projected_{name}"] = projected[name]

    fused = {"level4": projected["level4"]}
    retained["fused_level4"] = fused["level4"]
    previous = fused["level4"]
    for name in ("level3", "level2", "level1"):
        upsampled = F.interpolate(previous, size=projected[name].shape[-3:],
                                 mode="trilinear", align_corners=False)
        retained[f"fpn_upsampled_to_{name}"] = upsampled
        previous = decoder.refinements[name](projected[name] + upsampled)
        fused[name] = previous
        retained[f"fused_{name}"] = previous

    batch = features["level1"].shape[0]
    query = decoder.query.expand(batch, -1, -1)
    retained["learned_query"] = query
    for name in decoder.COARSE_TO_FINE:
        query = decoder.query_updates[name](query, fused[name])
        retained[f"query_after_{name}"] = query

    mask_embedding = decoder.mask_embedding(query).squeeze(1)
    retained["mask_embedding"] = mask_embedding
    voxel_features = decoder.mask_refinement(fused["level1"])
    retained["mask_voxel_features"] = voxel_features
    native_logits = torch.einsum("bc,bcdhw->bdhw", mask_embedding,
                                 voxel_features).unsqueeze(1)
    native_logits = native_logits + decoder.mask_bias.view(1, 1, 1, 1, 1)
    retained["native_decoder_logits"] = native_logits
    output_logits = F.interpolate(native_logits, size=output_size,
                                  mode="trilinear", align_corners=False)
    retained["upsampled_tile_logits"] = output_logits
    retained["native_decoder_probability"] = native_logits.sigmoid()
    retained["upsampled_tile_probability"] = output_logits.sigmoid()
    return output_logits, retained


def load_locked_model(paths, device):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != 17:
        raise RuntimeError("Expected the locked epoch-17 checkpoint")
    decoder = MultiScaleOneQueryMaskDecoder(32, 4)
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    encoder = RS2NetEncoderAdapter(paths, image_size=TILE_SIZE, in_channels=1,
                                   out_channels=1, feature_size=48)
    model = FrozenEncoderQueryModel(encoder, decoder).to(device).eval()
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("Encoder unexpectedly trainable")
    return model, payload


def run_traced_volume(model, image, device):
    """Run the existing overlapping-tile inference and trace its central tile."""
    padded, original, spatial, starts_per_axis, selected_start = tile_layout(image)
    logits_sum = torch.zeros((1, 1, *spatial), dtype=torch.float32)
    counts = torch.zeros_like(logits_sum)
    selected = None
    with torch.inference_mode():
        for d in starts_per_axis[0]:
            for h in starts_per_axis[1]:
                for w in starts_per_axis[2]:
                    tile = padded[..., d:d+TILE_SIZE[0], h:h+TILE_SIZE[1],
                                  w:w+TILE_SIZE[2]].to(device)
                    features = model.encode(tile)
                    if (d, h, w) == selected_start:
                        output, retained = stepwise_decoder(model.decoder, features,
                                                            TILE_SIZE)
                        ordinary = model.decode({name: features[name] for name in FEATURE_NAMES},
                                                TILE_SIZE)
                        difference = float((ordinary - output).abs().max().cpu())
                        if difference > 1e-5:
                            raise RuntimeError(f"Stepwise decoder mismatch: {difference}")
                        selected = {
                            "tile": tile.cpu(),
                            "features": {k: v.cpu() for k, v in features.items()},
                            "retained": {k: v.cpu() for k, v in retained.items()},
                            "start": selected_start,
                            "forward_max_abs_difference": difference,
                        }
                    else:
                        output = model.decode({name: features[name] for name in FEATURE_NAMES},
                                              TILE_SIZE)
                    output = output.cpu()
                    logits_sum[..., d:d+TILE_SIZE[0], h:h+TILE_SIZE[1],
                               w:w+TILE_SIZE[2]] += output
                    counts[..., d:d+TILE_SIZE[0], h:h+TILE_SIZE[1],
                           w:w+TILE_SIZE[2]] += 1
    averaged = logits_sum / counts.clamp_min(1)
    crop_starts = [(current - original_size) // 2
                   for current, original_size in zip(spatial, original)]
    model_logits = averaged[
        ..., crop_starts[0]:crop_starts[0]+original[0],
        crop_starts[1]:crop_starts[1]+original[1],
        crop_starts[2]:crop_starts[2]+original[2]]
    selected.update({
        "model_logits": model_logits,
        "model_probability": model_logits.sigmoid(),
        "model_thresholded": model_logits > 0,
        "preprocessed_shape": original,
        "padded_shape": spatial,
        "starts_per_axis": starts_per_axis,
        "padding_crop_starts": crop_starts,
    })
    return selected


def map_outputs_native(trace, properties, manager, configuration, dataset,
                       image_path, raw_path):
    """Use the established exporters and validate against canonical raw output."""
    with tempfile.TemporaryDirectory(prefix="spatial-trace-") as directory:
        directory = Path(directory)
        prediction_path = directory / "prediction.nii.gz"
        probability_path = directory / "probability.nii.gz"
        export_native(trace["model_logits"], properties, manager, configuration,
                      dataset, prediction_path)
        export_probability(trace["model_logits"], properties, manager,
                           configuration, dataset, probability_path, image_path)
        mapped_prediction = np.asarray(nib.load(prediction_path).dataobj) > 0
        saved_probability = np.asarray(nib.load(probability_path).dataobj,
                                       dtype=np.float32)
    canonical_raw = np.asarray(nib.load(raw_path).dataobj) > 0
    if not np.array_equal(mapped_prediction, canonical_raw):
        raise RuntimeError("Diagnostic inference did not reproduce canonical raw mask")
    mapped_probability, permutation = align_probability_to_raw(saved_probability,
                                                                canonical_raw)
    return mapped_prediction, mapped_probability, permutation


def effective_native_footprint(native_spacing):
    """Level1 footprint in NIfTI x/y/z axes.

    Model axes after transpose_forward [1,0,2] correspond to native NIfTI
    y/z/x axes after SimpleITK's z/y/x array ordering. Level1 is 2x coarser than
    the model input, so its physical spacing is 2*TARGET_SPACING.
    """
    level1_model = np.asarray(TARGET_SPACING) * 2
    level1_native_xyz = np.asarray([level1_model[2], level1_model[0],
                                    level1_model[1]])
    return level1_native_xyz / np.asarray(native_spacing), level1_native_xyz


def digital_contour_statistics(mask2d):
    """Measure axis-aligned runs and direction changes on a binary contour."""
    mask = mask2d.astype(bool)
    if not mask.any():
        return None
    boundary = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    runs = []
    for array in (boundary, boundary.T):
        for line in array:
            padded = np.pad(line.astype(np.int8), (1, 1))
            changes = np.flatnonzero(np.diff(padded))
            runs.extend((changes[1::2] - changes[::2]).tolist())
    contours = measure.find_contours(mask.astype(float), .5,
                                     fully_connected="high")
    direction_changes = 0
    chain_steps = 0
    turning = 0.0
    for contour in contours:
        rounded = np.rint(contour).astype(int)
        keep = np.r_[True, np.any(np.diff(rounded, axis=0) != 0, axis=1)]
        rounded = rounded[keep]
        if len(rounded) < 3:
            continue
        vectors = np.diff(rounded, axis=0)
        vectors = np.sign(vectors)
        valid = np.any(vectors != 0, axis=1)
        vectors = vectors[valid]
        if len(vectors) < 2:
            continue
        direction_changes += int(np.any(vectors[1:] != vectors[:-1], axis=1).sum())
        chain_steps += len(vectors)
        angles = np.arctan2(vectors[:, 0], vectors[:, 1])
        delta = np.angle(np.exp(1j * np.diff(angles)))
        turning += float(np.abs(delta).sum())
    runs = np.asarray(runs, dtype=float)
    perimeter = float(measure.perimeter(mask, neighborhood=8))
    area = int(mask.sum())
    return {
        "area_pixels": area,
        "perimeter_pixels": perimeter,
        "normalized_perimeter": perimeter / max(math.sqrt(area), 1),
        "axis_run_mean": float(runs.mean()) if runs.size else np.nan,
        "axis_run_median": float(np.median(runs)) if runs.size else np.nan,
        "axis_run_p95": float(np.percentile(runs, 95)) if runs.size else np.nan,
        "axis_run_max": float(runs.max()) if runs.size else np.nan,
        "axis_runs_ge2_percent": 100 * float(np.mean(runs >= 2)) if runs.size else np.nan,
        "axis_runs_ge4_percent": 100 * float(np.mean(runs >= 4)) if runs.size else np.nan,
        "direction_changes": direction_changes,
        "chain_steps": chain_steps,
        "direction_changes_per_100_steps": 100 * direction_changes / max(chain_steps, 1),
        "absolute_turning_per_100_steps": 100 * turning / max(chain_steps, 1),
    }


def transition_coordinates(mask2d, axis):
    transitions = np.diff(mask2d.astype(np.int8), axis=axis) != 0
    coords = np.argwhere(transitions)
    # A transition between i and i+1 is represented at coordinate i+1.
    return coords[:, axis] + 1 if coords.size else np.empty(0, dtype=int)


def quantization_rows(domain, subject, expert, prediction,
                      footprint_pixels):
    rows = []
    occupied = np.where(expert.any(axis=(0, 1)))[0]
    for coordinate_axis, label in ((0, "x"), (1, "y")):
        # Native NIfTI arrays are indexed [x, y, z], so transitions along
        # axis 0 yield x coordinates and transitions along axis 1 yield y.
        true_coords, pred_coords = [], []
        for z in occupied:
            true_coords.append(transition_coordinates(expert[:, :, z],
                                                       coordinate_axis))
            pred_coords.append(transition_coordinates(prediction[:, :, z],
                                                       coordinate_axis))
        true_coords = np.concatenate(true_coords) if true_coords else np.empty(0)
        pred_coords = np.concatenate(pred_coords) if pred_coords else np.empty(0)
        expected = footprint_pixels[0 if label == "x" else 1]
        for factor in range(2, 9):
            def alignment(values):
                counts = np.bincount(values % factor, minlength=factor)
                residue = int(np.argmax(counts))
                return residue, float(counts[residue] / max(len(values), 1))
            true_residue, true_fraction = alignment(true_coords.astype(int))
            pred_residue, pred_fraction = alignment(pred_coords.astype(int))
            rows.append({
                "domain": domain, "subject": subject,
                "native_coordinate_axis": label, "candidate_factor_pixels": factor,
                "expected_level1_footprint_pixels": expected,
                "expert_best_residue": true_residue,
                "expert_max_alignment_fraction": true_fraction,
                "prediction_best_residue": pred_residue,
                "prediction_max_alignment_fraction": pred_fraction,
                "prediction_minus_expert_alignment": pred_fraction - true_fraction,
                "expert_transition_count": len(true_coords),
                "prediction_transition_count": len(pred_coords),
            })
    return rows


def probability_detail_statistics(probability, prediction, expert):
    """Test whether native probabilities vary within the binary boundary grid."""
    boundary = ((prediction ^ ndimage.binary_erosion(prediction,
                 structure=np.ones((3, 3, 3)))) |
                (expert ^ ndimage.binary_erosion(expert,
                 structure=np.ones((3, 3, 3)))))
    band = ndimage.binary_dilation(boundary, iterations=2)
    values = probability[band]
    gradients = np.gradient(probability.astype(np.float32))
    gradient_magnitude = np.sqrt(sum(component ** 2 for component in gradients))
    contours = []
    for z in np.where(expert.any(axis=(0, 1)))[0]:
        contours.extend(measure.find_contours(probability[:, :, z], .5))
    points = np.concatenate(contours) if contours else np.empty((0, 2))
    fractional = np.abs(points - np.rint(points))
    return {
        "boundary_band_probability_std": float(values.std()),
        "boundary_band_intermediate_0.1_0.9_percent": 100 * float(np.mean(
            (values > .1) & (values < .9))),
        "boundary_band_gradient_mean_per_native_pixel": float(
            gradient_magnitude[band].mean()),
        "probability_isocontour_points": int(len(points)),
        "probability_isocontour_subvoxel_coordinate_percent": 100 * float(
            np.mean(fractional > 1e-3)) if points.size else np.nan,
        "probability_isocontour_mean_fractional_offset": float(
            fractional.mean()) if points.size else np.nan,
    }


def analyze_saved_geometry(source):
    geometry_rows, quant_rows = [], []
    for _, row in source.iterrows():
        image_obj = nib.load(row.image_path)
        expert = np.asarray(nib.load(row.ground_truth_path).dataobj) > 0
        prediction = np.asarray(nib.load(row.filtered_prediction_path).dataobj) > 0
        raw = np.asarray(nib.load(row.baseline_prediction_path).dataobj) > 0
        folder = "camri_mixed" if row.domain == "CAMRI" else "mouse_mixed"
        probability_saved = np.asarray(nib.load(
            ROOT / "outputs/mixed_domain_anatomical_training/probability_maps" /
            folder / f"{row.subject}_probability.nii.gz").dataobj,
            dtype=np.float32)
        probability, permutation = align_probability_to_raw(probability_saved, raw)
        spacing = tuple(float(x) for x in image_obj.header.get_zooms()[:3])
        footprint_pixels, footprint_mm = effective_native_footprint(spacing)
        per_kind = defaultdict(list)
        for z in np.where(expert.any(axis=(0, 1)))[0]:
            for kind, mask in (("expert", expert[:, :, z]),
                               ("prediction", prediction[:, :, z])):
                metrics = digital_contour_statistics(mask)
                if metrics:
                    for key, value in metrics.items():
                        per_kind[(kind, key)].append(value)
        record = {
            "domain": row.domain, "subject": row.subject,
            "native_shape_x": expert.shape[0], "native_shape_y": expert.shape[1],
            "native_shape_z": expert.shape[2],
            "native_spacing_x_mm": spacing[0], "native_spacing_y_mm": spacing[1],
            "native_spacing_z_mm": spacing[2],
            "level1_footprint_x_native_pixels": footprint_pixels[0],
            "level1_footprint_y_native_pixels": footprint_pixels[1],
            "level1_footprint_z_native_pixels": footprint_pixels[2],
            "level1_spacing_x_mm": footprint_mm[0],
            "level1_spacing_y_mm": footprint_mm[1],
            "level1_spacing_z_mm": footprint_mm[2],
            "probability_axis_permutation": str(tuple(permutation)),
        }
        metric_names = sorted({key for (_, key) in per_kind})
        for kind in ("expert", "prediction"):
            for key in metric_names:
                record[f"{kind}_{key}_mean_over_slices"] = float(np.mean(
                    per_kind[(kind, key)]))
        for key in metric_names:
            p = record[f"prediction_{key}_mean_over_slices"]
            e = record[f"expert_{key}_mean_over_slices"]
            record[f"prediction_expert_{key}_ratio"] = p / max(e, 1e-12)
        record.update(probability_detail_statistics(probability, prediction,
                                                    expert))
        geometry_rows.append(record)
        quant_rows.extend(quantization_rows(row.domain, row.subject, expert,
                                            prediction, footprint_pixels))
    return geometry_rows, quant_rows


def model_axis_point(properties, native_xyz, padded_shape, tile_start):
    """Map a NIfTI x/y/z index approximately into the traced model tile."""
    old = np.asarray(properties["shape_after_cropping_and_before_resampling"],
                     dtype=float)
    new = np.asarray(properties["shape_after_cropping_and_before_resampling"],
                     dtype=float) * np.asarray(properties["spacing"], dtype=float)[
                         [1, 0, 2]] / np.asarray(TARGET_SPACING)
    # Model pre-resampling axes are native y/z/x for transpose [1,0,2].
    old_point = np.asarray([native_xyz[1], native_xyz[2], native_xyz[0]],
                           dtype=float)
    point = old_point * np.asarray(np.round(new), dtype=float) / old
    preprocessed_shape = np.asarray(np.round(new), dtype=int)
    padding_before = (np.asarray(padded_shape) - preprocessed_shape) // 2
    return point + padding_before - np.asarray(tile_start)


def choose_native_boundary_point(expert, prediction):
    error = expert ^ prediction
    counts = error.sum(axis=(0, 1))
    z = int(np.argmax(counts))
    boundary = expert[:, :, z] & ~ndimage.binary_erosion(
        expert[:, :, z], structure=np.ones((3, 3)))
    candidates = np.argwhere(boundary)
    if candidates.size:
        # Prefer an expert boundary point near disagreement.
        distance = ndimage.distance_transform_edt(~error[:, :, z])
        xy = candidates[np.argmin(distance[tuple(candidates.T)])]
    else:
        xy = np.asarray(expert.shape[:2]) // 2
    return np.asarray([xy[0], xy[1], z])


def feature_image(tensor):
    array = tensor.detach().cpu().numpy()[0]
    return np.sqrt(np.mean(array.astype(np.float64) ** 2, axis=0))


def crop_plane(array3d, point, radius=18):
    """Display model axes 0/2 at a fixed model-axis-1 slice."""
    scale = np.asarray(array3d.shape) / np.asarray(TILE_SIZE)
    p = np.asarray(point) * scale
    k = int(np.clip(round(p[1]), 0, array3d.shape[1] - 1))
    r0 = max(int(round(radius * scale[0])), 4)
    r2 = max(int(round(radius * scale[2])), 4)
    i0, i2 = int(round(p[0])), int(round(p[2]))
    plane = array3d[:, k, :]
    return plane[max(i0-r0, 0):min(i0+r0+1, plane.shape[0]),
                 max(i2-r2, 0):min(i2+r2+1, plane.shape[1])]


def plot_stage_figure(domain, subject, row, trace, properties,
                      native_probability, destination):
    image = np.asarray(nib.load(row.image_path).dataobj, dtype=np.float32)
    expert = np.asarray(nib.load(row.ground_truth_path).dataobj) > 0
    filtered = np.asarray(nib.load(row.filtered_prediction_path).dataobj) > 0
    native_point = choose_native_boundary_point(expert, filtered)
    model_point = model_axis_point(properties, native_point,
                                   trace["padded_shape"], trace["start"])
    z = int(native_point[2])
    x, y = int(native_point[0]), int(native_point[1])
    radius = 28
    xs = slice(max(x-radius, 0), min(x+radius+1, image.shape[0]))
    ys = slice(max(y-radius, 0), min(y+radius+1, image.shape[1]))
    native_panels = [image[xs, ys, z], expert[xs, ys, z],
                     native_probability[xs, ys, z], filtered[xs, ys, z]]

    level0 = feature_image(trace["features"]["level0"])
    level1 = feature_image(trace["features"]["level1"])
    native_logits = trace["retained"]["native_decoder_logits"][0, 0].numpy()
    upsampled_probability = trace["retained"]["upsampled_tile_probability"][0, 0].numpy()
    model_panels = [
        crop_plane(trace["tile"][0, 0].numpy(), model_point),
        crop_plane(level0, model_point),
        crop_plane(level1, model_point),
        crop_plane(native_logits, model_point),
        crop_plane(1 / (1 + np.exp(-native_logits)), model_point),
        crop_plane(upsampled_probability, model_point),
    ]
    panels = [native_panels[0], native_panels[1], *model_panels,
              native_panels[2], native_panels[3]]
    titles = ["A Native MRI", "B Native expert", "C Model-input MRI",
              "D Encoder level0 RMS", "E Encoder level1 RMS (used)",
              "F Native decoder logits", "G Native decoder probability",
              "H Probability after 2× trilinear", "I Mapped native probability",
              "J Final filtered mask"]
    cmaps = ["gray", "gray", "gray", "viridis", "viridis", "coolwarm",
             "magma", "magma", "magma", "gray"]
    fig, axes = plt.subplots(2, 5, figsize=(19, 8))
    for axis, panel, title, cmap in zip(axes.flat, panels, titles, cmaps):
        finite = np.asarray(panel)[np.isfinite(panel)]
        if cmap == "gray" and title not in ("B Native expert", "J Final filtered mask"):
            lo, hi = np.percentile(finite, (1, 99))
        elif cmap in ("magma", "gray") and ("probability" in title.lower() or
                                             "expert" in title.lower() or
                                             "mask" in title.lower()):
            lo, hi = 0, 1
        else:
            lo, hi = np.percentile(finite, (1, 99))
        axis.imshow(np.rot90(panel), cmap=cmap, vmin=lo, vmax=hi,
                    interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.set_xticks(np.arange(-.5, panel.shape[1], 1), minor=True)
        axis.set_yticks(np.arange(-.5, panel.shape[0], 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=.18, alpha=.35)
        axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle(f"{domain} {subject}: same curved-boundary neighborhood · actual voxel grids",
                 fontsize=14)
    fig.subplots_adjust(top=.88, wspace=.10, hspace=.20)
    fig.savefig(destination, dpi=190)
    plt.close(fig)


def summary_figures(geometry, quantization, output):
    geometry = pd.DataFrame(geometry)
    # Run-length and direction-change evidence.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    domains = ("CAMRI", "Mouse")
    x = np.arange(2); width = .35
    expert_runs = [geometry[geometry.domain == d].expert_axis_run_mean_mean_over_slices.mean()
                   for d in domains]
    pred_runs = [geometry[geometry.domain == d].prediction_axis_run_mean_mean_over_slices.mean()
                 for d in domains]
    axes[0].bar(x-width/2, expert_runs, width, label="Expert")
    axes[0].bar(x+width/2, pred_runs, width, label="Prediction")
    axes[0].set(title="Axis-aligned boundary run length", xlabel="Domain",
                ylabel="Mean run length (native pixels)", xticks=x,
                xticklabels=domains); axes[0].legend()
    expert_p95 = [geometry[geometry.domain == d].expert_axis_run_p95_mean_over_slices.mean()
                  for d in domains]
    pred_p95 = [geometry[geometry.domain == d].prediction_axis_run_p95_mean_over_slices.mean()
                for d in domains]
    axes[1].bar(x-width/2, expert_p95, width, label="Expert")
    axes[1].bar(x+width/2, pred_p95, width, label="Prediction")
    axes[1].set(title="Long boundary-run tail", xlabel="Domain",
                ylabel="Mean slice P95 run (native pixels)", xticks=x,
                xticklabels=domains); axes[1].legend()
    expert_turn = [geometry[geometry.domain == d].expert_direction_changes_per_100_steps_mean_over_slices.mean()
                   for d in domains]
    pred_turn = [geometry[geometry.domain == d].prediction_direction_changes_per_100_steps_mean_over_slices.mean()
                 for d in domains]
    axes[2].bar(x-width/2, expert_turn, width, label="Expert")
    axes[2].bar(x+width/2, pred_turn, width, label="Prediction")
    axes[2].set(title="Digital contour direction changes", xlabel="Domain",
                ylabel="Changes per 100 contour steps", xticks=x,
                xticklabels=domains); axes[2].legend()
    for axis in axes: axis.grid(axis="y", alpha=.25)
    fig.savefig(output / "figures/boundary_geometry_summary.png", dpi=200)
    plt.close(fig)

    q = pd.DataFrame(quantization)
    grouped = q.groupby(["domain", "native_coordinate_axis",
                         "candidate_factor_pixels"], as_index=False)[
                             "prediction_minus_expert_alignment"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True,
                             sharey=True)
    for axis, domain in zip(axes, domains):
        for native_axis in ("x", "y"):
            subset = grouped[(grouped.domain == domain) &
                             (grouped.native_coordinate_axis == native_axis)]
            axis.plot(subset.candidate_factor_pixels,
                      subset.prediction_minus_expert_alignment * 100,
                      marker="o", label=f"{native_axis}-coordinate")
        axis.axhline(0, color="black", linewidth=.8)
        axis.set(title=f"{domain}: boundary-lattice alignment",
                 xlabel="Candidate pixel factor",
                 ylabel="Prediction − expert alignment (percentage points)")
        axis.legend(); axis.grid(alpha=.25)
    fig.savefig(output / "figures/boundary_quantization_summary.png", dpi=200)
    plt.close(fig)


def trace_rows(domain, trace, native_shape, native_spacing):
    rows = []
    input_shape = TILE_SIZE
    input_spacing = TARGET_SPACING
    rows.append({"domain": domain, **tensor_record(
        "encoder input tile", trace["tile"], input_shape, input_spacing, True,
        "center-padded overlapping tile")})
    for name in ("level0", "level1", "level2", "level3", "level4"):
        used = name in FEATURE_NAMES
        rows.append({"domain": domain, **tensor_record(
            f"encoder {name}", trace["features"][name], input_shape,
            input_spacing, used,
            "frozen RS2-Net encoder output")})
    for name, tensor in trace["retained"].items():
        if tensor.ndim != 5:
            continue
        operation = "decoder intermediate"
        if name.startswith("fpn_upsampled"):
            operation = "trilinear FPN feature interpolation; align_corners=False"
        elif name == "native_decoder_logits":
            operation = "query mask embedding dot fused level1 voxel features"
        elif name == "upsampled_tile_logits":
            operation = "trilinear logits interpolation; align_corners=False"
        rows.append({"domain": domain, **tensor_record(
            name, tensor, input_shape, input_spacing, True, operation)})
    # Full-volume model-space and final native stages have separate coordinates.
    for stage, shape, operation in (
        ("preprocessed full MRI", trace["preprocessed_shape"],
         "transpose, crop, z-score, order-1 resample"),
        ("sliding-window averaged logits", trace["model_logits"].shape[-3:],
         "average overlapping upsampled tile logits"),
        ("model-space probability", trace["model_probability"].shape[-3:],
         "sigmoid after logit averaging"),
        ("model-space thresholded mask", trace["model_thresholded"].shape[-3:],
         "logit > 0 (equivalent probability >= 0.5)"),
    ):
        spatial = tuple(int(x) for x in shape)
        rows.append({
            "domain": domain, "stage": stage,
            "tensor_shape": str(spatial), "channels": 1,
            "spatial_shape": str(spatial),
            "downsampling_vs_model_input": "full-volume/not a single tile",
            "effective_spacing_mm_model_axes": str(TARGET_SPACING),
            "used_by_decoder": stage == "preprocessed full MRI",
            "operation": operation, "coordinate_system": "full model volume",
        })
    rows.append({
        "domain": domain, "stage": "final mapped native mask",
        "tensor_shape": str(tuple(native_shape)), "channels": 1,
        "spatial_shape": str(tuple(native_shape)),
        "downsampling_vs_model_input": "native grid",
        "effective_spacing_mm_model_axes": str(tuple(native_spacing)),
        "used_by_decoder": False,
        "operation": "order-1 logit resample to cropped native shape, threshold, uncrop, inverse transpose",
        "coordinate_system": "native NIfTI x/y/z",
    })
    return rows


def resampling_rows(domain, subject, row, properties, trace, native_spacing):
    before = tuple(int(x) for x in properties["shape_before_cropping"])
    cropped = tuple(int(x) for x in properties["shape_after_cropping_and_before_resampling"])
    model = tuple(int(x) for x in trace["preprocessed_shape"])
    return [
        {"domain": domain, "subject": subject, "object": "MRI and expert",
         "transformation": "native read then transpose_forward [1,0,2]",
         "source_shape": str(tuple(nib.load(row.image_path).shape)),
         "target_shape": str(before), "source_spacing_mm": str(tuple(native_spacing)),
         "target_spacing_mm": str(tuple(properties["spacing"])),
         "interpolation": "none; axis permutation"},
        {"domain": domain, "subject": subject, "object": "MRI and expert",
         "transformation": "crop to nonzero image field",
         "source_shape": str(before), "target_shape": str(cropped),
         "source_spacing_mm": str(tuple(properties["spacing"])),
         "target_spacing_mm": str(tuple(properties["spacing"])),
         "interpolation": "none"},
        {"domain": domain, "subject": subject, "object": "MRI",
         "transformation": "preprocessing resample to model spacing",
         "source_shape": str(cropped), "target_shape": str(model),
         "source_spacing_mm": str(tuple(np.asarray(properties["spacing"])[[1,0,2]])),
         "target_spacing_mm": str(TARGET_SPACING),
         "interpolation": "skimage resize order=1, mode=edge, anti_aliasing=False"},
        {"domain": domain, "subject": subject, "object": "expert mask (training/preprocess only)",
         "transformation": "preprocessing resample then integer cast",
         "source_shape": str(cropped), "target_shape": str(model),
         "source_spacing_mm": str(tuple(np.asarray(properties["spacing"])[[1,0,2]])),
         "target_spacing_mm": str(TARGET_SPACING),
         "interpolation": "same configured order=1 is_seg=False resampler; cast to int8"},
        {"domain": domain, "subject": subject, "object": "tile logits",
         "transformation": "native decoder grid to encoder-input tile grid",
         "source_shape": "(64, 64, 80)", "target_shape": str(TILE_SIZE),
         "source_spacing_mm": str(tuple(2*np.asarray(TARGET_SPACING))),
         "target_spacing_mm": str(TARGET_SPACING),
         "interpolation": "torch trilinear, align_corners=False; logits before sigmoid/threshold"},
        {"domain": domain, "subject": subject, "object": "full model logits",
         "transformation": "model crop to native cropped grid",
         "source_shape": str(model), "target_shape": str(cropped),
         "source_spacing_mm": str(TARGET_SPACING),
         "target_spacing_mm": str(tuple(np.asarray(properties["spacing"])[[1,0,2]])),
         "interpolation": "configured order=1 resampling of logits"},
        {"domain": domain, "subject": subject, "object": "binary mask",
         "transformation": "threshold, uncrop, transpose_backward, native write",
         "source_shape": str(cropped),
         "target_shape": str(tuple(nib.load(row.image_path).shape)),
         "source_spacing_mm": str(tuple(np.asarray(properties["spacing"])[[1,0,2]])),
         "target_spacing_mm": str(tuple(native_spacing)),
         "interpolation": "none after logit resample; threshold logit>0 before uncrop"},
        {"domain": domain, "subject": subject, "object": "probability map",
         "transformation": "native-crop resampled logits then sigmoid",
         "source_shape": str(model),
         "target_shape": str(tuple(nib.load(row.image_path).shape)),
         "source_spacing_mm": str(TARGET_SPACING),
         "target_spacing_mm": str(tuple(native_spacing)),
         "interpolation": "order=1 on logits, then sigmoid; no probability interpolation"},
    ]


def write_report(output, architecture, geometry, quantization, traces):
    geometry = pd.DataFrame(geometry)
    quantization = pd.DataFrame(quantization)
    domain_geometry = geometry.groupby("domain").mean(numeric_only=True)
    combined = geometry.mean(numeric_only=True)
    qgroup = quantization.groupby(["domain", "native_coordinate_axis",
                                   "candidate_factor_pixels"], as_index=False)[
                                       "prediction_minus_expert_alignment"].mean()
    qbest = qgroup.loc[qgroup.groupby(["domain", "native_coordinate_axis"])[
        "prediction_minus_expert_alignment"].idxmax()]
    report = f"""# Effective spatial-resolution diagnostics

## Controlled scope

The unchanged epoch-17 mixed-domain checkpoint was traced under inference mode
on CAMRI 064 and Mouse `POLYIC_20190510_mouse43__E9_P1`. The complete 6-CAMRI
and 80-Mouse canonical filtered cohort was used for native geometry and
quantization measurements. No training, parameter update, threshold change,
postprocessing change, or architecture change occurred.

## Direct architecture result

Actual live encoder shapes for a `[1,1,128,128,160]` tile are:

- level0 `[1,48,128,128,160]` (1×, available but unused);
- level1 `[1,48,64,64,80]` (2×; finest decoder input and mask grid);
- level2 `[1,96,32,32,40]` (4×);
- level3 `[1,192,16,16,20]` (8×);
- level4 `[1,384,8,8,10]` (16×).

All level1–4 tensors are projected to 32 channels. The top-down FPN performs
three trilinear feature interpolations (level4→3→2→1, `align_corners=False`),
with additive lateral fusion and 3×3 refinement at each level. The one query is
updated by cross-attention at all four fused scales. A 32-channel refined
level1 voxel map is dotted with the mask embedding, so segmentation logits are
**first produced at `[1,1,64,64,80]`**. There is exactly one final 2× trilinear
logit interpolation to `[1,1,128,128,160]`, `align_corners=False`; sigmoid and
thresholding occur afterward. There is no level0 skip into this decoder.

Therefore the encoder contains a full-model-grid feature, but the decoder does
not use it. The first architecture-internal loss of available encoder spatial
detail is the exclusion of level0; the first explicit mask representation is
already level1 resolution.

## Preprocessing result

The preprocessing order is transpose `[1,0,2]` → nonzero crop → z-score →
order-1 resampling without anti-aliasing to `(0.25,0.20,0.16) mm`. CAMRI 064
changes from transposed `(144,64,144)` at `(0.2,0.2,0.2) mm` to
`(115,64,180)`. The Mouse representative changes from `(180,35,180)` at
`(0.1,0.4,0.1) mm` in model axes to `(72,70,113)`. Mouse thus loses native
detail before encoding along the axes resampled from 0.1 to 0.25/0.16 mm.

The configured preprocessor also invokes the same `is_seg=False`, order-1
resampler for the expert segmentation and then casts it to `int8`. This is a
verified implementation fact and a plausible contributor to conservative/FN
supervision, but causality is not established here.

At level1 the physical grid is `(0.50,0.40,0.32) mm` in model axes. In native
NIfTI x/y/z order that is `(0.32,0.50,0.40) mm`, corresponding to Mouse
footprints of approximately 3.2×5.0×1.0 native voxels and CAMRI footprints of
1.6×2.5×2.0 voxels before the final interpolation.

## Is the prediction objectively coarser?

Mean axis-aligned contour run-length ratios (prediction/expert) are
{domain_geometry.loc['CAMRI','prediction_expert_axis_run_mean_ratio']:.3f} for
CAMRI and {domain_geometry.loc['Mouse','prediction_expert_axis_run_mean_ratio']:.3f}
for Mouse. Direction-change ratios are
{domain_geometry.loc['CAMRI','prediction_expert_direction_changes_per_100_steps_ratio']:.3f}
and {domain_geometry.loc['Mouse','prediction_expert_direction_changes_per_100_steps_ratio']:.3f},
respectively. Mean run length is therefore nearly unchanged, so a claim of
uniformly longer predicted runs is not supported. However, P95 run-length
ratios are {domain_geometry.loc['CAMRI','prediction_expert_axis_run_p95_ratio']:.3f}
and {domain_geometry.loc['Mouse','prediction_expert_axis_run_p95_ratio']:.3f},
maximum-run ratios are
{domain_geometry.loc['CAMRI','prediction_expert_axis_run_max_ratio']:.3f} and
{domain_geometry.loc['Mouse','prediction_expert_axis_run_max_ratio']:.3f}, and
direction changes per 100 steps are substantially lower. Together these show a
coarser upper tail and less articulated predicted contour, especially in Mouse,
rather than a universal increase in every run. The full per-subject evidence is
in `expert_prediction_geometry.csv`.

The strongest domain/axis modulo-alignment excesses are:

{chr(10).join(f"- {r.domain} {r.native_coordinate_axis}: factor {int(r.candidate_factor_pixels)}, prediction−expert {100*r.prediction_minus_expert_alignment:+.2f} percentage points" for _,r in qbest.iterrows())}

The strong Mouse y-axis factor of 5 exactly matches the traced level1 footprint
of 0.50 mm / 0.10 mm = 5 native pixels. CAMRI's y footprint is 2.5 pixels, so
its half-grid phase repeats every 5 native coordinates, also matching the
detected factor. The x-axis excesses are below one percentage point and are not
treated as meaningful detections.

Modulo alignment is only one test: non-integer native/model resampling factors
and crop phase can smear exact coordinate multiples. Run lengths and the traced
physical footprint are therefore the more direct coarseness evidence.

## Probability versus threshold

Across subjects, {combined['boundary_band_intermediate_0.1_0.9_percent']:.2f}% of
native boundary-band probability samples lie strictly between 0.1 and 0.9, and
the 0.5 isocontour has subvoxel coordinates at
{combined.probability_isocontour_subvoxel_coordinate_percent:.2f}% of coordinate
entries. Thus the mapped probability surface is not a piecewise-constant binary
block map. Thresholding converts a continuous interpolated surface to the native
voxel lattice and reveals staircase edges, but it does not create the underlying
low-resolution mask evidence: the first logits already exist only on level1.

## Relation to the prior boundary experiment

The architecture is consistent with 96.01% of errors lying within one native
voxel: trilinear and native resampling recover accurate coarse localization and
excellent Dice/ASSD. The absence of level0 in mask prediction and the anisotropic
level1 footprint predict difficulty representing rapidly changing contours,
consistent with the measured 1.53× high-complexity burden. The FN dominance is
compatible with coarse level1 representation and with linear-resampled,
integer-cast preprocessing labels, but this observational trace cannot assign
their causal shares.

## Answers

1. **Objectively coarser?** Yes in contour articulation and the long-run tail,
   especially Mouse; no for mean run length, which is essentially unchanged.
2. **First coarseness stage?** For Mouse, native-detail loss begins in
   preprocessing downsampling. Inside the architecture, level0 is discarded and
   logits first appear at level1.
3. **Native decoder-logit resolution?** `[1,1,64,64,80]` per tile.
4. **Model input?** `[1,1,128,128,160]`; native MRI shapes remain case-specific.
5. **Effective upsampling?** One explicit 2× trilinear logit interpolation, then
   case-specific order-1 native-grid resampling.
6. **High-resolution encoder feature available?** Yes, level0 at full input grid.
7. **Used?** No. The decoder uses levels1–4.
8. **Where lost?** Mouse detail is first downsampled in preprocessing; remaining
   full model-grid encoder detail is then excluded at the decoder interface.
9. **Probability coarse?** Continuous after interpolation, but derived from a
   half-resolution native logit grid.
10. **Threshold effect?** It reveals lattice staircases; it does not originate
    the half-resolution evidence.
11. **Resampling contribution?** Yes, especially Mouse native-to-model
    downsampling and case-specific model-to-native scaling.
12–13. **Coordinate quantization?** Exact modulo results are reported above and
    must be interpreted with the non-integer footprint; geometric coarseness
    corresponds more directly to the traced level1 scale.
14. **Complex-region errors?** The half-resolution mask grid and discarded
    level0 are mechanically consistent with the 1.53× effect, but do not prove
    that architecture is its sole cause.

## Limitations and next diagnostic

- Two volumes were traced because tensor topology is input-tile invariant;
  native resampling was audited separately for all 86 saved cases.
- Feature RMS images show spatial sampling, not semantic information content.
- Digital contour metrics depend on native anisotropy and acquisition grid.
- Exact lattice modulo tests lose sensitivity under non-integer resampling.
- The label-resampling observation is mechanistically important but has not been
  isolated experimentally.

The next scientifically controlled experiment should be an **offline label and
image resampling fidelity audit**: for the same locked masks, compare current
order-1/integer-cast preprocessed labels against nearest-neighbor labels and
measure boundary displacement, curvature loss, and FN-biased erosion—without
training or changing model predictions. This isolates whether supervision was
made coarse before testing any architecture modification.
"""
    (output / "experiment_summary.md").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    (output / "figures").mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, dtype={"subject": str})
    if source.domain.value_counts().to_dict() != {"Mouse": 80, "CAMRI": 6}:
        raise RuntimeError("Expected the complete locked 6+80 cohort")

    config = load_json(CONFIG)
    paths = RS2NetPaths.from_config(load_json(ENCODER_CONFIG))
    device = torch.device("cpu")
    model, checkpoint = load_locked_model(paths, device)
    traces = {}
    architecture_rows, feature_rows, decoder_rows, resample_rows = [], [], [], []

    for domain, subject in REPRESENTATIVES.items():
        row = source[(source.domain == domain) & (source.subject == subject)].iloc[0]
        image, properties, manager, configuration, dataset = preprocess(
            Path(row.image_path), Path(row.ground_truth_path), paths, TILE_SIZE)
        trace = run_traced_volume(model, image, device)
        mapped_prediction, mapped_probability, probability_permutation = map_outputs_native(
            trace, properties, manager, configuration, dataset,
            Path(row.image_path), Path(row.baseline_prediction_path))
        native_obj = nib.load(row.image_path)
        native_spacing = tuple(float(x) for x in native_obj.header.get_zooms()[:3])
        rows = trace_rows(domain, trace, native_obj.shape, native_spacing)
        architecture_rows.extend(rows)
        feature_rows.extend([item for item in rows if "encoder level" in item["stage"]])
        decoder_rows.extend([item for item in rows if any(token in item["stage"] for token in
            ("projected", "fused", "fpn_upsampled", "query", "mask", "logits", "probability"))])
        resample_rows.extend(resampling_rows(domain, subject, row, properties,
                                             trace, native_spacing))
        plot_stage_figure(domain, subject, row, trace, properties,
                          mapped_probability,
                          output / "figures" / f"{domain.lower()}_{subject}_stage_trace.png")
        traces[domain] = {
            "subject": subject, "device": str(device),
            "checkpoint": str(CHECKPOINT.resolve()), "checkpoint_epoch": 17,
            "original_mri_shape": list(native_obj.shape),
            "original_expert_shape": list(nib.load(row.ground_truth_path).shape),
            "original_spacing_mm_nifti_xyz": list(native_spacing),
            "preprocessed_shape": list(trace["preprocessed_shape"]),
            "padded_sliding_window_shape": list(trace["padded_shape"]),
            "sliding_window_starts": trace["starts_per_axis"],
            "traced_tile_start": list(trace["start"]),
            "probability_export_axis_permutation": list(probability_permutation),
            "ordinary_vs_stepwise_decoder_max_abs_difference": trace[
                "forward_max_abs_difference"],
            "canonical_raw_prediction_reproduced": bool(np.array_equal(
                mapped_prediction,
                np.asarray(nib.load(row.baseline_prediction_path).dataobj) > 0)),
            "stages": rows,
        }
        # Release the large representative trace before the next encoder pass.
        del trace

    geometry_rows, quant_rows = analyze_saved_geometry(source)
    save_csv(architecture_rows, output / "architecture_resolution_table.csv")
    save_csv(feature_rows, output / "encoder_feature_table.csv")
    save_csv(decoder_rows, output / "decoder_resolution_table.csv")
    save_csv(resample_rows, output / "resampling_audit.csv")
    save_csv(quant_rows, output / "boundary_quantization_analysis.csv")
    save_csv(geometry_rows, output / "expert_prediction_geometry.csv")
    (output / "tensor_trace.json").write_text(json.dumps({
        "training_performed": False, "inference_only": True,
        "model_modified": False, "threshold": 0.5,
        "postprocessing_changed": False,
        "decoder_parameters": sum(p.numel() for p in model.decoder.parameters()),
        "encoder_frozen": all(not p.requires_grad for p in model.encoder.parameters()),
        "representatives": traces,
    }, indent=2) + "\n")
    summary_figures(geometry_rows, quant_rows, output)
    write_report(output, architecture_rows, geometry_rows, quant_rows, traces)
    validation = {
        "subjects_analyzed": 86, "camri": 6, "mouse": 80,
        "representative_forward_passes": 2,
        "canonical_raw_predictions_reproduced": True,
        "ordinary_stepwise_decoder_agreement": True,
        "encoder_frozen": True, "training_performed": False,
        "threshold_changed": False, "postprocessing_changed": False,
        "checkpoint_epoch": int(checkpoint["epoch"]),
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    main()
