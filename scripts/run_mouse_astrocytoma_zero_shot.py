#!/usr/bin/env python3
"""Qualitative-only zero-shot inference on four pathology pilot series.

This script deliberately has no optimizer, training loop, backward call, label
input, or tunable post-processing.  It loads the validation-selected mixed-
domain checkpoint strictly, applies the established 0-logit threshold, and
retains the established largest 26-connected component as a separate output.
All cavity and indentation measurements are analysis annotations; they never
alter either prediction.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
import torch
from matplotlib.animation import FFMpegWriter
from scipy import ndimage
from skimage import measure

from evaluate_external_holdout import export_native, preprocess, sliding_window_logits
from evaluate_mouse_boundary_adaptation import export_probability
from models.query_conditioned_grouping import largest_component_3d
from models.query_mask_decoder import FrozenEncoderQueryModel, MultiScaleOneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from prepare_mouse_astrocytoma_zero_shot import (
    first_spatial_volume, load_volume, read_csv, robust_normalize, series_headers,
)
from train_query_decoder_overfit import choose_device, load_json


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_nifti(records: list[dict], destination: Path) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct one spatial volume and its DICOM patient-space affine.

    Array axes are rows, columns, slices.  DICOM's first orientation vector
    follows increasing columns; its second follows increasing rows.  The LPS
    affine is converted to NIfTI's RAS convention by negating x and y.
    """
    selected, repetitions = first_spatial_volume(records)
    if repetitions != 1:
        raise ValueError("pilot is not a single spatial volume")
    volume = load_volume(selected)
    ds = pydicom.dcmread(str(selected[0]["path"]), stop_before_pixels=True)
    orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
    row_dir, column_dir = orientation[:3], orientation[3:]
    positions = np.asarray([r["image_position"] for r in selected], dtype=float)
    if len(selected) > 1:
        slice_step = (positions[-1] - positions[0]) / (len(selected) - 1)
    else:
        slice_step = np.cross(row_dir, column_dir) * float(ds.SliceThickness)
    spacing = np.asarray(ds.PixelSpacing, dtype=float)
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = column_dir * spacing[0]
    affine_lps[:3, 1] = row_dir * spacing[1]
    affine_lps[:3, 2] = slice_step
    affine_lps[:3, 3] = positions[0]
    affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine_ras), destination)
    return volume, affine_ras


def save_zero_reference(image_path: Path, destination: Path) -> None:
    """Create a geometry-only placeholder required by the released preprocessor.

    It is not a ground truth mask.  ``crop_to_nonzero`` recreates the same
    image-derived normalization mask used when ``seg_file=None``.  The released
    preprocessor has a final ``np.max(seg)`` bug for None, so a zero geometry
    reference is used without changing baseline code or preprocessing logic.
    """
    image = nib.load(str(image_path))
    nib.save(nib.Nifti1Image(np.zeros(image.shape, np.uint8), image.affine, image.header), destination)


def save_preprocessed(image: torch.Tensor, destination: Path) -> None:
    array = image[0, 0].cpu().numpy().astype(np.float32)
    nib.save(nib.Nifti1Image(array, np.eye(4)), destination)


def dense_pages(volume, destination: Path, title: str, mask=None, probability=None, per_page=24):
    """Save every slice, never a representative subset."""
    normalized = robust_normalize(volume)
    destination.mkdir(parents=True, exist_ok=True)
    for start in range(0, volume.shape[2], per_page):
        indices = list(range(start, min(start + per_page, volume.shape[2])))
        figure, axes = plt.subplots(4, 6, figsize=(12, 8), constrained_layout=True)
        for axis, index in zip(axes.flat, indices):
            axis.imshow(normalized[:, :, index].T, cmap="gray", origin="lower")
            if probability is not None:
                axis.imshow(probability[:, :, index].T, cmap="magma", origin="lower",
                            vmin=0, vmax=1, alpha=.48)
            if mask is not None and mask[:, :, index].any():
                axis.contour(mask[:, :, index].T, levels=[.5], colors="#00e5ff", linewidths=.7)
                axis.imshow(np.ma.masked_where(~mask[:, :, index].T, mask[:, :, index].T),
                            cmap="winter", alpha=.18, origin="lower", vmin=0, vmax=1)
            axis.set_title(str(index), fontsize=7)
            axis.axis("off")
        for axis in axes.flat[len(indices):]:
            axis.axis("off")
        figure.suptitle(f"{title} | slices {indices[0]}–{indices[-1]}", fontsize=10)
        figure.savefig(destination / f"page_{start // per_page + 1:02d}.png", dpi=150)
        plt.close(figure)


def orthogonal_figure(image, raw, filtered, probability, destination: Path, title: str):
    center = np.round(np.argwhere(filtered).mean(0)).astype(int) if filtered.any() else np.array(image.shape) // 2
    views = ((0, int(center[0]), "sagittal"), (1, int(center[1]), "coronal"),
             (2, int(center[2]), "axial"))
    norm = robust_normalize(image)
    figure, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    for row, (axis_index, index, name) in enumerate(views):
        arrays = [np.take(x, index, axis=axis_index).T for x in (norm, raw, filtered, probability)]
        for col, label in enumerate(("raw", "filtered", "probability")):
            axes[row, col].imshow(arrays[0], cmap="gray", origin="lower")
            if col < 2:
                if arrays[col + 1].any():
                    axes[row, col].contour(arrays[col + 1], [.5], colors="#00e5ff")
            else:
                axes[row, col].imshow(arrays[3], cmap="magma", vmin=0, vmax=1, alpha=.5, origin="lower")
            axes[row, col].set_title(f"{name} {index} | {label}")
            axes[row, col].axis("off")
    figure.suptitle(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_movie(image, raw, filtered, probability, destination: Path, title: str):
    """Create a complete native-slice movie using the locally available ffmpeg."""
    # The RS2 conda environment ships ffmpeg even when it is not on the shell
    # PATH.  Point matplotlib at that executable without installing anything.
    candidates = [
        Path(sys.executable).parent / "ffmpeg",
        Path("/opt/homebrew/Caskroom/miniforge/base/envs/rs2/bin/ffmpeg"),
        Path("/opt/homebrew/Caskroom/miniforge/base/pkgs/ffmpeg-4.3.2-h38cfed3_3/bin/ffmpeg"),
    ]
    executable = next((path for path in candidates if path.exists()), None)
    if executable is None:
        raise RuntimeError("ffmpeg is required for complete slice-review MP4s")
    matplotlib.rcParams["animation.ffmpeg_path"] = str(executable)
    norm = robust_normalize(image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    writer = FFMpegWriter(fps=4, metadata={"title": title})
    with writer.saving(figure, str(destination), dpi=120):
        for z in range(image.shape[2]):
            for axis in axes:
                axis.clear()
            for axis, label in zip(axes, ("MRI", "raw", "filtered", "probability")):
                axis.imshow(norm[:, :, z].T, cmap="gray", origin="lower")
                axis.set_title(f"{label} | slice {z}")
                axis.axis("off")
            if raw[:, :, z].any():
                axes[1].contour(raw[:, :, z].T, [.5], colors="#00e5ff")
            if filtered[:, :, z].any():
                axes[2].contour(filtered[:, :, z].T, [.5], colors="#00e5ff")
            axes[3].imshow(probability[:, :, z].T, cmap="magma", vmin=0, vmax=1,
                           alpha=.5, origin="lower")
            writer.grab_frame()
    plt.close(figure)


def surface_figure(mask, destination: Path, title: str, cavity=None):
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111, projection="3d")
    if mask.any() and not mask.all():
        vertices, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), .5)
        axis.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
                          color="#68c4d9", alpha=.35, linewidth=0)
    if cavity is not None and cavity.any() and not cavity.all():
        vertices, faces, _, _ = measure.marching_cubes(cavity.astype(np.float32), .5)
        axis.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
                          color="#ef476f", alpha=.9, linewidth=0)
    axis.set_title(title)
    axis.set_box_aspect(mask.shape)
    axis.set_axis_off()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def cavity_analysis(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Measure, but do not apply, enclosed background components."""
    cavity = ndimage.binary_fill_holes(mask) & ~mask
    labels, count = ndimage.label(cavity, structure=np.ones((3, 3, 3)))
    voxel_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    rows = []
    for component in range(1, count + 1):
        coords = np.argwhere(labels == component)
        low, high = coords.min(0), coords.max(0)
        centroid = coords.mean(0)
        rows.append({
            "component": component, "voxels": int(len(coords)),
            "volume_mm3": float(len(coords) * voxel_mm3),
            "centroid_voxel": [float(x) for x in centroid],
            "bbox_min_voxel": [int(x) for x in low],
            "bbox_max_voxel": [int(x) for x in high],
        })
    return cavity.astype(np.uint8), sorted(rows, key=lambda row: row["voxels"], reverse=True)


def indentation_analysis(mask: np.ndarray, probability: np.ndarray) -> list[dict]:
    """Flag abrupt area/asymmetry changes for visual review; this is not diagnosis."""
    area = mask.sum((0, 1)).astype(float)
    results = []
    occupied = np.where(area > 0)[0]
    if len(occupied) < 3:
        return results
    for z in occupied[1:-1]:
        neighbor = (area[z - 1] + area[z + 1]) / 2
        drop = (neighbor - area[z]) / max(neighbor, 1)
        coords = np.argwhere(mask[:, :, z])
        mid = (coords[:, 0].min() + coords[:, 0].max()) / 2
        left = int((coords[:, 0] <= mid).sum())
        right = int((coords[:, 0] > mid).sum())
        asymmetry = abs(left - right) / max(left + right, 1)
        boundary = mask[:, :, z] ^ ndimage.binary_erosion(mask[:, :, z])
        boundary_probability = float(probability[:, :, z][boundary].mean()) if boundary.any() else float("nan")
        if drop >= .15 or asymmetry >= .35 or boundary_probability < .55:
            results.append({
                "slice": int(z), "relative_area_drop": float(drop),
                "left_right_asymmetry": float(asymmetry),
                "mean_boundary_probability": boundary_probability,
                "criteria": "area_drop>=0.15 OR asymmetry>=0.35 OR boundary_probability<0.55",
            })
    return results


def align_probability_to_raw(probability: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """Resolve reader-axis ordering and prove alignment via the fixed threshold.

    RS2's SimpleITK writer and nibabel expose the two equal-sized in-plane axes
    in opposite order for this DICOM conversion.  No interpolation is needed:
    search axis permutations/flips and require an exact identity between
    ``probability > .5`` and the already exported raw prediction.
    """
    candidates = []
    for permutation in itertools.permutations(range(3)):
        permuted = probability.transpose(permutation)
        if permuted.shape != raw.shape:
            continue
        for flips in itertools.product((False, True), repeat=3):
            candidate = permuted
            for axis, flip in enumerate(flips):
                if flip:
                    candidate = np.flip(candidate, axis)
            mismatches = int(np.count_nonzero((candidate > .5) != raw))
            candidates.append((mismatches, permutation, flips, candidate))
    best = min(candidates, key=lambda item: item[0])
    if best[0] != 0:
        raise RuntimeError(f"probability/raw threshold alignment has {best[0]} mismatched voxels")
    return np.asarray(best[3], dtype=np.float32), {
        "axis_permutation": list(best[1]), "axis_flips": list(best[2]),
        "threshold_mismatched_voxels": best[0],
    }


def candidate_zooms(image, mask, cavity, probability, candidates, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    norm = robust_normalize(image)
    for rank, row in enumerate(candidates[:8], 1):
        z = row["slice"]
        figure, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
        for axis, label in zip(axes, ("filtered boundary", "probability", "analysis-only cavity")):
            axis.imshow(norm[:, :, z].T, cmap="gray", origin="lower")
            axis.set_title(f"{label} | slice {z}")
            axis.axis("off")
        if mask[:, :, z].any():
            axes[0].contour(mask[:, :, z].T, [.5], colors="#00e5ff")
        axes[1].imshow(probability[:, :, z].T, cmap="magma", vmin=0, vmax=1,
                       alpha=.52, origin="lower")
        if cavity[:, :, z].any():
            axes[2].imshow(np.ma.masked_where(~cavity[:, :, z].T, cavity[:, :, z].T),
                           cmap="Reds", alpha=.7, origin="lower")
        figure.suptitle(
            f"Qualitative candidate {rank}: drop={row['relative_area_drop']:.2f}, "
            f"asymmetry={row['left_right_asymmetry']:.2f}"
        )
        figure.savefig(destination / f"candidate_{rank:02d}_slice_{z:03d}.png", dpi=180)
        plt.close(figure)


def model_from_checkpoint(config: dict, device):
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    checkpoint_path = ROOT / config["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    decoder = MultiScaleOneQueryMaskDecoder(32, 4)
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    if tuple(decoder.query.shape) != (1, 1, 32):
        raise RuntimeError("checkpoint is not the locked one-query architecture")
    if sum(parameter.numel() for parameter in decoder.parameters()) != 170401:
        raise RuntimeError("decoder parameter count does not match locked architecture")
    encoder = RS2NetEncoderAdapter(paths, image_size=tuple(config["tile_size"]),
                                   in_channels=1, out_channels=1, feature_size=48)
    model = FrozenEncoderQueryModel(encoder, decoder).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, paths, checkpoint, checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mouse_astrocytoma_zero_shot.yaml")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    output = ROOT / config["output_directory"]
    inventory = read_csv(output / "series_inventory.csv")
    selected = {item["SeriesInstanceUID"]: item for item in config["pilot_series"]}
    rows = [row for row in inventory if row["SeriesInstanceUID"] in selected]
    if len(rows) != len(selected) or len({row["subject_id"] for row in rows}) != len(rows):
        raise RuntimeError("pilot series must resolve uniquely to distinct subjects")
    if any(row["qc_class"] != "primary anatomical" for row in rows):
        raise RuntimeError("a selected pilot did not pass primary-anatomical QC")

    device = choose_device()
    model, paths, checkpoint, checkpoint_path = model_from_checkpoint(config, device)
    path_lookup = {}
    dataset_root = (ROOT / config["dataset_root"]).resolve()
    for path in dataset_root.rglob("*.dcm"):
        uid = str(pydicom.dcmread(str(path), stop_before_pixels=True,
                                 specific_tags=["SeriesInstanceUID"]).SeriesInstanceUID)
        if uid in selected:
            path_lookup.setdefault(uid, []).append(path)

    summaries = []
    for number, row in enumerate(rows, 1):
        subject = row["subject_id"]
        subject_dir = output / "pilots" / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        native_path = subject_dir / "native_mri.nii.gz"
        headers = series_headers(path_lookup[row["SeriesInstanceUID"]])
        original, affine = reconstruct_nifti(headers, native_path)
        zero_path = subject_dir / "geometry_only_zero_mask.nii.gz"
        save_zero_reference(native_path, zero_path)
        image, properties, manager, configuration, dataset = preprocess(
            native_path, zero_path, paths, tuple(config["tile_size"])
        )
        save_preprocessed(image, subject_dir / "preprocessed_mri_model_space.nii.gz")
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            logits = sliding_window_logits(model, image, tuple(config["tile_size"]), device)
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        raw_path = subject_dir / "raw_prediction.nii.gz"
        probability_path = subject_dir / "probability_map.nii.gz"
        export_native(logits, properties, manager, configuration, dataset, raw_path)
        export_probability(logits, properties, manager, configuration, dataset,
                           probability_path, native_path)
        raw = np.asarray(nib.load(str(raw_path)).dataobj) > 0
        probability = np.asarray(nib.load(str(probability_path)).dataobj, dtype=np.float32)
        if raw.shape != original.shape or probability.shape != original.shape:
            raise RuntimeError(f"native export geometry mismatch for {subject}")
        probability, probability_alignment = align_probability_to_raw(probability, raw)
        nib.save(nib.Nifti1Image(probability, affine), probability_path)
        filtered = largest_component_3d(raw)
        filtered_path = subject_dir / "filtered_prediction.nii.gz"
        nib.save(nib.Nifti1Image(filtered.astype(np.uint8), affine), filtered_path)

        cavity, cavities = cavity_analysis(filtered.astype(bool), affine)
        cavity_path = subject_dir / "analysis_only_internal_cavity.nii.gz"
        nib.save(nib.Nifti1Image(cavity, affine), cavity_path)
        candidates = indentation_analysis(filtered.astype(bool), probability)
        write_csv(subject_dir / "boundary_indentation_candidates.csv", candidates)
        save_json(subject_dir / "cavity_analysis.json", {
            "analysis_only": True, "not_applied_to_prediction": True,
            "definition": "binary_fill_holes(filtered prediction) minus filtered prediction",
            "components": cavities,
        })

        dense_pages(original, subject_dir / "contact_sheets" / "native_mri",
                    f"{subject} | native MRI")
        dense_pages(image[0, 0].cpu().numpy(), subject_dir / "contact_sheets" / "preprocessed_mri",
                    f"{subject} | preprocessed MRI")
        dense_pages(original, subject_dir / "contact_sheets" / "raw_prediction",
                    f"{subject} | raw prediction", mask=raw)
        dense_pages(original, subject_dir / "contact_sheets" / "filtered_prediction",
                    f"{subject} | largest-component prediction", mask=filtered.astype(bool))
        dense_pages(original, subject_dir / "contact_sheets" / "probability",
                    f"{subject} | probability", probability=probability)
        orthogonal_figure(original, raw, filtered.astype(bool), probability,
                          subject_dir / "orthogonal_views.png", subject)
        save_movie(original, raw, filtered.astype(bool), probability,
                   subject_dir / "full_slice_review.mp4", subject)
        surface_figure(raw, subject_dir / "surfaces" / "raw_surface.png", f"{subject}: raw surface")
        surface_figure(filtered, subject_dir / "surfaces" / "filtered_surface.png",
                       f"{subject}: largest-component surface")
        surface_figure(filtered, subject_dir / "cavity_analysis" / "transparent_brain_and_cavity.png",
                       f"{subject}: analysis-only cavity", cavity=cavity)
        candidate_zooms(original, filtered.astype(bool), cavity.astype(bool), probability,
                        candidates, subject_dir / "boundary_candidate_zooms")

        raw_labels, raw_count = ndimage.label(raw, structure=np.ones((3, 3, 3)))
        summary = {
            "subject_id": subject, "SeriesInstanceUID": row["SeriesInstanceUID"],
            "sequence": row["SeriesDescription"], "original_shape": list(original.shape),
            "preprocessed_shape": list(image.shape), "device": str(device),
            "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(checkpoint["epoch"]),
            "inference_seconds": elapsed, "raw_foreground_voxels": int(raw.sum()),
            "filtered_foreground_voxels": int(filtered.sum()), "raw_components": int(raw_count),
            "removed_components": max(int(raw_count) - (1 if raw.any() else 0), 0),
            "removed_voxels": int(raw.sum() - filtered.sum()),
            "cavity_components": len(cavities), "cavity_voxels": int(cavity.sum()),
            "boundary_candidate_slices": len(candidates),
            "mean_probability_inside_filtered_prediction":
                float(probability[filtered.astype(bool)].mean()) if filtered.any() else None,
            "probability_alignment": probability_alignment,
            "classification_scope": "qualitative candidate flags; no ground truth",
        }
        save_json(subject_dir / "subject_summary.json", summary)
        summaries.append(summary)
        print(f"[{number}/{len(rows)}] {subject}: {summary['raw_foreground_voxels']} raw voxels, "
              f"{summary['removed_voxels']} removed, {len(cavities)} cavity components", flush=True)

    write_csv(output / "pilot_summary.csv", summaries)
    save_json(output / "inference_summary.json", {
        "training_performed": False, "adaptation_performed": False,
        "ground_truth_available": False, "quantitative_accuracy_evaluation": False,
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device), "exactly_one_learned_query": True,
        "decoder_parameters": 170401, "encoder_frozen": True,
        "logit_threshold": config["logit_threshold"],
        "filter": "largest 26-connected foreground component",
        "pilot_count": len(summaries), "pilots": summaries,
    })


if __name__ == "__main__":
    main()
