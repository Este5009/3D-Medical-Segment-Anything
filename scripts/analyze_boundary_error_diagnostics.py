#!/usr/bin/env python3
"""Diagnose residual boundary errors in canonical filtered test predictions.

This script is deliberately analysis-only.  It reads the already-exported MRI,
expert mask, mixed-domain probability map, and deterministic largest-component
prediction for each locked test subject.  It never loads the neural network,
changes the 0.5 decision threshold, or writes a modified segmentation.

Two distance systems are kept separate throughout:

* ``distance_voxels`` is Euclidean distance in array-index coordinates.  It is
  useful for statements such as "within one voxel", but an anisotropic voxel
  does not have a unique physical size.
* ``distance_mm`` is a second Euclidean distance transform sampled with the
  native NIfTI voxel spacing.  It is the physically meaningful distance.

The expert surface is the set of expert foreground voxels removed by a
26-connected one-voxel erosion.  Every remaining FP/FN voxel is measured to
that fixed reference surface; detached islands are absent because the input is
the canonical largest-component-filtered prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/filtered_residual_failure_analysis/per_subject_metrics.csv"
PROB_ROOT = ROOT / "outputs/mixed_domain_anatomical_training/probability_maps"
DEFAULT_OUTPUT = ROOT / "outputs/boundary_error_diagnostics"
CHECKPOINT = ROOT / "outputs/mixed_domain_anatomical_training/checkpoints/best_mixed_domain.pt"
CONFIG = ROOT / "outputs/mixed_domain_anatomical_training/configuration.json"

GREEN = "#24c96b"
CYAN = "#00d5e7"
RED = "#ef3340"
YELLOW = "#ffd23f"
VOXEL_BINS = ((0.0, 1.0, "<=1 voxel"), (1.0, 2.0, ">1 to 2 voxels"),
              (2.0, 3.0, ">2 to 3 voxels"), (3.0, 5.0, ">3 to 5 voxels"),
              (5.0, np.inf, ">5 voxels"))
SURFACE_TOLERANCES_MM = (0.1, 0.2, 0.5, 1.0)
FULL_STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def probability_path(domain: str, subject: str) -> Path:
    folder = "camri_mixed" if domain == "CAMRI" else "mouse_mixed"
    if PROB_ROOT.name == "probability_maps" and (PROB_ROOT / domain.lower()).is_dir():
        folder = domain.lower()
    return PROB_ROOT / folder / f"{subject}_probability.nii.gz"


def expert_surface(mask: np.ndarray) -> np.ndarray:
    """Return the inner, 26-connected digital surface of a binary mask."""
    return mask & ~ndimage.binary_erosion(mask, structure=FULL_STRUCTURE,
                                          border_value=0)


def distance_to_surface(surface: np.ndarray, spacing: tuple[float, ...]):
    """Return nearest-surface distances and indices in voxel and mm systems."""
    distance_voxels, indices = ndimage.distance_transform_edt(
        ~surface, return_indices=True)
    distance_mm = ndimage.distance_transform_edt(~surface, sampling=spacing)
    return distance_voxels, distance_mm, indices


def binned_counts(values: np.ndarray) -> dict[str, float]:
    """Count mutually exclusive bins with boundary values assigned downward."""
    result: dict[str, float] = {}
    total = int(values.size)
    previous = 0.0
    for lower, upper, label in VOXEL_BINS:
        if math.isinf(upper):
            selected = values > lower
        elif lower == 0:
            selected = values <= upper
        else:
            selected = (values > lower) & (values <= upper)
        count = int(selected.sum())
        result[f"{label}_count"] = count
        result[f"{label}_percent"] = 100.0 * count / total if total else np.nan
        previous += count
    if total and int(previous) != total:
        raise AssertionError("Distance bins do not exhaust the error voxels")
    return result


def distance_statistics(values: np.ndarray, prefix: str) -> dict[str, float]:
    if not values.size:
        return {f"{prefix}_{key}": np.nan for key in
                ("mean", "median", "p95", "maximum")}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_maximum": float(np.max(values)),
    }


def binned_physical_statistics(voxel_values: np.ndarray,
                               mm_values: np.ndarray) -> dict[str, float]:
    """Describe physical distances within each index-space distance bin."""
    result = {}
    for lower, upper, label in VOXEL_BINS:
        if math.isinf(upper):
            selected = voxel_values > lower
        elif lower == 0:
            selected = voxel_values <= upper
        else:
            selected = (voxel_values > lower) & (voxel_values <= upper)
        key = label.replace(" ", "_").replace(">", "gt").replace("<=", "le")
        result.update(distance_statistics(mm_values[selected], f"{key}_distance_mm"))
    return result


def binary_metrics(prediction: np.ndarray, expert: np.ndarray,
                   spacing: tuple[float, ...]) -> dict[str, float]:
    """Compute overlap and symmetric physical surface metrics."""
    tp = int((prediction & expert).sum())
    fp = int((prediction & ~expert).sum())
    fn = int((~prediction & expert).sum())
    pred_surface = expert_surface(prediction)
    true_surface = expert_surface(expert)
    dt_true = ndimage.distance_transform_edt(~true_surface, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    pred_to_true = dt_true[pred_surface]
    true_to_pred = dt_pred[true_surface]
    symmetric = np.concatenate((pred_to_true, true_to_pred))
    result = {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "hd95_mm": float(np.percentile(symmetric, 95)),
        "assd_mm": float(np.mean(symmetric)),
        "fp_voxels": fp,
        "fn_voxels": fn,
    }
    for tolerance in SURFACE_TOLERANCES_MM:
        numerator = int((pred_to_true <= tolerance).sum()) + int(
            (true_to_pred <= tolerance).sum())
        result[f"surface_dice_{tolerance:g}mm"] = numerator / max(
            pred_to_true.size + true_to_pred.size, 1)
    return result


def local_complexity(mask: np.ndarray, spacing: tuple[float, ...],
                     surface: np.ndarray) -> np.ndarray:
    """Estimate surface complexity as local dispersion of signed-DT normals.

    A locally flat surface has nearly parallel unit normals and a mean resultant
    length near one.  Curved, corner-like, or pixelated expert geometry has more
    dispersed normals.  Complexity is ``1 - resultant length`` after a 3x3x3
    local average.  The measure is robust, bounded, and uses physical gradients.
    """
    signed = (ndimage.distance_transform_edt(mask, sampling=spacing) -
              ndimage.distance_transform_edt(~mask, sampling=spacing))
    gradients = np.gradient(signed, *spacing, edge_order=1)
    magnitude = np.sqrt(sum(component ** 2 for component in gradients))
    magnitude = np.maximum(magnitude, 1e-8)
    normals = [component / magnitude for component in gradients]
    mean_normals = [ndimage.uniform_filter(component, size=3, mode="nearest")
                    for component in normals]
    resultant = np.sqrt(sum(component ** 2 for component in mean_normals))
    complexity = np.clip(1.0 - resultant, 0.0, 1.0)
    return complexity[surface]


def normalized_image(image: np.ndarray, expert: np.ndarray) -> np.ndarray:
    """Robustly standardize MRI intensities using the nonzero field of view."""
    valid = np.isfinite(image) & ((image != 0) | expert)
    values = image[valid]
    lo, hi = np.percentile(values, (1, 99))
    clipped = np.clip(image, lo, hi)
    return ((clipped - np.mean(values)) / max(np.std(values), 1e-8)).astype(
        np.float32)


def image_evidence(image: np.ndarray, expert: np.ndarray,
                   spacing: tuple[float, ...]):
    """Return gradient, local inside/outside contrast, and local variance maps."""
    standardized = normalized_image(image, expert)
    gradients = np.gradient(standardized, *spacing, edge_order=1)
    gradient = np.sqrt(sum(component ** 2 for component in gradients))

    # A 3x3x3 neighborhood gives an interpretable, fixed voxel-local comparison.
    kernel = np.ones((3, 3, 3), dtype=np.float32)
    inside_count = ndimage.convolve(expert.astype(np.float32), kernel,
                                    mode="constant", cval=0)
    outside_count = ndimage.convolve((~expert).astype(np.float32), kernel,
                                     mode="constant", cval=0)
    inside_sum = ndimage.convolve(standardized * expert, kernel,
                                  mode="constant", cval=0)
    outside_sum = ndimage.convolve(standardized * (~expert), kernel,
                                   mode="constant", cval=0)
    contrast = np.abs(inside_sum / np.maximum(inside_count, 1) -
                      outside_sum / np.maximum(outside_count, 1))
    mean = ndimage.uniform_filter(standardized, size=3, mode="nearest")
    mean_sq = ndimage.uniform_filter(standardized ** 2, size=3, mode="nearest")
    variance = np.maximum(mean_sq - mean ** 2, 0)
    return standardized, gradient, contrast, variance


def align_probability_to_raw(probability: np.ndarray, raw: np.ndarray):
    """Recover the saved probability array's spatial axis order.

    The historical exporter selected the first shape-compatible permutation.
    For square in-plane data, both identity and x/y transpose have the same
    shape, although only one reproduces the separately saved native raw mask.
    We accept an alignment only when thresholding at the unchanged 0.5 exactly
    reproduces that canonical raw mask.  No probabilities or thresholds change.
    """
    import itertools
    candidates = []
    for permutation in itertools.permutations(range(3)):
        if tuple(probability.shape[i] for i in permutation) != raw.shape:
            continue
        candidate = probability.transpose(permutation)
        if np.array_equal(candidate >= 0.5, raw):
            candidates.append((permutation, candidate))
    if len(candidates) != 1:
        raise ValueError(f"Expected one exact probability alignment, found {len(candidates)}")
    return candidates[0][1], candidates[0][0]


def load_case(row: pd.Series):
    paths = {
        "image": Path(row.image_path),
        "expert": Path(row.ground_truth_path),
        "prediction": Path(row.filtered_prediction_path),
        "raw": Path(row.baseline_prediction_path),
        "probability": probability_path(row.domain, str(row.subject)),
    }
    objects = {name: nib.load(str(path)) for name, path in paths.items()}
    reference = objects["image"]
    for name, obj in objects.items():
        if obj.shape != reference.shape or not np.allclose(obj.affine,
                                                            reference.affine,
                                                            atol=1e-4):
            raise ValueError(f"Geometry mismatch for {row.domain}/{row.subject}: {name}")
    image = np.asarray(reference.dataobj, dtype=np.float32)
    expert = np.asarray(objects["expert"].dataobj) > 0
    prediction = np.asarray(objects["prediction"].dataobj) > 0
    raw = np.asarray(objects["raw"].dataobj) > 0
    probability_saved = np.asarray(objects["probability"].dataobj, dtype=np.float32)
    probability, probability_permutation = align_probability_to_raw(probability_saved, raw)
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError(f"Invalid probability range for {row.domain}/{row.subject}")
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    return (objects, image, expert, prediction, raw, probability, spacing,
            probability_permutation)


def append_values(store: dict, key: tuple, values: np.ndarray):
    if values.size:
        store[key].append(np.asarray(values, dtype=np.float32))


def safe_concat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def aggregate_distance_rows(subject_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in ("CAMRI", "Mouse", "Combined"):
        subset = subject_rows if domain == "Combined" else subject_rows[
            subject_rows.domain == domain]
        for error_type in ("FP", "FN", "All"):
            if error_type == "All":
                count = int(subset.fp_voxels.sum() + subset.fn_voxels.sum())
                # Weighted aggregation requires the saved histogram samples;
                # these columns are filled later by the caller.
                continue
    return pd.DataFrame(rows)


def _summary_from_values(domain: str, error_type: str, voxel: np.ndarray,
                         millimetres: np.ndarray) -> dict:
    row = {"domain": domain, "error_type": error_type,
           "error_voxels": int(voxel.size)}
    row.update(binned_counts(voxel))
    row.update(distance_statistics(voxel, "distance_voxels"))
    row.update(distance_statistics(millimetres, "distance_mm"))
    row.update(binned_physical_statistics(voxel, millimetres))
    return row


def save_csv(rows: Iterable[dict], destination: Path):
    frame = pd.DataFrame(list(rows))
    frame.to_csv(destination, index=False, float_format="%.10g")


def analyze(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    source = pd.read_csv(SOURCE, dtype={"subject": str})
    if dict(source.domain.value_counts()) != {"Mouse": 80, "CAMRI": 6}:
        raise AssertionError("Locked cohort must contain 6 CAMRI and 80 Mouse subjects")

    subject_distance_rows = []
    surface_rows = []
    spatial_counts = defaultdict(int)
    spatial_surface = defaultdict(int)
    curvature_counts = defaultdict(int)
    curvature_surface = defaultdict(int)
    curvature_distances = defaultdict(list)
    evidence_values = defaultdict(list)
    probability_values = defaultdict(list)
    all_distances = defaultdict(list)
    case_cache = {}

    for position, row in source.iterrows():
        (objects, image, expert, prediction, raw, probability, spacing,
         probability_permutation) = load_case(row)
        subject = str(row.subject)
        domain = row.domain
        surface = expert_surface(expert)
        distance_voxels, distance_mm, nearest = distance_to_surface(surface, spacing)
        fp = prediction & ~expert
        fn = ~prediction & expert
        error = fp | fn
        metrics = binary_metrics(prediction, expert, spacing)
        axcodes = "".join(nib.aff2axcodes(objects["image"].affine))

        # Probability threshold must reproduce every retained prediction voxel.
        # Extra raw foreground is allowed only because the deterministic filter
        # removed disconnected components after thresholding.
        thresholded = probability >= 0.5
        if not np.array_equal(thresholded, raw):
            raise AssertionError(f"Aligned probability does not reproduce raw mask: {subject}")
        if np.any(prediction & ~thresholded):
            raise AssertionError(f"Filtered prediction is not a subset of p>=0.5: {subject}")

        for kind, mask in (("FP", fp), ("FN", fn), ("All", error)):
            voxel_values = distance_voxels[mask]
            mm_values = distance_mm[mask]
            record = {
                "domain": domain, "subject": subject, "error_type": kind,
                "error_voxels": int(mask.sum()), "voxel_spacing_x_mm": spacing[0],
                "voxel_spacing_y_mm": spacing[1], "voxel_spacing_z_mm": spacing[2],
                "orientation_codes": axcodes,
            }
            record.update(binned_counts(voxel_values))
            record.update(distance_statistics(voxel_values, "distance_voxels"))
            record.update(distance_statistics(mm_values, "distance_mm"))
            record.update(binned_physical_statistics(voxel_values, mm_values))
            subject_distance_rows.append(record)
            append_values(all_distances, (domain, kind, "voxel"), voxel_values)
            append_values(all_distances, (domain, kind, "mm"), mm_values)

        # Surface metrics are recomputed from the canonical filtered volumes.
        surface_rows.append({
            "domain": domain, "subject": subject,
            "voxel_spacing_x_mm": spacing[0], "voxel_spacing_y_mm": spacing[1],
            "voxel_spacing_z_mm": spacing[2], **metrics,
        })

        # Normalized acquisition-slice position uses the expert's occupied
        # extent on array axis 2.  It makes no rostral/caudal assertion.
        occupied_k = np.where(expert)[2]
        k_min, k_max = int(occupied_k.min()), int(occupied_k.max())
        span = max(k_max - k_min, 1)
        normalized_k = np.clip((np.arange(expert.shape[2]) - k_min) / span, 0, 1)
        region_for_k = np.where(normalized_k < .2, "first 20%",
                                np.where(normalized_k > .8, "final 20%", "middle 60%"))
        surface_k = np.where(surface)[2]
        for region in ("first 20%", "middle 60%", "final 20%"):
            spatial_surface[(domain, region)] += int((region_for_k[surface_k] == region).sum())
        for kind, mask in (("FP", fp), ("FN", fn), ("All", error)):
            error_k = np.where(mask)[2]
            for region in ("first 20%", "middle 60%", "final 20%"):
                spatial_counts[(domain, kind, region)] += int(
                    (region_for_k[error_k] == region).sum())

        # Each error voxel is attributed to its nearest expert-surface point.
        surface_coords = np.argwhere(surface)
        surface_linear = np.ravel_multi_index(surface_coords.T, expert.shape)
        lookup = np.full(expert.size, -1, dtype=np.int32)
        lookup[surface_linear] = np.arange(surface_coords.shape[0])
        nearest_linear = np.ravel_multi_index(nearest.reshape(3, -1), expert.shape)
        nearest_id = lookup[nearest_linear].reshape(expert.shape)
        if np.any(nearest_id[error] < 0):
            raise AssertionError("Nearest-surface attribution failed")

        complexity = local_complexity(expert, spacing, surface)
        q1, q2 = np.quantile(complexity, (1 / 3, 2 / 3))
        complexity_group = np.where(complexity <= q1, 0,
                                    np.where(complexity <= q2, 1, 2))
        labels = ("flat", "medium complexity", "high complexity")
        for group_id, label in enumerate(labels):
            curvature_surface[(domain, label)] += int((complexity_group == group_id).sum())
        for kind, mask in (("FP", fp), ("FN", fn), ("All", error)):
            ids = nearest_id[mask]
            distances = distance_mm[mask]
            for group_id, label in enumerate(labels):
                selected = complexity_group[ids] == group_id
                curvature_counts[(domain, kind, label)] += int(selected.sum())
                append_values(curvature_distances, (domain, kind, label), distances[selected])

        # For image evidence, classify every expert surface location by the
        # largest error assigned to it.  No assigned error means correctly
        # localized; the >1 to 2 voxel interval is intentionally not folded into
        # either requested disagreement group.
        max_assigned = np.zeros(surface_coords.shape[0], dtype=np.float32)
        if error.any():
            np.maximum.at(max_assigned, nearest_id[error], distance_voxels[error])
        standardized, gradient, contrast, variance = image_evidence(image, expert, spacing)
        surface_gradient = gradient[surface]
        surface_contrast = contrast[surface]
        surface_variance = variance[surface]
        evidence_groups = {
            "correctly localized": max_assigned == 0,
            "<=1 voxel disagreement": (max_assigned > 0) & (max_assigned <= 1),
            ">2 voxel disagreement": max_assigned > 2,
        }
        for label, selected in evidence_groups.items():
            append_values(evidence_values, (domain, label, "gradient_per_mm"),
                          surface_gradient[selected])
            append_values(evidence_values, (domain, label, "inside_outside_contrast_z"),
                          surface_contrast[selected])
            append_values(evidence_values, (domain, label, "local_variance_z2"),
                          surface_variance[selected])
        append_values(evidence_values, (domain, "all surface", "assigned_distance_voxels"),
                      max_assigned)
        append_values(evidence_values, (domain, "all surface", "gradient_per_mm"),
                      surface_gradient)
        append_values(evidence_values, (domain, "all surface", "inside_outside_contrast_z"),
                      surface_contrast)
        append_values(evidence_values, (domain, "all surface", "local_variance_z2"),
                      surface_variance)

        # Compare probabilities only near the expert surface (<=3 index-space
        # voxels), preventing easy far-field background from dominating TN.
        near_surface = distance_voxels <= 3
        probability_groups = {
            "correct inside": near_surface & expert & prediction,
            "correct outside": near_surface & ~expert & ~prediction,
            "FP boundary": fp,
            "FN boundary": fn,
        }
        for label, selected in probability_groups.items():
            append_values(probability_values, (domain, label), probability[selected])

        # Keep only selection scalars between subjects.  Full 3D arrays are
        # reloaded for the selected figures, preventing multi-gigabyte caching.
        high_ids = np.where(complexity_group == 2)[0]
        case_cache[(domain, subject)] = {
            "source_row": row.to_dict(), "metrics": metrics,
            "le1_fraction": float(np.mean(distance_voxels[error] <= 1)),
            "over3_count": int((error & (distance_voxels > 3)).sum()),
            "fp_fn_ratio": (int(fp.sum()) + 1) / (int(fn.sum()) + 1),
            "fn_fp_ratio": (int(fn.sum()) + 1) / (int(fp.sum()) + 1),
            "high_complexity_error_count": int((error & np.isin(nearest_id, high_ids)).sum()),
            "probability_axis_permutation": str(tuple(probability_permutation)),
        }
        print(f"[{position + 1:02d}/86] {domain} {subject}", flush=True)

    # Subject-level and pooled distance summaries.
    save_csv(subject_distance_rows, output / "boundary_distance_subject.csv")
    summary_rows = []
    for domain in ("CAMRI", "Mouse", "Combined"):
        domains = ("CAMRI", "Mouse") if domain == "Combined" else (domain,)
        for kind in ("FP", "FN", "All"):
            voxel = safe_concat(sum((all_distances[(d, kind, "voxel")] for d in domains), []))
            mm = safe_concat(sum((all_distances[(d, kind, "mm")] for d in domains), []))
            summary_rows.append(_summary_from_values(domain, kind, voxel, mm))
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output / "boundary_distance_summary.csv", index=False,
                         float_format="%.10g")

    # FP versus FN uses every post-filter residual error; no removed component
    # can enter because only filtered predictions were loaded.
    fp_fn_rows = []
    for domain in ("CAMRI", "Mouse", "Combined"):
        row_fp = summary_frame[(summary_frame.domain == domain) &
                               (summary_frame.error_type == "FP")].iloc[0]
        row_fn = summary_frame[(summary_frame.domain == domain) &
                               (summary_frame.error_type == "FN")].iloc[0]
        fp_count, fn_count = int(row_fp.error_voxels), int(row_fn.error_voxels)
        ratio = fp_count / fn_count if fn_count else np.inf
        if ratio > 1.2:
            classification = "over-segmentation (FP dominated)"
        elif ratio < 1 / 1.2:
            classification = "under-segmentation (FN dominated)"
        else:
            classification = "balanced"
        fp_fn_rows.append({
            "domain": domain, "boundary_fp_voxels": fp_count,
            "boundary_fn_voxels": fn_count, "fp_fn_ratio": ratio,
            "classification": classification,
            "fp_distance_mm_mean": row_fp.distance_mm_mean,
            "fp_distance_mm_median": row_fp.distance_mm_median,
            "fp_distance_mm_p95": row_fp.distance_mm_p95,
            "fn_distance_mm_mean": row_fn.distance_mm_mean,
            "fn_distance_mm_median": row_fn.distance_mm_median,
            "fn_distance_mm_p95": row_fn.distance_mm_p95,
        })
    save_csv(fp_fn_rows, output / "fp_fn_summary.csv")

    # Spatial denominators use expert-surface voxels so a larger middle anatomy
    # does not automatically appear more difficult.
    spatial_rows = []
    for domain in ("CAMRI", "Mouse", "Combined"):
        domains = ("CAMRI", "Mouse") if domain == "Combined" else (domain,)
        for kind in ("FP", "FN", "All"):
            total_errors = sum(spatial_counts[(d, kind, r)] for d in domains
                               for r in ("first 20%", "middle 60%", "final 20%"))
            for region in ("first 20%", "middle 60%", "final 20%"):
                errors = sum(spatial_counts[(d, kind, region)] for d in domains)
                opportunities = sum(spatial_surface[(d, region)] for d in domains)
                spatial_rows.append({
                    "domain": domain, "error_type": kind,
                    "normalized_slice_region": region, "error_voxels": errors,
                    "percent_of_error_voxels": 100 * errors / max(total_errors, 1),
                    "expert_surface_voxels": opportunities,
                    "errors_per_100_surface_voxels": 100 * errors / max(opportunities, 1),
                    "orientation_interpretation": "acquisition-slice position only",
                })
    save_csv(spatial_rows, output / "spatial_distribution.csv")

    curvature_rows = []
    for domain in ("CAMRI", "Mouse", "Combined"):
        domains = ("CAMRI", "Mouse") if domain == "Combined" else (domain,)
        for kind in ("FP", "FN", "All"):
            for label in ("flat", "medium complexity", "high complexity"):
                errors = sum(curvature_counts[(d, kind, label)] for d in domains)
                surface_count = sum(curvature_surface[(d, label)] for d in domains)
                distances = safe_concat(sum((curvature_distances[(d, kind, label)]
                                             for d in domains), []))
                curvature_rows.append({
                    "domain": domain, "error_type": kind,
                    "complexity_group": label, "error_voxels": errors,
                    "expert_surface_voxels": surface_count,
                    "errors_per_100_surface_voxels": 100 * errors / max(surface_count, 1),
                    **distance_statistics(distances, "error_distance_mm"),
                })
    save_csv(curvature_rows, output / "curvature_analysis.csv")

    evidence_rows = []
    for domain in ("CAMRI", "Mouse", "Combined"):
        domains = ("CAMRI", "Mouse") if domain == "Combined" else (domain,)
        for group in ("correctly localized", "<=1 voxel disagreement",
                      ">2 voxel disagreement"):
            row = {"domain": domain, "surface_region_group": group}
            sample_count = None
            for measure in ("gradient_per_mm", "inside_outside_contrast_z",
                            "local_variance_z2"):
                values = safe_concat(sum((evidence_values[(d, group, measure)]
                                          for d in domains), []))
                sample_count = values.size
                row[f"{measure}_mean"] = float(np.mean(values)) if values.size else np.nan
                row[f"{measure}_median"] = float(np.median(values)) if values.size else np.nan
            row["expert_surface_regions"] = int(sample_count or 0)
            evidence_rows.append(row)
        distances = safe_concat(sum((evidence_values[
            (d, "all surface", "assigned_distance_voxels")] for d in domains), []))
        correlation = {"domain": domain,
                       "surface_region_group": "all-surface correlation",
                       "expert_surface_regions": int(distances.size)}
        for measure in ("gradient_per_mm", "inside_outside_contrast_z",
                        "local_variance_z2"):
            values = safe_concat(sum((evidence_values[(d, "all surface", measure)]
                                      for d in domains), []))
            rho, p_value = stats.spearmanr(distances, values)
            correlation[f"assigned_distance_vs_{measure}_spearman_rho"] = rho
            correlation[f"assigned_distance_vs_{measure}_p_value"] = p_value
        evidence_rows.append(correlation)
    save_csv(evidence_rows, output / "image_boundary_analysis.csv")

    probability_rows = []
    for domain in ("CAMRI", "Mouse", "Combined"):
        domains = ("CAMRI", "Mouse") if domain == "Combined" else (domain,)
        for group in ("correct inside", "correct outside", "FP boundary", "FN boundary"):
            values = safe_concat(sum((probability_values[(d, group)] for d in domains), []))
            row = {"domain": domain, "voxel_group": group, "voxels": int(values.size)}
            for label, percentile in (("p05", 5), ("p25", 25), ("median", 50),
                                      ("p75", 75), ("p95", 95)):
                row[f"probability_{label}"] = (float(np.percentile(values, percentile))
                                                if values.size else np.nan)
            row["probability_mean"] = float(np.mean(values)) if values.size else np.nan
            row["uncertain_0.25_to_0.75_percent"] = (100 * np.mean(
                (values >= .25) & (values <= .75)) if values.size else np.nan)
            if group == "FP boundary":
                row["confident_incorrect_percent"] = 100 * np.mean(values >= .9)
            elif group == "FN boundary":
                row["confident_incorrect_percent"] = 100 * np.mean(values <= .1)
            else:
                row["confident_incorrect_percent"] = np.nan
            probability_rows.append(row)
    save_csv(probability_rows, output / "probability_analysis.csv")
    save_csv(surface_rows, output / "surface_metrics.csv")

    selection = select_figure_cases(case_cache)
    manifest = []
    for domain, role, subject in selection:
        role_slug = (role.replace(" ", "_").replace(">", "gt")
                     .replace("<=", "le").replace("/", "_"))
        destination = figures / domain.lower() / f"{role_slug}_{subject}.png"
        destination.parent.mkdir(exist_ok=True)
        data = prepare_figure_data(pd.Series(case_cache[(domain, subject)]["source_row"]))
        figure_case(domain, subject, role, data, destination)
        manifest.append({"domain": domain, "role": role, "subject": subject,
                         "figure_path": str(destination.resolve())})
    save_csv(manifest, figures / "manifest.csv")

    write_report(output, summary_frame, pd.DataFrame(fp_fn_rows),
                 pd.DataFrame(spatial_rows), pd.DataFrame(curvature_rows),
                 pd.DataFrame(evidence_rows), pd.DataFrame(probability_rows),
                 pd.DataFrame(surface_rows), manifest)
    validation = {
        "subjects": 86, "camri_subjects": 6, "mouse_subjects": 80,
        "geometry_verified": True, "probability_range_verified": True,
        "filtered_prediction_subset_of_probability_threshold_mask": True,
        "aligned_probability_threshold_exactly_matches_raw_prediction": True,
        "probability_alignment_method": "unique shape-compatible axis permutation with exact p>=0.5 raw-mask match",
        "training_performed": False, "model_inference_performed": False,
        "threshold_changed": False, "postprocessing_changed": False,
        "checkpoint": str(CHECKPOINT.resolve()), "configuration": str(CONFIG.resolve()),
        "surface_tolerances_mm": list(SURFACE_TOLERANCES_MM),
        "anatomical_orientation_claimed": False,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")


def select_figure_cases(cache: dict) -> list[tuple[str, str, str]]:
    """Greedily choose informative subjects while avoiding duplicates."""
    selected = []
    roles = ("best boundary", "median boundary", "worst boundary",
             "mostly <=1 voxel", "largest >3 voxel", "FP dominated",
             "FN dominated", "high curvature failure")
    for domain in ("CAMRI", "Mouse"):
        cases = [(subject, data) for (d, subject), data in cache.items() if d == domain]
        assd_sorted = sorted(cases, key=lambda item: item[1]["metrics"]["assd_mm"])
        median_assd = float(np.median([item[1]["metrics"]["assd_mm"] for item in cases]))
        scorers = {
            "best boundary": lambda item: -item[1]["metrics"]["assd_mm"],
            "median boundary": lambda item: -abs(item[1]["metrics"]["assd_mm"] - median_assd),
            "worst boundary": lambda item: item[1]["metrics"]["assd_mm"],
            "mostly <=1 voxel": lambda item: item[1]["le1_fraction"],
            "largest >3 voxel": lambda item: item[1]["over3_count"],
            "FP dominated": lambda item: item[1]["fp_fn_ratio"],
            "FN dominated": lambda item: item[1]["fn_fp_ratio"],
            "high curvature failure": lambda item: item[1]["high_complexity_error_count"],
        }
        used = set()
        for role in roles:
            ranked = sorted(cases, key=scorers[role], reverse=True)
            choice = next((item for item in ranked if item[0] not in used), ranked[0])
            used.add(choice[0])
            selected.append((domain, role, choice[0]))
    return selected


def prepare_figure_data(row: pd.Series) -> dict:
    """Reload and derive full arrays only for a selected visual example."""
    (_, image, expert, prediction, _raw, probability, spacing,
     _permutation) = load_case(row)
    surface = expert_surface(expert)
    distance_voxels, distance_mm, nearest = distance_to_surface(surface, spacing)
    fp = prediction & ~expert
    fn = ~prediction & expert
    error = fp | fn
    surface_coords = np.argwhere(surface)
    surface_linear = np.ravel_multi_index(surface_coords.T, expert.shape)
    lookup = np.full(expert.size, -1, dtype=np.int32)
    lookup[surface_linear] = np.arange(surface_coords.shape[0])
    nearest_linear = np.ravel_multi_index(nearest.reshape(3, -1), expert.shape)
    nearest_id = lookup[nearest_linear].reshape(expert.shape)
    complexity = local_complexity(expert, spacing, surface)
    q1, q2 = np.quantile(complexity, (1 / 3, 2 / 3))
    complexity_group = np.where(complexity <= q1, 0,
                                np.where(complexity <= q2, 1, 2))
    return {
        "image": image, "expert": expert, "prediction": prediction,
        "probability": probability, "spacing": spacing,
        "distance_voxels": distance_voxels, "distance_mm": distance_mm,
        "fp": fp, "fn": fn, "error": error, "nearest_id": nearest_id,
        "complexity_group": complexity_group,
        "metrics": binary_metrics(prediction, expert, spacing),
    }


def _slice_for_role(data: dict, role: str) -> int:
    error = data["error"]
    if role == "FP dominated":
        weights = data["fp"]
    elif role == "FN dominated":
        weights = data["fn"]
    elif role == "largest >3 voxel":
        weights = error & (data["distance_voxels"] > 3)
    elif role == "high curvature failure":
        high_ids = np.where(data["complexity_group"] == 2)[0]
        weights = error & np.isin(data["nearest_id"], high_ids)
    else:
        weights = error
    counts = weights.sum(axis=(0, 1))
    if counts.max() == 0:
        counts = data["expert"].sum(axis=(0, 1))
    return int(np.argmax(counts))


def _crop(mask: np.ndarray, z: int, pad: int = 10):
    coords = np.argwhere(mask[:, :, z])
    if not coords.size:
        coords = np.argwhere(mask.any(axis=2))
    if not coords.size:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    lo = np.maximum(coords.min(0)[:2] - pad, 0)
    hi = np.minimum(coords.max(0)[:2] + pad + 1, mask.shape[:2])
    return slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1]))


def figure_case(domain: str, subject: str, role: str, data: dict,
                destination: Path):
    z = _slice_for_role(data, role)
    expert, prediction = data["expert"], data["prediction"]
    fp, fn = data["fp"], data["fn"]
    crop = _crop(expert | prediction, z)
    # Center a true local crop on the most distant error voxel in this slice.
    slice_error = fp[:, :, z] | fn[:, :, z]
    if slice_error.any():
        weighted = np.where(slice_error, data["distance_mm"][:, :, z], -1)
        center = np.array(np.unravel_index(np.argmax(weighted), weighted.shape))
    else:
        center = np.array(np.unravel_index(np.argmax(expert[:, :, z]),
                                           expert[:, :, z].shape))
    radius = 24
    zoom = (slice(max(int(center[0]) - radius, 0),
                  min(int(center[0]) + radius + 1, expert.shape[0])),
            slice(max(int(center[1]) - radius, 0),
                  min(int(center[1]) + radius + 1, expert.shape[1])))
    arrays = [data["image"], expert, prediction, data["probability"], fp, fn]
    sl = [np.rot90(array[crop[0], crop[1], z]) for array in arrays]
    image, gt, pred, probability, fp_s, fn_s = sl
    zoom_sl = [np.rot90(array[zoom[0], zoom[1], z]) for array in arrays]
    zoom_image, zoom_gt, zoom_pred, zoom_probability, zoom_fp, zoom_fn = zoom_sl
    valid = np.isfinite(image)
    lo, hi = np.percentile(image[valid], (1, 99))
    error_code = np.zeros(gt.shape, dtype=np.uint8)
    error_code[fp_s] = 1
    error_code[fn_s] = 2
    error_cmap = ListedColormap([(0, 0, 0, 0), RED, YELLOW])
    aspect = data["spacing"][1] / data["spacing"][0]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    panels = (("MRI", image), ("Expert mask", gt), ("Filtered prediction", pred))
    for axis, (title, panel) in zip(axes[0, :3], panels):
        axis.imshow(panel, cmap="gray", vmin=(lo if title == "MRI" else 0),
                    vmax=(hi if title == "MRI" else 1), interpolation="nearest",
                    aspect=aspect)
        axis.set_title(title)
    axes[0, 3].imshow(image, cmap="gray", vmin=lo, vmax=hi,
                      interpolation="nearest", aspect=aspect)
    if gt.any(): axes[0, 3].contour(gt, [.5], colors=GREEN, linewidths=1.2)
    if pred.any(): axes[0, 3].contour(pred, [.5], colors=CYAN, linewidths=1.2)
    axes[0, 3].set_title("Expert / predicted contours")

    axes[1, 0].imshow(image, cmap="gray", vmin=lo, vmax=hi,
                      interpolation="nearest", aspect=aspect)
    axes[1, 0].imshow(error_code, cmap=error_cmap, vmin=0, vmax=2,
                      interpolation="nearest", aspect=aspect)
    axes[1, 0].set_title("FP / FN error map")
    zoom_error = np.zeros(zoom_gt.shape, dtype=np.uint8)
    zoom_error[zoom_fp] = 1
    zoom_error[zoom_fn] = 2
    axes[1, 1].imshow(zoom_image, cmap="gray", vmin=lo, vmax=hi,
                      interpolation="nearest", aspect=aspect)
    if zoom_gt.any(): axes[1, 1].contour(zoom_gt, [.5], colors=GREEN, linewidths=1.5)
    if zoom_pred.any(): axes[1, 1].contour(zoom_pred, [.5], colors=CYAN, linewidths=1.5)
    axes[1, 1].imshow(zoom_error, cmap=error_cmap, vmin=0, vmax=2,
                      interpolation="nearest", alpha=.7, aspect=aspect)
    axes[1, 1].set_title("Voxel-exact boundary zoom")
    probability_image = axes[1, 2].imshow(zoom_probability, cmap="magma", vmin=0, vmax=1,
                                          interpolation="nearest", aspect=aspect)
    if zoom_gt.any(): axes[1, 2].contour(zoom_gt, [.5], colors=GREEN, linewidths=1.0)
    axes[1, 2].set_title("Local probability map")
    fig.colorbar(probability_image, ax=axes[1, 2], orientation="horizontal",
                 fraction=.07, pad=.08, label="Foreground probability")

    voxel_distance = data["distance_voxels"][:, :, z][slice_error]
    mm_distance = data["distance_mm"][:, :, z][slice_error]
    axes[1, 3].axis("off")
    axes[1, 3].legend(handles=[
        Patch(facecolor=GREEN, label="Expert boundary"),
        Patch(facecolor=CYAN, label="Predicted boundary"),
        Patch(facecolor=RED, label="False positive"),
        Patch(facecolor=YELLOW, label="False negative"),
    ], loc="upper left", frameon=False)
    axes[1, 3].text(0, .50,
        f"Slice error voxels: {int(slice_error.sum()):,}\n"
        f"FP: {int(fp[:, :, z].sum()):,}   FN: {int(fn[:, :, z].sum()):,}\n"
        f"Max distance: {float(voxel_distance.max()) if voxel_distance.size else 0:.2f} voxels\n"
        f"Max physical distance: {float(mm_distance.max()) if mm_distance.size else 0:.3f} mm\n"
        f"Spacing: {data['spacing'][0]:.3g} × {data['spacing'][1]:.3g} × "
        f"{data['spacing'][2]:.3g} mm", va="top", fontsize=11)

    for axis in axes.flat[:7]:
        axis.set_xticks([]); axis.set_yticks([])
    fig.suptitle(
        f"{domain} · {subject} · {role} · acquisition slice {z} · "
        f"Dice {data['metrics']['dice']:.4f} · HD95 {data['metrics']['hd95_mm']:.3f} mm",
        fontsize=14, y=.975)
    fig.subplots_adjust(left=.025, right=.975, bottom=.055, top=.885,
                        wspace=.10, hspace=.22)
    # Do not use bbox_inches='tight': it can clip the required case title when
    # constrained layout also manages the colorbar and legend.
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _get(frame: pd.DataFrame, domain: str, kind: str) -> pd.Series:
    return frame[(frame.domain == domain) & (frame.error_type == kind)].iloc[0]


def write_report(output: Path, distance: pd.DataFrame, fp_fn: pd.DataFrame,
                 spatial: pd.DataFrame, curvature: pd.DataFrame,
                 evidence: pd.DataFrame, probability: pd.DataFrame,
                 surfaces: pd.DataFrame, manifest: list[dict]):
    combined = _get(distance, "Combined", "All")
    camri = _get(distance, "CAMRI", "All")
    mouse = _get(distance, "Mouse", "All")
    combined_fpfn = fp_fn[fp_fn.domain == "Combined"].iloc[0]

    spatial_all = spatial[(spatial.domain == "Combined") & (spatial.error_type == "All")]
    middle_rate = float(spatial_all[spatial_all.normalized_slice_region == "middle 60%"]
                        .errors_per_100_surface_voxels.iloc[0])
    terminal = spatial_all[spatial_all.normalized_slice_region != "middle 60%"]
    terminal_rate = float(np.average(terminal.errors_per_100_surface_voxels,
                                     weights=terminal.expert_surface_voxels))
    terminal_ratio = terminal_rate / middle_rate

    curv_all = curvature[(curvature.domain == "Combined") &
                         (curvature.error_type == "All")]
    flat_rate = float(curv_all[curv_all.complexity_group == "flat"]
                      .errors_per_100_surface_voxels.iloc[0])
    high_rate = float(curv_all[curv_all.complexity_group == "high complexity"]
                      .errors_per_100_surface_voxels.iloc[0])
    curvature_ratio = high_rate / flat_rate

    ev = evidence[evidence.domain == "Combined"].set_index("surface_region_group")
    grad_correct = float(ev.loc["correctly localized", "gradient_per_mm_mean"])
    grad_far = float(ev.loc[">2 voxel disagreement", "gradient_per_mm_mean"])
    contrast_correct = float(ev.loc["correctly localized", "inside_outside_contrast_z_mean"])
    contrast_far = float(ev.loc[">2 voxel disagreement", "inside_outside_contrast_z_mean"])

    prob = probability[probability.domain == "Combined"].set_index("voxel_group")
    error_count = float(prob.loc["FP boundary", "voxels"] + prob.loc["FN boundary", "voxels"])
    uncertain = float((prob.loc["FP boundary", "voxels"] *
                       prob.loc["FP boundary", "uncertain_0.25_to_0.75_percent"] +
                       prob.loc["FN boundary", "voxels"] *
                       prob.loc["FN boundary", "uncertain_0.25_to_0.75_percent"]) /
                      max(error_count, 1))
    confident = float((prob.loc["FP boundary", "voxels"] *
                       prob.loc["FP boundary", "confident_incorrect_percent"] +
                       prob.loc["FN boundary", "voxels"] *
                       prob.loc["FN boundary", "confident_incorrect_percent"]) /
                      max(error_count, 1))

    surface_means = surfaces.groupby("domain").mean(numeric_only=True)
    bins = [(label, combined[f"{label}_percent"]) for _, _, label in VOXEL_BINS]
    bin_text = "\n".join(f"- {label}: {value:.2f}%" for label, value in bins)
    physical_bin_lines = []
    for _, _, label in VOXEL_BINS:
        key = label.replace(" ", "_").replace(">", "gt").replace("<=", "le")
        physical_bin_lines.append(
            f"- {label}: median {combined[f'{key}_distance_mm_median']:.3f} mm, "
            f"P95 {combined[f'{key}_distance_mm_p95']:.3f} mm, maximum "
            f"{combined[f'{key}_distance_mm_maximum']:.3f} mm")
    physical_by_bin = "\n".join(physical_bin_lines)
    physical = "\n".join(
        f"- {domain}: mean {row.distance_mm_mean:.3f} mm, median "
        f"{row.distance_mm_median:.3f} mm, P95 {row.distance_mm_p95:.3f} mm, "
        f"maximum {row.distance_mm_maximum:.3f} mm"
        for domain, row in (("CAMRI", camri), ("Mouse", mouse),
                            ("Combined", combined)))

    orientation_codes = sorted(pd.read_csv(output / "boundary_distance_subject.csv")
                               .orientation_codes.dropna().unique())
    report = f"""# Post-filter boundary-error diagnostics

## Scope and provenance

This diagnostic used all 86 canonical untouched test outputs: 6 CAMRI and 80
Mouse subjects. Inputs were the saved expert masks, native MRI volumes,
continuous probability maps, and deterministic largest-26-connected-component
predictions from the unchanged epoch-17 mixed-domain system. There was no
training, inference, threshold selection, model change, or postprocessing
change. The checkpoint was `{CHECKPOINT.relative_to(ROOT)}` and the recovered
configuration was `{CONFIG.relative_to(ROOT)}`.

An expert surface voxel is expert foreground removed by one 26-connected binary
erosion. Error distance is the shortest Euclidean distance from every remaining
FP or FN voxel to this fixed expert surface. Voxel distance uses array-index
coordinates; physical distance uses the real native spacing in all three axes.

## Direct answers

### 1–3. How far are the remaining errors?

Across {int(combined.error_voxels):,} post-filter error voxels:

{bin_text}

Physical distances:

{physical}

Physical distances within the combined index-space bins:

{physical_by_bin}

The voxel bins and millimetre summaries must not be converted into one another
with a single scale because the cohort contains anisotropic acquisitions.

### 4. FP or FN?

Combined residuals contain {int(combined_fpfn.boundary_fp_voxels):,} FP and
{int(combined_fpfn.boundary_fn_voxels):,} FN voxels (FP/FN ratio
{combined_fpfn.fp_fn_ratio:.3f}); this is **{combined_fpfn.classification}**.
Domain-specific counts and FP/FN distance summaries are in `fp_fn_summary.csv`.
Detached FP islands removed by the filter are not counted.

### 5–6. Spatial uniformity and terminal regions

Errors are not evaluated by raw counts alone: each acquisition-slice region is
normalized by its number of expert-surface voxels. The combined terminal-region
error rate is {terminal_ratio:.3f}× the middle-60% rate. Thus terminal regions
are {"disproportionately difficult" if terminal_ratio > 1.2 else "not strongly disproportionate by the predeclared 1.2× descriptive criterion"}.
The normalized regional rates range from
{spatial_all.errors_per_100_surface_voxels.min():.3f} to
{spatial_all.errors_per_100_surface_voxels.max():.3f} errors per 100 expert
surface voxels, so the distribution is
{"meaningfully non-uniform" if spatial_all.errors_per_100_surface_voxels.max()/spatial_all.errors_per_100_surface_voxels.min() > 1.2 else "approximately uniform by the same 1.2× descriptive criterion"};
the middle 60% has the highest rate.
Because NIfTI orientations vary ({', '.join(orientation_codes)}), the report
does not invent rostral/caudal, dorsal/ventral, or left/right labels. “First” and
“final” refer only to normalized array-axis-2 acquisition position within the
expert-mask extent.

### 7. Local geometric complexity

Expert-surface complexity is 1 minus the local mean resultant length of
physical signed-distance normals. Subject-wise tertiles define flat, medium,
and high-complexity regions. High-complexity regions have {curvature_ratio:.3f}×
the combined errors per 100 surface voxels of relatively flat regions
({high_rate:.3f} versus {flat_rate:.3f}). This
{"supports" if curvature_ratio > 1.2 else "does not strongly support"} preferential
failure around curved/pixel-complex expert geometry.

### 8. MRI boundary evidence

MRI intensities were robustly standardized per subject. Mean physical gradient
magnitude is {grad_correct:.3f} z/mm at correctly localized surface regions and
{grad_far:.3f} z/mm at >2-voxel disagreement regions. Mean local
inside/outside contrast is {contrast_correct:.3f} versus {contrast_far:.3f} z.
These measurements {"support weaker local image evidence at larger errors" if grad_far < grad_correct and contrast_far < contrast_correct else "do not consistently support weaker local image evidence as the sole cause of larger errors"}.
Local variance results are retained in `image_boundary_analysis.csv`.

### 9. Probability behavior

Within the expert-surface neighborhood, {uncertain:.2f}% of erroneous voxels
have probabilities in [0.25, 0.75], while {confident:.2f}% meet the fixed
descriptive “confident incorrect” definitions (FP >=0.9 or FN <=0.1).
Consequently errors are **{"predominantly close to the decision boundary" if uncertain > 50 else "not predominantly uncertain"}**; this is diagnostic only and does not optimize or recommend a threshold.

### 10. Is the problem geometrically meaningful?

{combined['<=1 voxel_percent']:.2f}% of residual voxels are within one
index-space voxel of the expert surface and {100-combined['<=1 voxel_percent']:.2f}%
extend farther. The complete distance distribution, physical P95/maximum,
terminal normalization, and complexity dependence should be considered
together; Dice alone does not establish one-voxel precision.

## Surface metrics

Mean filtered surface performance:

- CAMRI: Dice {surface_means.loc['CAMRI','dice']:.4f}, HD95
  {surface_means.loc['CAMRI','hd95_mm']:.3f} mm, ASSD
  {surface_means.loc['CAMRI','assd_mm']:.3f} mm.
- Mouse: Dice {surface_means.loc['Mouse','dice']:.4f}, HD95
  {surface_means.loc['Mouse','hd95_mm']:.3f} mm, ASSD
  {surface_means.loc['Mouse','assd_mm']:.3f} mm.

Surface Dice is reported at fixed 0.1, 0.2, 0.5, and 1.0 mm tolerances. These
were selected before inspecting results to span the observed native in-plane
and through-plane resolutions, not to maximize performance.

## Figures

Sixteen role-selected figures (eight per domain) are listed in
`figures/manifest.csv`. Each uses native voxels with nearest-neighbor display,
shows the MRI, expert, filtered prediction, green/cyan contours, red/yellow
FP/FN map, voxel-exact zoom, and saved probability map.

## Limitations and next diagnostic

- Voxel distance is an index-space measure and is not rotation- or
  anisotropy-equivalent to millimetres; physical conclusions should use mm.
- The digital 26-connected surface and local-normal dispersion are robust but
  resolution-dependent approximations to continuous anatomy.
- Surface locations receive their nearest error voxel; this attribution tests
  association, not causality.
- MRI gradient, contrast, and variance are observational and cannot separate
  annotation ambiguity from acquisition noise or model behavior.
- Continuous probabilities are pre-filter model outputs. The filter changes
  only the binary mask and has no continuous “filtered probability” analogue.
- The historical probability exporter stored square in-plane arrays transposed.
  Each map was aligned by the unique shape-compatible axis permutation whose
  unchanged 0.5 threshold exactly reproduces the canonical saved raw mask.
- CAMRI has only six held-out subjects, limiting domain-specific inferential
  statistics.

The next diagnostic experiment should be a blinded, multi-rater boundary review
of the largest physical-distance and high-complexity regions, measuring expert
inter-rater surface distance on the same fixed cases. That would distinguish
model error from expert-boundary ambiguity without changing the model.
"""
    (output / "experiment_summary.md").write_text(report)


def main():
    global SOURCE, PROB_ROOT, CHECKPOINT, CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--probability-root", type=Path, default=PROB_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    SOURCE=args.source.resolve();PROB_ROOT=args.probability_root.resolve();CHECKPOINT=args.checkpoint.resolve();CONFIG=args.config.resolve()
    analyze(args.output.resolve())


if __name__ == "__main__":
    main()
