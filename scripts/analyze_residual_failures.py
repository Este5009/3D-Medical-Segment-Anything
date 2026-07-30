#!/usr/bin/env python3
"""Classify residual errors in all saved CAMRI and Mouse test predictions.

This is analysis only. It loads native MRI, expert-mask, and already generated
prediction NIfTI files. It never imports the model or inference pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label


ROOT = Path(__file__).resolve().parents[1]
CONNECTIVITY = generate_binary_structure(3, 3)  # deterministic 26-connectivity
CATEGORIES = (
    "Boundary Error",
    "Detached False Positive Island",
    "Detached False Negative Region",
    "Terminal Slice Failure",
    "Leakage",
    "Major Localization Failure",
)
ATLAS_DIRECTORIES = {
    "Boundary Error": "Boundary",
    "Detached False Positive Island": "Detached_FP",
    "Detached False Negative Region": "Detached_FN",
    "Terminal Slice Failure": "Terminal",
    "Leakage": "Leakage",
    "Major Localization Failure": "Localization",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer columns for empty table: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def robust_normalize(image):
    low, high = np.percentile(image, (1, 99))
    return np.clip((image - low) / max(high - low, 1e-8), 0, 1)


def component_bbox(mask):
    coordinates = np.argwhere(mask)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0) + 1
    return low, high


def world_point(affine, point):
    return nib.affines.apply_affine(affine, np.asarray(point, dtype=float))


def connected_component_rows(domain, subject, prediction, target, image, affine, spacing):
    labels, count = label(prediction, structure=CONNECTIVITY)
    if count == 0:
        return [], labels, None
    sizes = np.bincount(labels.ravel())[1:]
    ordered_ids = list(np.argsort(sizes)[::-1] + 1)
    largest_id = ordered_ids[0]
    largest = labels == largest_id
    largest_centroid = np.argwhere(largest).mean(axis=0)
    largest_world = world_point(affine, largest_centroid)
    distance_to_largest = distance_transform_edt(~largest, sampling=spacing)
    target_mean = float(image[target].mean())
    target_std = max(float(image[target].std()), 1e-8)
    voxel_volume = float(np.prod(spacing))
    rows = []
    for rank, component_id in enumerate(ordered_ids, 1):
        component = labels == component_id
        coordinates = np.argwhere(component)
        centroid = coordinates.mean(axis=0)
        centroid_world = world_point(affine, centroid)
        low, high = component_bbox(component)
        overlap = int((component & target).sum())
        raw_mean = float(image[component].mean())
        rows.append(
            {
                "domain": domain,
                "subject": subject,
                "component_rank": rank,
                "component_label": int(component_id),
                "is_largest_component": int(component_id == largest_id),
                "voxel_volume": int(component.sum()),
                "physical_volume_mm3": float(component.sum() * voxel_volume),
                "centroid_i": float(centroid[0]),
                "centroid_j": float(centroid[1]),
                "centroid_k": float(centroid[2]),
                "centroid_world_x": float(centroid_world[0]),
                "centroid_world_y": float(centroid_world[1]),
                "centroid_world_z": float(centroid_world[2]),
                "centroid_distance_to_largest_mm": float(
                    np.linalg.norm(centroid_world - largest_world)
                ),
                "minimum_distance_to_largest_mm": (
                    0.0 if component_id == largest_id else float(distance_to_largest[component].min())
                ),
                "bbox_i0": int(low[0]),
                "bbox_j0": int(low[1]),
                "bbox_k0": int(low[2]),
                "bbox_i1_exclusive": int(high[0]),
                "bbox_j1_exclusive": int(high[1]),
                "bbox_k1_exclusive": int(high[2]),
                "occupied_slices": int(np.unique(coordinates[:, 2]).size),
                "average_image_intensity": raw_mean,
                "brain_intensity_zscore": (raw_mean - target_mean) / target_std,
                "locally_brain_like": int(abs((raw_mean - target_mean) / target_std) <= 1.0),
                "expert_overlap_voxels": overlap,
                "expert_overlap_fraction": overlap / max(int(component.sum()), 1),
            }
        )
    return rows, labels, largest_id


def endpoint_information(mask, affine):
    occupied = np.where(mask.any(axis=(0, 1)))[0]
    if not occupied.size:
        return None
    endpoints = [int(occupied[0]), int(occupied[-1])]
    world_z = [
        float(world_point(affine, [mask.shape[0] / 2, mask.shape[1] / 2, index])[2])
        for index in endpoints
    ]
    inferior_position = int(np.argmin(world_z))
    superior_position = 1 - inferior_position
    return {
        "first_index": endpoints[0],
        "last_index": endpoints[1],
        "inferior_index": endpoints[inferior_position],
        "superior_index": endpoints[superior_position],
    }


def terminal_mask_and_stats(prediction, target, affine):
    pred_extent = endpoint_information(prediction, affine)
    target_extent = endpoint_information(target, affine)
    mask = np.zeros_like(target, dtype=bool)
    if pred_extent is None:
        mask[:] = target
        return mask, {
            "inferior_incorrect_slices": target_extent["last_index"] - target_extent["first_index"] + 1,
            "superior_incorrect_slices": 0,
            "first_incorrect_slice": target_extent["first_index"],
            "last_incorrect_slice": target_extent["last_index"],
        }
    endpoint_differences = {}
    incorrect_indices = []
    for anatomical_endpoint in ("inferior", "superior"):
        pred_index = pred_extent[f"{anatomical_endpoint}_index"]
        target_index = target_extent[f"{anatomical_endpoint}_index"]
        difference = abs(pred_index - target_index)
        endpoint_differences[f"{anatomical_endpoint}_incorrect_slices"] = difference
        if difference:
            low, high = sorted((pred_index, target_index))
            indices = list(range(low, high + 1))
            incorrect_indices.extend(indices)
            mask[:, :, low : high + 1] = prediction[:, :, low : high + 1] ^ target[:, :, low : high + 1]
    endpoint_differences["first_incorrect_slice"] = min(incorrect_indices) if incorrect_indices else -1
    endpoint_differences["last_incorrect_slice"] = max(incorrect_indices) if incorrect_indices else -1
    return mask, endpoint_differences


def error_components(mask):
    labels, count = label(mask, structure=CONNECTIVITY)
    return [labels == component_id for component_id in range(1, count + 1)]


def analyze_subject(row, boundary_mm):
    image_object = nib.load(row["image_path"])
    target_object = nib.load(row["ground_truth_path"])
    prediction_object = nib.load(row["prediction_path"])
    image = np.asarray(image_object.dataobj, dtype=np.float32)
    target = np.asarray(target_object.dataobj) > 0
    prediction = np.asarray(prediction_object.dataobj) > 0
    spacing = tuple(map(float, target_object.header.get_zooms()[:3]))
    affine = image_object.affine
    voxel_volume = float(np.prod(spacing))
    tp = prediction & target
    fp = prediction & ~target
    fn = ~prediction & target
    total_error = int(fp.sum() + fn.sum())

    component_rows, prediction_labels, largest_id = connected_component_rows(
        row["domain"], row["subject"], prediction, target, image, affine, spacing
    )
    largest_component = prediction_labels == largest_id if largest_id is not None else np.zeros_like(target)
    detached_fp_mask = np.zeros_like(target)
    for component in component_rows:
        if component["is_largest_component"] or component["expert_overlap_voxels"] > 0:
            continue
        component_mask = prediction_labels == component["component_label"]
        detached_fp_mask |= component_mask

    distance_to_target = distance_transform_edt(~target, sampling=spacing)
    distance_to_prediction = distance_transform_edt(~prediction, sampling=spacing)
    fp_boundary = fp & (distance_to_target <= boundary_mm)
    fn_boundary = fn & (distance_to_prediction <= boundary_mm)
    terminal_mask, terminal_stats = terminal_mask_and_stats(prediction, target, affine)
    terminal_error = terminal_mask & (fp | fn)

    leakage_mask = fp & largest_component & (distance_to_target > boundary_mm)
    detached_fn_mask = np.zeros_like(target)
    detached_fn_components = []
    for index, component in enumerate(error_components(fn), 1):
        far_fraction = float(np.mean(distance_to_prediction[component] > boundary_mm))
        terminal_fraction = float(np.mean(terminal_error[component])) if component.any() else 0.0
        if far_fraction >= 0.5 and terminal_fraction < 0.5:
            detached_fn_mask |= component
            coordinates = np.argwhere(component)
            detached_fn_components.append(
                {
                    "component_index": index,
                    "voxels": int(component.sum()),
                    "physical_volume_mm3": float(component.sum() * voxel_volume),
                    "occupied_slices": int(np.unique(coordinates[:, 2]).size),
                    "centroid_k": float(coordinates[:, 2].mean()),
                }
            )

    distant_unassigned = (fp | fn) & ~(
        detached_fp_mask | terminal_error | leakage_mask | detached_fn_mask | fp_boundary | fn_boundary
    )
    major_threshold = 0.05 * max(int(target.sum()), 1)
    major_localization = int(distant_unassigned.sum()) >= major_threshold or float(row["dice"]) < 0.90

    # Exclusive attribution priority. This ensures percentages sum to exactly
    # 100 even though a subject may carry several overlapping failure labels.
    remaining = fp | fn
    attributed = {}
    priority_masks = (
        ("Detached False Positive Island", detached_fp_mask),
        ("Terminal Slice Failure", terminal_error),
        ("Leakage", leakage_mask),
        ("Detached False Negative Region", detached_fn_mask),
        ("Major Localization Failure", distant_unassigned if major_localization else np.zeros_like(target)),
    )
    for category, category_mask in priority_masks:
        assigned = remaining & category_mask
        attributed[category] = int(assigned.sum())
        remaining &= ~assigned
    attributed["Boundary Error"] = int(remaining.sum())

    category_masks = {
        "Boundary Error": fp_boundary | fn_boundary,
        "Detached False Positive Island": detached_fp_mask,
        "Detached False Negative Region": detached_fn_mask,
        "Terminal Slice Failure": terminal_error,
        "Leakage": leakage_mask,
        "Major Localization Failure": distant_unassigned if major_localization else np.zeros_like(target),
    }
    # Boundary Error is a subject-level category only when local boundary errors
    # dominate and no major localization failure is present.
    category_affected = {
        category: bool(mask.any()) for category, mask in category_masks.items()
    }
    category_affected["Boundary Error"] = bool(
        (fp_boundary | fn_boundary).any() and not major_localization
    )

    subject_summary = {
        "domain": row["domain"],
        "subject": row["subject"],
        "dice": float(row["dice"]),
        "expert_voxels": int(target.sum()),
        "prediction_voxels": int(prediction.sum()),
        "false_positive_voxels": int(fp.sum()),
        "false_negative_voxels": int(fn.sum()),
        "total_error_voxels": total_error,
        "false_positive_volume_mm3": float(fp.sum() * voxel_volume),
        "false_negative_volume_mm3": float(fn.sum() * voxel_volume),
        "predicted_components": len(component_rows),
        "detached_fp_components": sum(
            not component["is_largest_component"] and component["expert_overlap_fraction"] == 0
            for component in component_rows
        ),
        "detached_fp_voxels": int(detached_fp_mask.sum()),
        "detached_fn_components": len(detached_fn_components),
        "detached_fn_voxels": int(detached_fn_mask.sum()),
        "terminal_error_voxels": int(terminal_error.sum()),
        "leakage_voxels": int(leakage_mask.sum()),
        "major_localization_voxels": int(category_masks["Major Localization Failure"].sum()),
        "boundary_error_voxels": int((fp_boundary | fn_boundary).sum()),
        **terminal_stats,
        **{f"affected_{ATLAS_DIRECTORIES[category].lower()}": int(affected) for category, affected in category_affected.items()},
        **{f"attributed_{ATLAS_DIRECTORIES[category].lower()}_voxels": count for category, count in attributed.items()},
        "image_path": row["image_path"],
        "ground_truth_path": row["ground_truth_path"],
        "prediction_path": row["prediction_path"],
    }
    return {
        "summary": subject_summary,
        "components": component_rows,
        "detached_fn_components": detached_fn_components,
        "category_masks": category_masks,
        "category_affected": category_affected,
        "attributed": attributed,
        "image": image,
        "normalized_image": robust_normalize(image),
        "target": target,
        "prediction": prediction,
        "prediction_labels": prediction_labels,
        "largest_component": largest_component,
        "voxel_volume_mm3": voxel_volume,
        "spacing": spacing,
        "affine": affine,
    }


def instance_metric(category, analysis):
    summary = analysis["summary"]
    mapping = {
        "Boundary Error": summary["boundary_error_voxels"],
        "Detached False Positive Island": summary["detached_fp_voxels"],
        "Detached False Negative Region": summary["detached_fn_voxels"],
        "Terminal Slice Failure": summary["terminal_error_voxels"],
        "Leakage": summary["leakage_voxels"],
        "Major Localization Failure": summary["major_localization_voxels"],
    }
    return mapping[category]


def component_details_for_category(category, analysis):
    if category == "Detached False Positive Island":
        components = [
            row for row in analysis["components"] if not row["is_largest_component"]
        ]
        if components:
            return max(components, key=lambda row: row["voxel_volume"])
    if category == "Detached False Negative Region" and analysis["detached_fn_components"]:
        return max(analysis["detached_fn_components"], key=lambda row: row["voxels"])
    return None


def representative_slice(category, analysis):
    mask = analysis["category_masks"][category]
    if mask.any():
        return int(np.argmax(mask.sum(axis=(0, 1))))
    disagreement = analysis["prediction"] ^ analysis["target"]
    return int(np.argmax(disagreement.sum(axis=(0, 1))))


def explanation(category, analysis):
    summary = analysis["summary"]
    details = component_details_for_category(category, analysis)
    if category == "Boundary Error":
        return (
            f"{summary['boundary_error_voxels']:,} error voxels lie within the "
            "0.5 mm contour band; global localization remains correct."
        )
    if category == "Detached False Positive Island":
        return (
            f"Detached component: {details['physical_volume_mm3']:.2f} mm3, "
            f"{details['centroid_distance_to_largest_mm']:.2f} mm centroid distance, "
            f"{details['occupied_slices']} slices, no expert overlap."
        )
    if category == "Detached False Negative Region":
        return (
            f"A {details['physical_volume_mm3']:.2f} mm3 expert region is missing "
            f"across {details['occupied_slices']} slices away from the local contour band."
        )
    if category == "Terminal Slice Failure":
        return (
            f"Prediction/expert endpoints differ by "
            f"{summary['inferior_incorrect_slices']} inferior and "
            f"{summary['superior_incorrect_slices']} superior slices "
            f"(indices {summary['first_incorrect_slice']}–{summary['last_incorrect_slice']})."
        )
    if category == "Leakage":
        return (
            f"{summary['leakage_voxels']:,} false-positive voxels extend more than "
            "0.5 mm from expert anatomy while remaining connected to the main component."
        )
    return (
        f"Distant unassigned error is {summary['major_localization_voxels']:,} voxels "
        f"({100*summary['major_localization_voxels']/summary['expert_voxels']:.1f}% "
        "of expert volume)."
    )


def save_atlas_figure(category, analysis, severity, destination):
    index = representative_slice(category, analysis)
    image = analysis["normalized_image"][:, :, index]
    target = analysis["target"][:, :, index]
    prediction = analysis["prediction"][:, :, index]
    labels = analysis["prediction_labels"][:, :, index]
    overlay = np.zeros((*target.shape, 3), dtype=float)
    overlay[prediction & target] = (0.1, 0.8, 0.2)
    overlay[prediction & ~target] = (1.0, 0.15, 0.1)
    overlay[~prediction & target] = (0.1, 0.35, 1.0)
    component_colors = plt.get_cmap("tab20")(
        (labels.astype(int) % 20) / 19.0
    )[:, :, :3]
    component_colors[labels == 0] = 0
    responsible = analysis["category_masks"][category][:, :, index]
    component_colors[responsible] = (1.0, 1.0, 0.0)

    figure, axes = plt.subplots(1, 5, figsize=(15, 3.6), constrained_layout=True)
    panels = (image, target, prediction, overlay, component_colors)
    titles = (
        "Original MRI",
        "Expert mask",
        "Prediction",
        "TP green | FP red | FN blue",
        "Components; failure yellow",
    )
    cmaps = ("gray", "gray", "gray", None, None)
    for axis, panel, title, cmap in zip(axes, panels, titles, cmaps):
        axis.imshow(panel.transpose(1, 0, 2) if panel.ndim == 3 else panel.T, cmap=cmap, origin="lower")
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    summary = analysis["summary"]
    figure.suptitle(
        f"{summary['subject']} | {summary['domain']} | Dice {summary['dice']:.4f} | "
        f"{category} | {severity} | slice {index}",
        fontsize=11,
    )
    figure.text(0.5, 0.01, explanation(category, analysis), ha="center", fontsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def select_representatives(category, analyses):
    eligible = [analysis for analysis in analyses if analysis["category_affected"][category]]
    eligible.sort(key=lambda analysis: instance_metric(category, analysis))
    if len(eligible) <= 3:
        labels = ("mild", "representative", "severe")[-len(eligible) :]
        return list(zip(labels, eligible))
    indices = (0, len(eligible) // 2, len(eligible) - 1)
    return list(zip(("mild", "representative", "severe"), [eligible[index] for index in indices]))


def aggregate_failure_statistics(analyses, component_rows):
    total_subjects = len(analyses)
    total_error = sum(analysis["summary"]["total_error_voxels"] for analysis in analyses)
    statistics = []
    for category in CATEGORIES:
        affected = [analysis for analysis in analyses if analysis["category_affected"][category]]
        component_sizes = []
        occupied_slices = []
        distances = []
        for analysis in affected:
            if category == "Detached False Positive Island":
                rows = [
                    row
                    for row in analysis["components"]
                    if not row["is_largest_component"] and row["expert_overlap_voxels"] == 0
                ]
                component_sizes.extend(row["voxel_volume"] for row in rows)
                occupied_slices.extend(row["occupied_slices"] for row in rows)
                distances.extend(row["centroid_distance_to_largest_mm"] for row in rows)
            elif category == "Detached False Negative Region":
                rows = analysis["detached_fn_components"]
                component_sizes.extend(row["voxels"] for row in rows)
                occupied_slices.extend(row["occupied_slices"] for row in rows)
            else:
                category_mask = analysis["category_masks"][category]
                component_sizes.extend(int(component.sum()) for component in error_components(category_mask))
                occupied_slices.append(
                    int(np.count_nonzero(category_mask.any(axis=(0, 1))))
                )
                if category_mask.any() and analysis["largest_component"].any():
                    error_centroid = np.argwhere(category_mask).mean(axis=0)
                    brain_centroid = np.argwhere(analysis["largest_component"]).mean(axis=0)
                    error_world = world_point(analysis["affine"], error_centroid)
                    brain_world = world_point(analysis["affine"], brain_centroid)
                    distances.append(float(np.linalg.norm(error_world - brain_world)))
        attributed = sum(analysis["attributed"][category] for analysis in analyses)
        fp_volumes = [
            float(
                (
                    analysis["category_masks"][category]
                    & analysis["prediction"]
                    & ~analysis["target"]
                ).sum()
                * analysis["voxel_volume_mm3"]
            )
            for analysis in affected
        ]
        fn_volumes = [
            float(
                (
                    analysis["category_masks"][category]
                    & ~analysis["prediction"]
                    & analysis["target"]
                ).sum()
                * analysis["voxel_volume_mm3"]
            )
            for analysis in affected
        ]
        statistics.append(
            {
                "failure_category": category,
                "affected_subjects": len(affected),
                "affected_subject_percentage": 100.0 * len(affected) / total_subjects,
                "average_component_size_voxels": float(np.mean(component_sizes)) if component_sizes else 0.0,
                "average_slices_occupied": float(np.mean(occupied_slices)) if occupied_slices else 0.0,
                "average_centroid_distance_mm": float(np.mean(distances)) if distances else 0.0,
                "average_false_positive_volume_mm3": float(np.mean(fp_volumes)) if fp_volumes else 0.0,
                "average_false_negative_volume_mm3": float(np.mean(fn_volumes)) if fn_volumes else 0.0,
                "attributed_error_voxels": attributed,
                "percentage_total_error_attributed": 100.0 * attributed / max(total_error, 1),
            }
        )
    return statistics


def grouping_summary(analyses):
    islands = [
        component
        for analysis in analyses
        for component in analysis["components"]
        if not component["is_largest_component"] and component["expert_overlap_fraction"] == 0
    ]
    brain_like = [component for component in islands if component["locally_brain_like"]]
    island_voxels = sum(component["voxel_volume"] for component in islands)
    brain_like_voxels = sum(component["voxel_volume"] for component in brain_like)
    total_error = sum(analysis["summary"]["total_error_voxels"] for analysis in analyses)
    grouping_categories = (
        "Detached False Positive Island",
        "Detached False Negative Region",
        "Leakage",
        "Major Localization Failure",
    )
    grouping_attributed = sum(
        analysis["attributed"][category]
        for analysis in analyses
        for category in grouping_categories
    )
    return {
        "detached_fp_islands": len(islands),
        "detached_fp_island_voxels": island_voxels,
        "locally_brain_like_detached_islands": len(brain_like),
        "locally_brain_like_detached_island_voxels": brain_like_voxels,
        "islands_theoretically_removable_by_global_grouping": len(brain_like),
        "island_voxels_theoretically_removable_by_global_grouping": brain_like_voxels,
        "grouping_attributed_error_voxels": grouping_attributed,
        "grouping_attributed_error_percentage": 100.0 * grouping_attributed / max(total_error, 1),
        "total_error_voxels": total_error,
    }


def write_report(statistics, grouping, analyses, output):
    stat = {row["failure_category"]: row for row in statistics}
    boundary = stat["Boundary Error"]["percentage_total_error_attributed"]
    grouping_percent = grouping["grouping_attributed_error_percentage"]
    terminal = stat["Terminal Slice Failure"]["percentage_total_error_attributed"]
    if boundary > 60 and grouping_percent < 20:
        recommendation = "A. Current architecture is primarily boundary-limited."
    elif grouping_percent > 60 and boundary < 20:
        recommendation = "B. Current architecture is primarily anatomical-grouping limited."
    else:
        recommendation = "C. Current architecture is limited by both."
    table = "\n".join(
        f"| {row['failure_category']} | {row['affected_subjects']} "
        f"({row['affected_subject_percentage']:.1f}%) | "
        f"{row['average_component_size_voxels']:.1f} | "
        f"{row['average_slices_occupied']:.2f} | "
        f"{row['average_centroid_distance_mm']:.2f} | "
        f"{row['percentage_total_error_attributed']:.2f}% |"
        for row in statistics
    )
    text = f"""# Residual failure analysis

## Evaluation contract

All {len(analyses)} untouched test predictions (6 CAMRI, 80 Mouse) were analyzed
in native space. No inference, training, threshold change, cleanup, or
post-processing was performed. Components use 26-connectivity. Local boundary
errors are defined within 0.5 mm of the opposite mask.

## Failure statistics

| Category | Affected subjects | Mean size, voxels | Mean slices | Mean centroid distance, mm | Exclusive error attribution |
|---|---:|---:|---:|---:|---:|
{table}

Exclusive attribution uses this order: detached FP, terminal, leakage, detached
FN, major localization, then remaining boundary/local contour error. Therefore
the attribution percentages sum to 100%, although subjects may appear in more
than one category.

## Anatomical grouping evidence

There were {grouping['detached_fp_islands']} detached predicted islands with no
expert overlap, containing {grouping['detached_fp_island_voxels']:,} voxels.
{grouping['locally_brain_like_detached_islands']} islands
({grouping['locally_brain_like_detached_island_voxels']:,} voxels) had mean MRI
intensity within one expert-brain standard deviation. These are the measured
cases where local appearance could support foreground while 3D disconnection
contradicts membership. At most those
{grouping['islands_theoretically_removable_by_global_grouping']} islands are
theoretically removable by stronger global grouping from the collected image
and topology evidence.

Grouping-associated categories account for {grouping_percent:.2f}% of all
exclusive error voxels. Boundary/local contour errors account for
{boundary:.2f}%, and terminal endpoint errors account for {terminal:.2f}%.

## Scientific interpretation

1. **Dominant remaining failure mode:** the largest exclusive attribution is
   {max(statistics,key=lambda row:row['percentage_total_error_attributed'])['failure_category']}
   at {max(row['percentage_total_error_attributed'] for row in statistics):.2f}%.
2. **Is boundary prediction the primary limitation?** Boundary/local contour
   disagreement accounts for {boundary:.2f}% of measured error voxels.
3. **Is anatomical grouping becoming the primary limitation?** Grouping-related
   errors account for {grouping_percent:.2f}%; this is compared directly with
   the boundary fraction rather than inferred from Dice.
4. **Would stronger boundary supervision remove most errors?** It could address
   the measured {boundary:.2f}% boundary/local fraction, but not detached,
   terminal, leakage, or distant localization voxels.
5. **Would explicit 3D grouping remove a substantial fraction?** The evidence
   supports at most {grouping_percent:.2f}% of current error voxels, including
   {grouping['locally_brain_like_detached_islands']} locally brain-like detached
   islands. This is an upper attribution, not a guaranteed improvement.
6. **Estimated grouping insufficiency:** {grouping_percent:.2f}% of remaining
   errors under the declared exclusive classification.

## Recommendation

**{recommendation}**

This is the sole recommendation and follows the measured boundary versus
grouping attribution above.
"""
    (output / "failure_summary.md").write_text(text)
    return recommendation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory",
        default="outputs/mixed_domain_anatomical_training/fast_evaluation",
    )
    parser.add_argument(
        "--output-directory",
        default="outputs/mixed_domain_anatomical_training/failure_analysis",
    )
    parser.add_argument("--boundary-mm", type=float, default=0.5)
    args = parser.parse_args()
    input_directory = ROOT / args.input_directory
    output = ROOT / args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input_directory / "camri_subject_dice.csv") + read_csv(
        input_directory / "mouse_subject_dice.csv"
    )
    analyses = []
    component_rows = []
    for index, row in enumerate(rows, 1):
        analysis = analyze_subject(row, args.boundary_mm)
        analyses.append(analysis)
        component_rows.extend(analysis["components"])
        print(f"analyzed {index}/{len(rows)}: {row['domain']} {row['subject']}", flush=True)

    write_csv(output / "connected_component_statistics.csv", component_rows)
    write_csv(output / "subject_failure_classification.csv", [analysis["summary"] for analysis in analyses])
    statistics = aggregate_failure_statistics(analyses, component_rows)
    write_csv(output / "failure_statistics.csv", statistics)
    grouping = grouping_summary(analyses)

    atlas_manifest = []
    for category in CATEGORIES:
        representatives = select_representatives(category, analyses)
        for number, (severity, analysis) in enumerate(representatives, 1):
            destination = (
                output
                / "Failure_Atlas"
                / ATLAS_DIRECTORIES[category]
                / f"example_{number:02d}.png"
            )
            save_atlas_figure(category, analysis, severity, destination)
            atlas_manifest.append(
                {
                    "failure_category": category,
                    "severity": severity,
                    "domain": analysis["summary"]["domain"],
                    "subject": analysis["summary"]["subject"],
                    "dice": analysis["summary"]["dice"],
                    "severity_voxels": instance_metric(category, analysis),
                    "figure_path": str(destination),
                    "explanation": explanation(category, analysis),
                }
            )
    write_csv(
        output / "Failure_Atlas" / "manifest.csv",
        atlas_manifest,
        fieldnames=(
            "failure_category",
            "severity",
            "domain",
            "subject",
            "dice",
            "severity_voxels",
            "figure_path",
            "explanation",
        ),
    )
    recommendation = write_report(statistics, grouping, analyses, output)
    summary = {
        "subjects_analyzed": len(analyses),
        "camri_subjects": sum(analysis["summary"]["domain"] == "CAMRI" for analysis in analyses),
        "mouse_subjects": sum(analysis["summary"]["domain"] == "Mouse" for analysis in analyses),
        "boundary_mm": args.boundary_mm,
        "connectivity": 26,
        "failure_statistics": statistics,
        "grouping_analysis": grouping,
        "atlas_figures": len(atlas_manifest),
        "recommendation": recommendation,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
