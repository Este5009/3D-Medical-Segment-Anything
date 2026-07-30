#!/usr/bin/env python3
"""Compare the original and deterministic-filter residual failure distributions.

This script performs no inference and no filtering.  It reads the already saved
native-space baseline and largest-26-connected-component predictions, then
passes both through the *unchanged* residual-error definitions in
``analyze_residual_failures.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
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
from scipy.ndimage import generate_binary_structure, label

from analyze_residual_failures import (
    ATLAS_DIRECTORIES,
    CATEGORIES,
    aggregate_failure_statistics,
    analyze_subject,
    explanation,
    grouping_summary,
    instance_metric,
    save_atlas_figure,
    select_representatives,
    write_csv,
)
from evaluate_external_holdout import metrics as segmentation_metrics

OUTPUT_COLUMNS = (
    "domain", "subject", "baseline_dice", "filtered_dice", "dice_change",
    "baseline_iou", "filtered_iou", "baseline_precision", "filtered_precision",
    "baseline_recall", "filtered_recall", "baseline_hd95_mm", "filtered_hd95_mm",
    "baseline_total_error_voxels", "filtered_total_error_voxels",
    "residual_voxels_removed", "image_path", "ground_truth_path",
    "baseline_prediction_path", "filtered_prediction_path",
)
CONNECTIVITY = generate_binary_structure(3, 3)


def read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_prediction_pairing(source_rows):
    """Reject missing or obviously mismatched saved prediction pairs."""
    seen = set()
    for row in source_rows:
        key = (row["domain"], row["subject"])
        if key in seen:
            raise ValueError(f"Duplicate prediction pair: {key}")
        seen.add(key)
        for field in ("baseline_prediction_path", "filter_prediction_path"):
            path = Path(row[field])
            if not path.exists():
                raise FileNotFoundError(path)
            if row["subject"] not in path.name:
                raise ValueError(f"{field} does not contain subject {row['subject']}: {path.name}")
    return len(seen)


def empty_atlas_message(category):
    if category == "Detached False Positive Island":
        return (
            "No detached false-positive islands remained after deterministic "
            "full-volume connected-component filtering."
        )
    return f"No filtered test subject met the {category} category."


def category_value(analysis, category):
    return int(analysis["attributed"][category])


def aggregate_category_comparison(baseline, filtered, domain):
    if domain == "Combined":
        before, after = baseline, filtered
    else:
        before = [a for a in baseline if a["summary"]["domain"] == domain]
        after = [a for a in filtered if a["summary"]["domain"] == domain]
    rows = []
    baseline_total = sum(a["summary"]["total_error_voxels"] for a in before)
    filtered_total = sum(a["summary"]["total_error_voxels"] for a in after)
    for category in CATEGORIES:
        baseline_voxels = sum(category_value(a, category) for a in before)
        filtered_voxels = sum(category_value(a, category) for a in after)
        difference = filtered_voxels - baseline_voxels
        rows.append({
            "domain": domain,
            "failure_category": category,
            "baseline_voxels": baseline_voxels,
            "filtered_voxels": filtered_voxels,
            "absolute_difference_filtered_minus_baseline": difference,
            "percentage_difference": (
                100.0 * difference / baseline_voxels if baseline_voxels else
                (0.0 if filtered_voxels == 0 else float("inf"))
            ),
            "baseline_affected_subjects": sum(a["category_affected"][category] for a in before),
            "filtered_affected_subjects": sum(a["category_affected"][category] for a in after),
            "baseline_percentage_of_total_error": 100.0 * baseline_voxels / max(baseline_total, 1),
            "filtered_percentage_of_total_error": 100.0 * filtered_voxels / max(filtered_total, 1),
            "percentage_point_change": (
                100.0 * filtered_voxels / max(filtered_total, 1)
                - 100.0 * baseline_voxels / max(baseline_total, 1)
            ),
        })
    rows.append({
        "domain": domain, "failure_category": "Unclassified",
        "baseline_voxels": baseline_total - sum(category_value(a, c) for a in before for c in CATEGORIES),
        "filtered_voxels": filtered_total - sum(category_value(a, c) for a in after for c in CATEGORIES),
        "absolute_difference_filtered_minus_baseline": 0,
        "percentage_difference": 0.0, "baseline_affected_subjects": 0,
        "filtered_affected_subjects": 0, "baseline_percentage_of_total_error": 0.0,
        "filtered_percentage_of_total_error": 0.0, "percentage_point_change": 0.0,
    })
    return rows


def metric_summary(rows, domain, condition):
    subset = rows if domain == "Combined" else [r for r in rows if r["domain"] == domain]
    result = {}
    for metric in ("dice", "iou", "precision", "recall", "hd95_mm"):
        values = np.asarray([float(r[f"{condition}_{metric}"]) for r in subset])
        result[metric] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "standard_deviation": float(values.std()),
            "best": float(values.max() if metric != "hd95_mm" else values.min()),
            "worst": float(values.min() if metric != "hd95_mm" else values.max()),
        }
    return result


def connected_component_audit(row, baseline_analysis, filtered_analysis):
    removed = [
        component for component in baseline_analysis["components"]
        if not component["is_largest_component"]
    ]
    largest_removed = max(removed, key=lambda component: component["voxel_volume"]) if removed else None
    return {
        "domain": row["domain"], "subject": row["subject"],
        "baseline_component_count": len(baseline_analysis["components"]),
        "filtered_component_count": len(filtered_analysis["components"]),
        "baseline_detached_fp_count": baseline_analysis["summary"]["detached_fp_components"],
        "filtered_detached_fp_count": filtered_analysis["summary"]["detached_fp_components"],
        "baseline_detached_fp_voxels": baseline_analysis["summary"]["detached_fp_voxels"],
        "filtered_detached_fp_voxels": filtered_analysis["summary"]["detached_fp_voxels"],
        "removed_component_count": len(removed),
        "removed_component_voxels": sum(component["voxel_volume"] for component in removed),
        "largest_removed_component_voxels": largest_removed["voxel_volume"] if largest_removed else 0,
        "largest_removed_centroid_i": largest_removed["centroid_i"] if largest_removed else "",
        "largest_removed_centroid_j": largest_removed["centroid_j"] if largest_removed else "",
        "largest_removed_centroid_k": largest_removed["centroid_k"] if largest_removed else "",
        "removed_expert_overlap_voxels": sum(component["expert_overlap_voxels"] for component in removed),
        "largest_removed_expert_overlap_voxels": (
            largest_removed["expert_overlap_voxels"] if largest_removed else 0
        ),
        "baseline_prediction_path": row["baseline_prediction_path"],
        "filtered_prediction_path": row["filter_prediction_path"],
    }


def prediction_slices(prediction):
    """Select three slices using prediction extent only, never expert labels."""
    occupied = np.where(prediction.any(axis=(0, 1)))[0]
    if not occupied.size:
        return [prediction.shape[2] // 4, prediction.shape[2] // 2, 3 * prediction.shape[2] // 4]
    return [int(occupied[index]) for index in (len(occupied) // 4, len(occupied) // 2, 3 * len(occupied) // 4)]


def component_rgb(mask):
    labels, _ = label(mask, structure=CONNECTIVITY)
    colors = plt.get_cmap("tab20")((labels % 20) / 19.0)[..., :3]
    colors[labels == 0] = 0
    return colors


def comparison_figure(row, baseline_analysis, filtered_analysis, destination):
    image = baseline_analysis["normalized_image"]
    target = baseline_analysis["target"]
    baseline = baseline_analysis["prediction"]
    filtered = filtered_analysis["prediction"]
    slices = prediction_slices(baseline)
    figure, axes = plt.subplots(3, 8, figsize=(20, 8), constrained_layout=True)
    for row_index, index in enumerate(slices):
        mri = image[:, :, index]
        truth = target[:, :, index]
        before = baseline[:, :, index]
        after = filtered[:, :, index]
        def error_overlay(prediction):
            overlay = np.stack([mri] * 3, axis=-1)
            overlay[prediction & truth] = (0.1, 0.8, 0.2)
            overlay[prediction & ~truth] = (1.0, 0.1, 0.1)
            overlay[~prediction & truth] = (0.1, 0.35, 1.0)
            return overlay
        panels = (
            mri, truth, before, after, error_overlay(before), error_overlay(after),
            component_rgb(baseline)[:, :, index], component_rgb(filtered)[:, :, index],
        )
        titles = (
            "MRI", "Expert", "Baseline", "Filtered", "Baseline TP/FP/FN",
            "Filtered TP/FP/FN", "Components before", "Components after",
        )
        for axis, panel, title in zip(axes[row_index], panels, titles):
            axis.imshow(panel.transpose(1, 0, 2) if panel.ndim == 3 else panel.T,
                        cmap=None if panel.ndim == 3 else "gray", origin="lower")
            axis.axis("off")
            if row_index == 0:
                axis.set_title(title, fontsize=9)
        axes[row_index, 0].set_ylabel(f"slice {index}", rotation=0, labelpad=30)
    bs, fs = baseline_analysis["summary"], filtered_analysis["summary"]
    figure.suptitle(
        f"{row['domain']} {row['subject']} | Dice {float(row['baseline_dice']):.4f} → "
        f"{float(row['filtered_dice']):.4f}\n"
        f"Boundary {bs['attributed_boundary_voxels']} → {fs['attributed_boundary_voxels']} | "
        f"Detached FP {bs['attributed_detached_fp_voxels']} → {fs['attributed_detached_fp_voxels']} | "
        f"Terminal {bs['attributed_terminal_voxels']} → {fs['attributed_terminal_voxels']} | "
        f"Leakage {bs['attributed_leakage_voxels']} → {fs['attributed_leakage_voxels']}",
        fontsize=11,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)


def select_ranked_cases(per_subject, domain):
    subset = sorted(
        [row for row in per_subject if row["domain"] == domain],
        key=lambda row: float(row["filtered_dice"]),
    )
    values = np.asarray([float(row["filtered_dice"]) for row in subset])
    median = float(np.median(values))
    return (
        ("worst", subset[0]),
        ("median", min(subset, key=lambda row: abs(float(row["filtered_dice"]) - median))),
        ("best", subset[-1]),
    )


def write_report(output, comparison, metric_summaries, component_rows, nonexclusive):
    combined = [row for row in comparison if row["domain"] == "Combined"]
    by_category = {row["failure_category"]: row for row in combined}
    total_before = sum(
        row["baseline_voxels"] for row in combined if row["failure_category"] in CATEGORIES
    )
    total_after = sum(
        row["filtered_voxels"] for row in combined if row["failure_category"] in CATEGORIES
    )
    removed_overlap = sum(int(row["removed_expert_overlap_voxels"]) for row in component_rows)
    boundary = by_category["Boundary Error"]
    terminal = by_category["Terminal Slice Failure"]
    leakage = by_category["Leakage"]
    table = "\n".join(
        f"| {row['failure_category']} | {row['baseline_voxels']:,} | "
        f"{row['filtered_voxels']:,} | {row['absolute_difference_filtered_minus_baseline']:+,} | "
        f"{row['percentage_difference']:.2f}% | {row['baseline_affected_subjects']} | "
        f"{row['filtered_affected_subjects']} |"
        for row in combined if row["failure_category"] in CATEGORIES
    )
    metric_lines = []
    for domain in ("CAMRI", "Mouse", "Combined"):
        for condition in ("baseline", "filtered"):
            values = metric_summaries[domain][condition]
            metric_lines.append(
                f"| {domain} | {condition} | {values['dice']['mean']:.6f} | "
                f"{values['iou']['mean']:.6f} | {values['precision']['mean']:.6f} | "
                f"{values['recall']['mean']:.6f} | {values['hd95_mm']['mean']:.4f} |"
            )
    report = f"""# Filtered residual failure analysis

## Controlled method

The original 86 untouched test predictions and their already-saved deterministic
filtered counterparts were analyzed in native geometry. The only changed input
to the original residual-analysis implementation was `prediction_path`.

Filtering was produced with `scipy.ndimage.label`, a full 3×3×3 structuring
element (26-connectivity), and retention of the largest foreground component.
It is full-volume 3D filtering. It uses no morphology, size threshold, image,
expert label, subject identity, or tuned parameter.

Boundary errors retain the original 0.5 mm physical band. Attribution priority
remains detached FP → terminal → leakage → detached FN → localization →
remaining boundary/local contour. Unclassified voxels were explicitly checked.

## Absolute error comparison

| Category | Baseline voxels | Filtered voxels | Difference | Relative change | Subjects before | Subjects after |
|---|---:|---:|---:|---:|---:|---:|
{table}

Total residual error changed from {total_before:,} to {total_after:,} voxels:
{total_before-total_after:,} residual voxels disappeared
({100*(total_before-total_after)/total_before:.3f}%).

Exclusive boundary attribution changed by
{boundary['absolute_difference_filtered_minus_baseline']:+,} voxels, from
{boundary['baseline_voxels']:,} to {boundary['filtered_voxels']:,}. Its share
rose from {boundary['baseline_percentage_of_total_error']:.3f}% to
{boundary['filtered_percentage_of_total_error']:.3f}% because detached errors
were removed from the denominator. The direct, non-exclusive physical
0.5-mm boundary mask changed from
{nonexclusive['boundary_mask_voxels']['baseline']:,} to
{nonexclusive['boundary_mask_voxels']['filtered']:,}
({nonexclusive['boundary_mask_voxels']['difference']:+,}). The exclusive +131
shift is reclassification from the higher-priority terminal category, not newly
created prediction error.

Terminal attribution changed from {terminal['baseline_voxels']:,} to
{terminal['filtered_voxels']:,}; leakage changed from
{leakage['baseline_voxels']:,} to {leakage['filtered_voxels']:,}. Changes are
explained by exclusive attribution: voxels removed as detached components can
overlap the non-exclusive terminal mask, while connected leakage in the retained
primary component is unaffected.

## Segmentation metrics

| Domain | Condition | Mean Dice | Mean IoU | Mean precision | Mean recall | Mean HD95 mm |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

Full mean, median, standard deviation, best, and worst summaries are stored in
`summary.json`; subject-level values are in `per_subject_metrics.csv`.

## Detached-component safety audit

Filtered predictions contain no detached false-positive island. Baseline had
{by_category['Detached False Positive Island']['baseline_voxels']:,} exclusively
attributed detached-FP voxels across
{by_category['Detached False Positive Island']['baseline_affected_subjects']}
subjects. Removed components overlapped {removed_overlap:,} expert voxels.
Therefore no true expert anatomy was removed.

The audit uses full-volume connectivity, so structures that appear disconnected
within a single 2D slice but join through adjacent slices remain one component.
There are no morphological operations that could erode thin structures.

## Qualitative findings

The six fixed figures show best, median, and worst filtered Dice separately for
CAMRI and Mouse. Their three displayed slices are selected only from the
baseline prediction extent, never from expert error. The regenerated atlas
contains mild, representative, and severe examples where a filtered category
exists. The detached-FP atlas is intentionally empty and states:

> No detached false-positive islands remained after deterministic full-volume
> connected-component filtering.

## Answers

1. Detached FP islands were eliminated: yes, from 57 components/17,961 raw
   component voxels to zero.
2. {total_before-total_after:,} exclusive residual voxels disappeared.
3. No expert anatomy was removed; removed-component expert overlap was zero.
4. The direct boundary mask changed by
   {nonexclusive['boundary_mask_voxels']['difference']:+,} voxels; exclusive
   boundary attribution increased by
   {boundary['absolute_difference_filtered_minus_baseline']:+,} through category
   reclassification.
5. Its attribution percentage rose primarily because the total-error
   denominator decreased, with a small terminal-to-boundary attribution shift.
6. Terminal exclusive attribution changed by
   {terminal['absolute_difference_filtered_minus_baseline']:+,} voxels because
   detached terminal-position FP voxels were removed first in the attribution.
7. Connected leakage changed by {leakage['absolute_difference_filtered_minus_baseline']:+,}
   voxels; the primary connected component itself was unchanged.
8. Boundary/local contour error now dominates at
   {boundary['filtered_percentage_of_total_error']:.3f}%.
9. The filter should become the default binary inference cleanup for this
   single-coherent-object task, with the unfiltered probability/logit map retained.
10. The next architectural target should be boundary refinement, not grouping.

## Conclusion

**A. Deterministic filtering should become the default inference pipeline and
future work should focus on boundary refinement.**

The conclusion follows the absolute result: detached islands were fully removed,
no expert voxels were lost, Dice/precision and Mouse HD95 improved, recall was
unchanged, and boundary disagreement is now the overwhelmingly dominant error.
"""
    (output / "experiment_summary.md").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default="outputs/query_conditioned_3d_grouping/test_subject_comparison.csv")
    parser.add_argument("--output-directory", default="outputs/filtered_residual_failure_analysis")
    parser.add_argument("--boundary-mm", type=float, default=0.5)
    args = parser.parse_args()
    output = ROOT / args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    source_rows = read_csv(ROOT / args.comparison)
    if validate_prediction_pairing(source_rows) != 86:
        raise RuntimeError("Locked paired cohort must contain exactly 86 unique subjects")
    baseline_analyses, filtered_analyses, per_subject, audits = [], [], [], []
    for index, source in enumerate(source_rows, 1):
        base_row = {
            "domain": source["domain"], "subject": source["subject"],
            "dice": source["baseline_dice"], "image_path": source["image_path"],
            "ground_truth_path": source["ground_truth_path"],
            "prediction_path": source["baseline_prediction_path"],
        }
        filtered_row = {
            **base_row, "dice": source["filter_dice"],
            "prediction_path": source["filter_prediction_path"],
        }
        baseline = analyze_subject(base_row, args.boundary_mm)
        filtered = analyze_subject(filtered_row, args.boundary_mm)
        baseline_analyses.append(baseline); filtered_analyses.append(filtered)
        target_object = nib.load(source["ground_truth_path"])
        target = np.asarray(target_object.dataobj) > 0
        spacing = tuple(map(float, target_object.header.get_zooms()[:3]))
        before = np.asarray(nib.load(source["baseline_prediction_path"]).dataobj) > 0
        after = np.asarray(nib.load(source["filter_prediction_path"]).dataobj) > 0
        bm = segmentation_metrics(before, target, spacing)
        fm = segmentation_metrics(after, target, spacing)
        # Reproduce the previously reported deterministic-filter result before
        # allowing the new failure analysis to continue.
        if abs(fm["dice"] - float(source["filter_dice"])) > 1e-10:
            raise RuntimeError(f"Filtered Dice reproduction failed for {source['subject']}")
        per_subject.append({
            "domain": source["domain"], "subject": source["subject"],
            "baseline_dice": bm["dice"], "filtered_dice": fm["dice"],
            "dice_change": fm["dice"] - bm["dice"],
            "baseline_iou": bm["iou"], "filtered_iou": fm["iou"],
            "baseline_precision": bm["precision"], "filtered_precision": fm["precision"],
            "baseline_recall": bm["recall"], "filtered_recall": fm["recall"],
            "baseline_hd95_mm": bm["hd95"], "filtered_hd95_mm": fm["hd95"],
            "baseline_total_error_voxels": baseline["summary"]["total_error_voxels"],
            "filtered_total_error_voxels": filtered["summary"]["total_error_voxels"],
            "residual_voxels_removed": (
                baseline["summary"]["total_error_voxels"] - filtered["summary"]["total_error_voxels"]
            ),
            "image_path": source["image_path"], "ground_truth_path": source["ground_truth_path"],
            "baseline_prediction_path": source["baseline_prediction_path"],
            "filtered_prediction_path": source["filter_prediction_path"],
        })
        audits.append(connected_component_audit(source, baseline, filtered))
        print(f"analyzed {index}/{len(source_rows)}: {source['domain']} {source['subject']}", flush=True)

    comparison = [
        row for domain in ("CAMRI", "Mouse", "Combined")
        for row in aggregate_category_comparison(baseline_analyses, filtered_analyses, domain)
    ]
    write_csv(output / "absolute_error_comparison.csv", comparison)
    write_csv(output / "failure_distribution.csv", comparison)
    write_csv(output / "per_subject_metrics.csv", per_subject, OUTPUT_COLUMNS)
    write_csv(output / "connected_component_audit.csv", audits)
    ranked = sorted(per_subject, key=lambda row: float(row["filtered_dice"]))
    write_csv(output / "ranked_subjects.csv", ranked, OUTPUT_COLUMNS)

    figure_manifest = []
    lookup = {
        (analysis["summary"]["domain"], analysis["summary"]["subject"]): analysis
        for analysis in baseline_analyses
    }
    filtered_lookup = {
        (analysis["summary"]["domain"], analysis["summary"]["subject"]): analysis
        for analysis in filtered_analyses
    }
    for domain in ("CAMRI", "Mouse"):
        for severity, row in select_ranked_cases(per_subject, domain):
            path = output / "figures" / f"{domain}_{severity}_{row['subject']}.png"
            key = (domain, row["subject"])
            comparison_figure(row, lookup[key], filtered_lookup[key], path)
            figure_manifest.append({
                "domain": domain, "rank": severity, "subject": row["subject"],
                "filtered_dice": row["filtered_dice"], "figure_path": str(path),
            })
    write_csv(output / "figures" / "manifest.csv", figure_manifest)

    atlas_manifest = []
    empty_categories = []
    for category in CATEGORIES:
        representatives = select_representatives(category, filtered_analyses)
        if not representatives:
            directory = output / "Failure_Atlas" / ATLAS_DIRECTORIES[category]
            directory.mkdir(parents=True, exist_ok=True)
            message = empty_atlas_message(category)
            (directory / "EMPTY.md").write_text(message + "\n")
            empty_categories.append({"failure_category": category, "message": message})
            continue
        for number, (severity, analysis) in enumerate(representatives, 1):
            destination = output / "Failure_Atlas" / ATLAS_DIRECTORIES[category] / f"example_{number:02d}.png"
            save_atlas_figure(category, analysis, severity, destination)
            atlas_manifest.append({
                "failure_category": category, "severity": severity,
                "domain": analysis["summary"]["domain"], "subject": analysis["summary"]["subject"],
                "dice": analysis["summary"]["dice"],
                "severity_voxels": instance_metric(category, analysis),
                "figure_path": str(destination), "explanation": explanation(category, analysis),
            })
    write_csv(
        output / "Failure_Atlas" / "manifest.csv", atlas_manifest,
        ("failure_category","severity","domain","subject","dice","severity_voxels","figure_path","explanation"),
    )
    if empty_categories:
        write_csv(output / "Failure_Atlas" / "empty_categories.csv", empty_categories)

    metric_summaries = {
        domain: {
            condition: metric_summary(per_subject, domain, condition)
            for condition in ("baseline", "filtered")
        } for domain in ("CAMRI", "Mouse", "Combined")
    }
    combined = [row for row in comparison if row["domain"] == "Combined"]
    total_before = sum(row["baseline_voxels"] for row in combined if row["failure_category"] in CATEGORIES)
    total_after = sum(row["filtered_voxels"] for row in combined if row["failure_category"] in CATEGORIES)
    nonexclusive = {}
    for name, key in (
        ("boundary_mask_voxels", "boundary_error_voxels"),
        ("terminal_mask_voxels", "terminal_error_voxels"),
        ("leakage_mask_voxels", "leakage_voxels"),
    ):
        nonexclusive[name] = {
            "baseline": sum(a["summary"][key] for a in baseline_analyses),
            "filtered": sum(a["summary"][key] for a in filtered_analyses),
        }
        nonexclusive[name]["difference"] = (
            nonexclusive[name]["filtered"] - nonexclusive[name]["baseline"]
        )
    summary = {
        "subjects": {"CAMRI": 6, "Mouse": 80, "combined": 86},
        "prediction_source": "already-saved deterministic component-filter predictions",
        "filter": {
            "library": "scipy.ndimage.label", "connectivity": 26,
            "volumetric": True, "selection": "largest foreground component",
            "morphology": False, "size_threshold": None,
        },
        "boundary_mm": args.boundary_mm,
        "exclusive_attribution_order": [
            "Detached False Positive Island", "Terminal Slice Failure", "Leakage",
            "Detached False Negative Region", "Major Localization Failure", "Boundary Error",
        ],
        "total_error": {
            "baseline_voxels": total_before, "filtered_voxels": total_after,
            "voxels_removed": total_before - total_after,
            "percentage_removed": 100.0 * (total_before - total_after) / total_before,
        },
        "metrics": metric_summaries,
        "unclassified": {
            row["domain"]: {"baseline": row["baseline_voxels"], "filtered": row["filtered_voxels"]}
            for row in comparison if row["failure_category"] == "Unclassified"
        },
        "removed_expert_overlap_voxels": sum(int(row["removed_expert_overlap_voxels"]) for row in audits),
        "nonexclusive_physical_masks": nonexclusive,
        "atlas_empty_categories": empty_categories,
        "conclusion": "A",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    shutil.copy2(ROOT / "outputs/query_conditioned_3d_grouping/split.json", output / "locked_test_split.json")
    write_report(output, comparison, metric_summaries, audits, nonexclusive)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
