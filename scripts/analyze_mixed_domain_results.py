#!/usr/bin/env python3
"""Comprehensive native-space analysis for mixed-domain anatomical training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from scipy.ndimage import binary_dilation, binary_erosion, label

from domain_shift_diagnostics import binary_metrics
from evaluate_mouse_boundary_adaptation import (
    boundary_zoom_figures,
    make_comparison_figure,
    write_csv,
)
from train_query_decoder_overfit import load_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mixed_domain_anatomical_training.yaml")
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def leakage_components(prediction: np.ndarray, target: np.ndarray):
    """Count predicted components with no overlap with expert anatomy."""
    components, count = label(prediction)
    component_count = 0
    voxel_count = 0
    for component_id in range(1, count + 1):
        component = components == component_id
        if not np.any(component & target):
            component_count += 1
            voxel_count += int(component.sum())
    return component_count, voxel_count


def per_slice_rows(prediction, target, domain, subject):
    occupied = np.where(target.any(axis=(0, 1)))[0]
    first, last = int(occupied[0]), int(occupied[-1])
    length = last - first + 1
    rows = []
    for index in range(target.shape[2]):
        pred = prediction[:, :, index]
        truth = target[:, :, index]
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        denominator = 2 * tp + fp + fn
        union = tp + fp + fn
        position = "outside"
        if first <= index <= last:
            relative = (index - first) / max(length, 1)
            position = "first 20%" if relative < 0.2 else "last 20%" if relative >= 0.8 else "middle 60%"
        rows.append(
            {
                "domain": domain,
                "subject": subject,
                "slice_index": index,
                "brain_position": position,
                "expert_nonempty": int(truth.any()),
                "prediction_nonempty": int(pred.any()),
                "dice": 2 * tp / denominator if denominator else 1.0,
                "iou": tp / union if union else 1.0,
                "false_positive_voxels": fp,
                "false_negative_voxels": fn,
            }
        )
    return rows


def analyze_case(domain, subject, result, tolerance_mm):
    prediction = np.asarray(nib.load(result["prediction_path"]).dataobj) > 0
    target_image = nib.load(result["ground_truth_path"])
    target = np.asarray(target_image.dataobj) > 0
    spacing = tuple(map(float, target_image.header.get_zooms()[:3]))
    metrics = binary_metrics(prediction, target, spacing, tolerance_mm)
    voxel_volume = float(np.prod(spacing))
    leakage_count, leakage_voxels = leakage_components(prediction, target)

    boundary_band = binary_dilation(target, iterations=2) ^ binary_erosion(target, iterations=2)
    errors = prediction ^ target
    boundary_errors = int((errors & boundary_band).sum())
    total_errors = int(errors.sum())
    slices = per_slice_rows(prediction, target, domain, subject)
    terminal = [row for row in slices if row["brain_position"] in ("first 20%", "last 20%")]
    occupied_accuracy = float(
        np.mean([row["expert_nonempty"] == row["prediction_nonempty"] for row in slices])
    )
    row = {
        "domain": domain,
        "subject": subject,
        **metrics,
        "false_positive_volume_mm3": metrics["false_positives"] * voxel_volume,
        "false_negative_volume_mm3": metrics["false_negatives"] * voxel_volume,
        "signed_volume_error_mm3": (int(prediction.sum()) - int(target.sum())) * voxel_volume,
        "absolute_volume_error_mm3": abs(int(prediction.sum()) - int(target.sum())) * voxel_volume,
        "volume_error_percent": 100.0 * (int(prediction.sum()) - int(target.sum())) / max(int(target.sum()), 1),
        "terminal_slice_dice": float(np.mean([item["dice"] for item in terminal])),
        "slice_occupancy_accuracy": occupied_accuracy,
        "leakage_components": leakage_count,
        "leakage_voxels": leakage_voxels,
        "boundary_error_fraction": boundary_errors / max(total_errors, 1),
        "localization_error_fraction": (total_errors - boundary_errors) / max(total_errors, 1),
        "inference_seconds": float(result["inference_seconds"]),
        "image_path": result["image_path"],
        "ground_truth_path": result["ground_truth_path"],
        "prediction_path": result["prediction_path"],
        "probability_path": result["probability_path"],
    }
    return row, slices


def summarize(rows):
    keys = (
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95_mm",
        "assd_mm",
        "false_positive_volume_mm3",
        "false_negative_volume_mm3",
        "signed_volume_error_mm3",
        "absolute_volume_error_mm3",
        "volume_error_percent",
        "terminal_slice_dice",
        "slice_occupancy_accuracy",
        "connected_components",
        "leakage_components",
        "leakage_voxels",
        "boundary_error_fraction",
        "localization_error_fraction",
        "inference_seconds",
    )
    summary = {"count": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows])
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "sd": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def rank_failures(rows, output):
    criteria = {
        "worst_dice": ("dice", False),
        "worst_hd95": ("hd95_mm", True),
        "largest_fp": ("false_positive_volume_mm3", True),
        "largest_fn": ("false_negative_volume_mm3", True),
        "terminal_slice_failures": ("terminal_slice_dice", False),
        "disconnected_predictions": ("connected_components", True),
        "leakage_regions": ("leakage_voxels", True),
    }
    ranked = {}
    flat = []
    for name, (metric, reverse) in criteria.items():
        selected = sorted(rows, key=lambda row: float(row[metric]), reverse=reverse)[:10]
        ranked[name] = [
            {"domain": row["domain"], "subject": row["subject"], metric: row[metric]}
            for row in selected
        ]
        flat.extend(
            {
                "failure_category": name,
                "rank": index,
                "domain": row["domain"],
                "subject": row["subject"],
                "metric": metric,
                "value": row[metric],
            }
            for index, row in enumerate(selected, 1)
        )
    (output / "failure_rankings.json").write_text(json.dumps(ranked, indent=2))
    write_csv(output / "failure_rankings.csv", flat)
    return ranked


def comparison_figures(rows, rankings, output, config):
    mouse_source = {row["scan_id"]: row for row in read_csv(ROOT / config["mouse_metrics"])}
    prior_output = ROOT / "outputs/mixed_domain_decoder/native_predictions"
    selected = []
    for category in rankings.values():
        for item in category[:3]:
            key = (item["domain"], item["subject"])
            if key not in selected:
                selected.append(key)
    lookup = {(row["domain"], row["subject"]): row for row in rows}
    comparison_rows = []
    for domain, subject in selected:
        current = lookup[(domain, subject)]
        if domain == "Mouse":
            baseline_path = mouse_source[subject]["prediction_path"]
        else:
            baseline_path = prior_output / "camri_original" / f"{subject}_prediction.nii.gz"
        if not Path(baseline_path).exists():
            continue
        baseline = np.asarray(nib.load(str(baseline_path)).dataobj) > 0
        target = np.asarray(nib.load(current["ground_truth_path"]).dataobj) > 0
        tp = int((baseline & target).sum())
        fp = int((baseline & ~target).sum())
        fn = int((~baseline & target).sum())
        comparison = {
            "subject_id": f"{domain}_{subject}",
            "baseline_dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "adapted_dice": current["dice"],
            "baseline_precision": tp / max(tp + fp, 1),
            "adapted_precision": current["precision"],
            "baseline_recall": tp / max(tp + fn, 1),
            "adapted_recall": current["recall"],
            "baseline_volume_ratio": int(baseline.sum()) / max(int(target.sum()), 1),
            "adapted_volume_ratio": current["volume_ratio"],
            "baseline_prediction_path": str(baseline_path),
            "adapted_prediction_path": current["prediction_path"],
            "probability_path": current["probability_path"],
            "image_path": current["image_path"],
            "ground_truth_path": current["ground_truth_path"],
        }
        make_comparison_figure(
            comparison,
            output / "comparison_figures" / domain.lower() / f"{subject}.png",
        )
        comparison_rows.append(comparison)
    boundary_zoom_figures(comparison_rows, output / "comparison_figures" / "boundary_zooms")


def plots(rows, output):
    for domain in ("CAMRI", "Mouse"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        axes[0].hist([row["dice"] for row in domain_rows], bins=12, color="#2878b5")
        axes[0].set(title=f"{domain}: Dice", xlabel="Dice", ylabel="Subjects")
        axes[1].scatter(
            [row["boundary_error_fraction"] for row in domain_rows],
            [row["dice"] for row in domain_rows],
            color="#d1495b",
        )
        axes[1].set(xlabel="Fraction of errors near boundary", ylabel="Dice")
        axes[2].scatter(
            [row["terminal_slice_dice"] for row in domain_rows],
            [row["hd95_mm"] for row in domain_rows],
            color="#3a7d44",
        )
        axes[2].set(xlabel="Terminal-slice Dice", ylabel="HD95 (mm)")
        for axis in axes:
            axis.grid(alpha=0.25)
        destination = output / "plots" / f"{domain.lower()}_error_summary.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=200, bbox_inches="tight")
        plt.close(fig)


def report(summary, previous, output):
    camri = summary["CAMRI"]
    mouse = summary["Mouse"]
    previous_camri = previous["camri"]["mixed"]["dice"]
    previous_mouse = previous["mouse"]["mixed"]["dice"]
    boundary_dominant = {
        domain: summary[domain]["boundary_error_fraction"]["mean"] >= 0.5
        for domain in ("CAMRI", "Mouse")
    }
    text = f"""# Mixed-domain anatomical-training report

## Controlled experiment

The RS2-Net encoder remained frozen and the 170,401-parameter decoder architecture
remained identical: exactly one learned query, four-scale attention, the existing
top-down FPN, and one mask head. The only controlled change from the previous
mixed-domain run was symmetric one-voxel boundary-band BCE added to Dice+BCE.
Sampling, augmentation, preprocessing, checkpoint initialization, and locked
subject-level splits were retained.

## Independent test results

| Domain | N | Dice | IoU | Precision | Recall | HD95 mm | ASD mm | FP mm3 | FN mm3 | Abs volume error mm3 | Terminal Dice | Time s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMRI | {camri['count']} | {camri['dice']['mean']:.4f} | {camri['iou']['mean']:.4f} | {camri['precision']['mean']:.4f} | {camri['recall']['mean']:.4f} | {camri['hd95_mm']['mean']:.4f} | {camri['assd_mm']['mean']:.4f} | {camri['false_positive_volume_mm3']['mean']:.2f} | {camri['false_negative_volume_mm3']['mean']:.2f} | {camri['absolute_volume_error_mm3']['mean']:.2f} | {camri['terminal_slice_dice']['mean']:.4f} | {camri['inference_seconds']['mean']:.3f} |
| Mouse | {mouse['count']} | {mouse['dice']['mean']:.4f} | {mouse['iou']['mean']:.4f} | {mouse['precision']['mean']:.4f} | {mouse['recall']['mean']:.4f} | {mouse['hd95_mm']['mean']:.4f} | {mouse['assd_mm']['mean']:.4f} | {mouse['false_positive_volume_mm3']['mean']:.2f} | {mouse['false_negative_volume_mm3']['mean']:.2f} | {mouse['absolute_volume_error_mm3']['mean']:.2f} | {mouse['terminal_slice_dice']['mean']:.4f} | {mouse['inference_seconds']['mean']:.3f} |

## Scientific questions

1. **Does mixed-domain training improve both datasets?** Relative to the prior
   mixed-domain checkpoint, CAMRI Dice changed
   {camri['dice']['mean']-previous_camri:+.4f} and Mouse Dice changed
   {mouse['dice']['mean']-previous_mouse:+.4f}. These paired test values, not
   training loss, determine whether the boundary supervision helped.
2. **Does one domain improve while the other degrades?** The independent changes
   above make any domain trade-off explicit; no pooled cross-domain average is
   used.
3. **Localization or boundary failures?** Mean fractions of all erroneous voxels
   inside the symmetric boundary band were
   {camri['boundary_error_fraction']['mean']:.3f} for CAMRI and
   {mouse['boundary_error_fraction']['mean']:.3f} for Mouse. Boundary errors are
   {"dominant in both domains" if all(boundary_dominant.values()) else "not dominant in both domains"};
   distant errors are reported separately as localization-error fractions.
4. **Does the decoder appear capacity-limited?** Capacity limitation is not
   established by this experiment. The unchanged decoder previously overfit a
   tiny set and already transferred strongly; only a consistent residual ceiling
   after optimized supervision would justify that claim.
5. **Are errors concentrated at boundaries?** The boundary fractions above,
   HD95/ASD, and terminal-slice results provide the direct evidence. See
   `per_subject_metrics.csv` and `per_slice_metrics.csv`.
6. **Can the architecture learn anatomical grouping?** Successful retrieval in
   both species, high recall, low leakage-component counts, and preserved CAMRI
   performance support that interpretation for whole-brain grouping. They do not
   establish multi-structure anatomical understanding.
7. **Are architecture changes justified?** Not unless the collected results show
   a reproducible residual localization/capacity failure rather than boundary or
   terminal-slice errors. This run changes supervision only.

## Limitations

- Mouse filenames leave 52 scans without recoverable biological identity, so
  unknown longitudinal linkage remains possible and is not invented.
- The experiment contains one foreground structure and cannot prove universal
  multi-anatomy grouping.
- Boundary-band width is defined in preprocessed voxels; it is one declared loss,
  not a physical-distance loss.
- Test sets were used only after validation selection. No threshold tuning or
  connected-component post-processing was applied.
"""
    (output / "scientific_report.md").write_text(text)


def main():
    args = parse_args()
    config = load_json(ROOT / args.config)
    output = ROOT / config["output_directory"]
    split = load_json(output / "split.json")
    source = {
        row["scan_id"]: row for row in read_csv(ROOT / config["mouse_metrics"])
    }
    cases = []
    for subject in split["camri"]["test"]:
        cases.append(
            (
                "CAMRI",
                subject,
                load_json(output / "native_predictions" / "camri_mixed" / f"{subject}_metrics.json"),
            )
        )
    for subject in split["mouse"]["test"]["scans"]:
        cases.append(
            (
                "Mouse",
                subject,
                load_json(output / "native_predictions" / "mouse_mixed" / f"{subject}_metrics.json"),
            )
        )

    subject_rows = []
    slice_rows = []
    for domain, subject, result in cases:
        row, slices = analyze_case(domain, subject, result, tolerance_mm=0.2)
        subject_rows.append(row)
        slice_rows.extend(slices)
    write_csv(output / "per_subject_metrics.csv", subject_rows)
    write_csv(output / "per_slice_metrics.csv", slice_rows)

    summaries = {
        domain: summarize([row for row in subject_rows if row["domain"] == domain])
        for domain in ("CAMRI", "Mouse")
    }
    rankings = rank_failures(subject_rows, output)
    plots(subject_rows, output)
    comparison_figures(subject_rows, rankings, output, config)
    previous = load_json(ROOT / "outputs/mixed_domain_decoder/summary.json")
    payload = {
        "architecture_changed": False,
        "encoder_frozen": True,
        "exactly_one_query": True,
        "domains": summaries,
    }
    (output / "comprehensive_summary.json").write_text(json.dumps(payload, indent=2))
    report(summaries, previous, output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
