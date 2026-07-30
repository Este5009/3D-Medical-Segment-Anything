#!/usr/bin/env python3
"""Dice-only evaluation using completed native predictions.

This script intentionally performs no training and no inference. It verifies
that every locked test subject has a valid native-space prediction, recomputes
Dice directly from prediction and expert NIfTI files, compares against the
previous mixed-domain checkpoint, and creates exactly six compact figures.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dice(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = int((prediction & target).sum())
    denominator = int(prediction.sum()) + int(target.sum())
    return 2.0 * intersection / max(denominator, 1)


def validate_and_score(domain, subject, prediction_path, image_path, target_path):
    prediction_image = nib.load(str(prediction_path))
    target_image = nib.load(str(target_path))
    image = nib.load(str(image_path))
    if prediction_image.shape != target_image.shape or image.shape != target_image.shape:
        raise ValueError(f"{domain} {subject}: native shape mismatch")
    # The established native pipeline accepts sub-voxel header round-off from
    # SimpleITK/NIfTI conversion. The largest observed difference is 6.9e-5 in
    # one Mouse direction term; shapes and voxel correspondence are unchanged.
    if not np.allclose(prediction_image.affine, target_image.affine, atol=1e-4):
        raise ValueError(f"{domain} {subject}: prediction/target affine mismatch")
    prediction = np.asarray(prediction_image.dataobj) > 0
    target = np.asarray(target_image.dataobj) > 0
    if not target.any():
        raise ValueError(f"{domain} {subject}: empty expert mask")
    return {
        "domain": domain,
        "subject": subject,
        "dice": dice(prediction, target),
        "image_path": str(image_path),
        "ground_truth_path": str(target_path),
        "prediction_path": str(prediction_path),
    }


def summarize(rows):
    values = np.asarray([row["dice"] for row in rows], dtype=float)
    return {
        "count": len(rows),
        "mean_dice": float(values.mean()),
        "standard_deviation_dice": float(values.std()),
        "median_dice": float(np.median(values)),
        "minimum_dice": float(values.min()),
        "maximum_dice": float(values.max()),
    }


def select_cases(rows):
    ordered = sorted(rows, key=lambda row: row["dice"])
    values = np.asarray([row["dice"] for row in ordered])
    worst = ordered[0]
    medium = min(ordered, key=lambda row: abs(row["dice"] - float(np.median(values))))
    high_target = float(np.quantile(values, 0.90))
    high_candidates = [row for row in ordered if row["subject"] not in {worst["subject"], medium["subject"]}]
    good = min(high_candidates, key=lambda row: abs(row["dice"] - high_target))
    return {"good": good, "medium": medium, "worst": worst}


def normalize_slice(array):
    low, high = np.percentile(array, (1, 99))
    return np.clip((array - low) / max(high - low, 1e-8), 0, 1)


def choose_slices(prediction, target):
    occupied = np.where(target.any(axis=(0, 1)))[0]
    central = int(occupied[len(occupied) // 2])
    disagreement = np.logical_xor(prediction, target).sum(axis=(0, 1))
    worst = int(np.argmax(disagreement))
    terminals = [int(occupied[0]), int(occupied[-1])]
    terminal = max(terminals, key=lambda index: int(disagreement[index]))
    selected = []
    for index in (central, worst):
        if index not in selected:
            selected.append(index)
    if disagreement[terminal] > 0 and terminal not in selected:
        selected.append(terminal)
    if len(selected) < 3:
        for index in (int(occupied[len(occupied) // 4]), int(occupied[3 * len(occupied) // 4])):
            if index not in selected:
                selected.append(index)
            if len(selected) == 3:
                break
    return selected[:3]


def save_compact_figure(row, category, destination):
    image = np.asarray(nib.load(row["image_path"]).dataobj, dtype=np.float32)
    target = np.asarray(nib.load(row["ground_truth_path"]).dataobj) > 0
    prediction = np.asarray(nib.load(row["prediction_path"]).dataobj) > 0
    slices = choose_slices(prediction, target)
    figure, axes = plt.subplots(
        len(slices),
        3,
        figsize=(8.4, 2.7 * len(slices)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, index in enumerate(slices):
        mri = normalize_slice(image[:, :, index])
        truth = target[:, :, index]
        pred = prediction[:, :, index]
        overlay = np.zeros((*truth.shape, 3), dtype=float)
        overlay[pred & truth] = (0.1, 0.8, 0.2)  # true positive: green
        overlay[pred & ~truth] = (1.0, 0.15, 0.1)  # false positive: red
        overlay[~pred & truth] = (0.1, 0.35, 1.0)  # false negative: blue
        for axis in axes[row_index]:
            axis.axis("off")
        axes[row_index, 0].imshow(mri.T, cmap="gray", origin="lower")
        axes[row_index, 0].contour(truth.T, colors="cyan", linewidths=1)
        axes[row_index, 1].imshow(mri.T, cmap="gray", origin="lower")
        axes[row_index, 1].contour(pred.T, colors="yellow", linewidths=1)
        axes[row_index, 2].imshow(overlay.transpose(1, 0, 2), origin="lower")
        axes[row_index, 0].set_ylabel(f"slice {index}", rotation=0, labelpad=24)
        if row_index == 0:
            axes[row_index, 0].set_title("MRI + expert")
            axes[row_index, 1].set_title("MRI + prediction")
            axes[row_index, 2].set_title("TP green | FP red | FN blue")
    figure.suptitle(
        f"{row['domain']} {category}: {row['subject']} | Dice {row['dice']:.4f}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def status(change):
    if change > 1e-6:
        return "improvement"
    if change < -1e-6:
        return "regression"
    return "unchanged"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mixed_domain_anatomical_training.yaml")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    experiment = ROOT / config["output_directory"]
    output = experiment / "fast_evaluation"
    split = load_json(experiment / "split.json")

    camri_source = {row["subject"]: row for row in read_csv(ROOT / config["camri_metrics"])}
    mouse_source = {row["scan_id"]: row for row in read_csv(ROOT / config["mouse_metrics"])}
    failures = []
    domain_rows = {"CAMRI": [], "Mouse": []}

    for subject in split["camri"]["test"]:
        source = camri_source[subject]
        prediction = experiment / "native_predictions" / "camri_mixed" / f"{subject}_prediction.nii.gz"
        try:
            domain_rows["CAMRI"].append(
                validate_and_score(
                    "CAMRI", subject, prediction, Path(source["image_path"]), Path(source["mask_path"])
                )
            )
        except Exception as error:
            failures.append({"domain": "CAMRI", "subject": subject, "error": str(error)})

    for subject in split["mouse"]["test"]["scans"]:
        source = mouse_source[subject]
        prediction = experiment / "native_predictions" / "mouse_mixed" / f"{subject}_prediction.nii.gz"
        try:
            domain_rows["Mouse"].append(
                validate_and_score(
                    "Mouse",
                    subject,
                    prediction,
                    Path(source["image_path"]),
                    Path(source["ground_truth_path"]),
                )
            )
        except Exception as error:
            failures.append({"domain": "Mouse", "subject": subject, "error": str(error)})

    expected = {"CAMRI": len(split["camri"]["test"]), "Mouse": len(split["mouse"]["test"]["scans"])}
    for domain in ("CAMRI", "Mouse"):
        if len(domain_rows[domain]) != expected[domain]:
            raise RuntimeError(
                f"{domain}: evaluated {len(domain_rows[domain])}/{expected[domain]}; failures={failures}"
            )

    write_csv(output / "camri_subject_dice.csv", domain_rows["CAMRI"])
    write_csv(output / "mouse_subject_dice.csv", domain_rows["Mouse"])

    previous_camri = {
        row["subject"]: float(row["mixed_dice"])
        for row in read_csv(ROOT / "outputs/mixed_domain_decoder/camri_test_comparison.csv")
    }
    previous_mouse = {
        row["scan_id"]: float(row["mixed_dice"])
        for row in read_csv(ROOT / "outputs/mixed_domain_decoder/mouse_test_comparison.csv")
    }
    previous = {
        "CAMRI": float(np.mean([previous_camri[row["subject"]] for row in domain_rows["CAMRI"]])),
        "Mouse": float(np.mean([previous_mouse[row["subject"]] for row in domain_rows["Mouse"]])),
    }

    summary = {
        "checkpoint": str(experiment / "checkpoints" / "best_mixed_domain.pt"),
        "checkpoint_epoch": load_json(experiment / "training_summary.json")["best_epoch"],
        "threshold": 0.5,
        "post_processing": "none",
        "all_available_untouched_test_subjects_evaluated": True,
        "failed_subjects": failures,
        "datasets": {},
    }
    figure_paths = []
    for domain in ("CAMRI", "Mouse"):
        values = summarize(domain_rows[domain])
        change = values["mean_dice"] - previous[domain]
        summary["datasets"][domain] = {
            **values,
            "previous_mixed_domain_mean_dice": previous[domain],
            "absolute_mean_dice_change": change,
            "comparison": status(change),
        }
        for category, row in select_cases(domain_rows[domain]).items():
            path = output / "qualitative_figures" / domain.lower() / f"{category}_{row['subject']}.png"
            save_compact_figure(row, category, path)
            figure_paths.append(str(path))
    summary["qualitative_figures"] = figure_paths
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    camri = summary["datasets"]["CAMRI"]
    mouse = summary["datasets"]["Mouse"]
    text = f"""# Dice-only unseen-test evaluation

Every available untouched test subject was evaluated successfully. Threshold was
fixed at 0.5; no cleanup or post-processing was applied.

| Dataset | N | Mean ± SD Dice | Median | Min | Max | Previous | Change | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CAMRI | {camri['count']} | {camri['mean_dice']:.6f} ± {camri['standard_deviation_dice']:.6f} | {camri['median_dice']:.6f} | {camri['minimum_dice']:.6f} | {camri['maximum_dice']:.6f} | {camri['previous_mixed_domain_mean_dice']:.6f} | {camri['absolute_mean_dice_change']:+.6f} | {camri['comparison']} |
| Mouse | {mouse['count']} | {mouse['mean_dice']:.6f} ± {mouse['standard_deviation_dice']:.6f} | {mouse['median_dice']:.6f} | {mouse['minimum_dice']:.6f} | {mouse['maximum_dice']:.6f} | {mouse['previous_mixed_domain_mean_dice']:.6f} | {mouse['absolute_mean_dice_change']:+.6f} | {mouse['comparison']} |

Failed subjects: none.
"""
    (output / "summary.md").write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
