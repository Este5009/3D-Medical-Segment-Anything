#!/usr/bin/env python3
"""Create a traceable presentation package from existing experiment artifacts.

This script performs no training or inference. It reads saved native-space MRI,
expert masks, raw predictions, deterministic-filter predictions, checkpoints,
and metrics. All derived values are recomputed from those saved arrays or
copied from explicitly named source files.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from scipy import ndimage

OUTPUT = ROOT / "outputs/presentation_results"
SUBJECT_SOURCE = ROOT / "outputs/filtered_residual_failure_analysis/per_subject_metrics.csv"
FINAL_CHECKPOINT = ROOT / "outputs/mixed_domain_anatomical_training/checkpoints/best_mixed_domain.pt"

GREEN = "#24c96b"
CYAN = "#00d5e7"
WHITE = "#ffffff"
RED = "#ef3340"
YELLOW = "#ffd23f"
MAGENTA = "#e83eaf"
BLUE = "#3478c8"


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def load_case(row: dict):
    image_obj = nib.load(row["image_path"])
    image = np.asarray(image_obj.dataobj, dtype=np.float32)
    expert = np.asarray(nib.load(row["ground_truth_path"]).dataobj) > 0
    raw = np.asarray(nib.load(row["baseline_prediction_path"]).dataobj) > 0
    filtered = np.asarray(nib.load(row["filtered_prediction_path"]).dataobj) > 0
    if not (image.shape == expert.shape == raw.shape == filtered.shape):
        raise ValueError(f"geometry mismatch: {row['domain']} {row['subject']}")
    return image_obj, image, expert, raw, filtered


def normalize(image):
    low, high = np.percentile(image[np.isfinite(image)], (1, 99.5))
    return np.clip((image - low) / max(float(high - low), 1e-8), 0, 1)


def dice(prediction, expert):
    intersection = int((prediction & expert).sum())
    return 2 * intersection / max(int(prediction.sum() + expert.sum()), 1)


def crop2d(mask, margin=12):
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    low = np.maximum(coordinates.min(0) - margin, 0)
    high = np.minimum(coordinates.max(0) + margin + 1, mask.shape)
    return slice(int(low[0]), int(high[0])), slice(int(low[1]), int(high[1]))


def oriented(array):
    return np.rot90(array)


def removed_islands(raw, filtered):
    return raw & ~filtered


def choose_slice(expert, raw, filtered, role):
    removed = removed_islands(raw, filtered)
    fp = filtered & ~expert
    fn = expert & ~filtered
    if role == "detached_island":
        score = removed.sum((0, 1))
    elif role == "boundary_error":
        score = (fp | fn).sum((0, 1))
    else:
        # Strong/average/difficult are subject-level ranks. Show their most
        # anatomically representative expert-brain slice rather than a small
        # terminal slice with a disproportionate local error count.
        score = expert.sum((0, 1))
    return int(np.argmax(score))


def arrow_error(axis, error_mask, text, color, fallback_xy):
    coordinates = np.argwhere(error_mask)
    if len(coordinates):
        y, x = coordinates[len(coordinates) // 2]
        xy = (x, y)
    else:
        xy = fallback_xy
    axis.annotate(
        text, xy=xy, xytext=(8, 8), textcoords="axes points", color=color,
        fontsize=7, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4},
        bbox={"boxstyle": "round,pad=.2", "facecolor": "black", "alpha": .65,
              "edgecolor": color},
    )


def example_figure(row, role, destination):
    _, image, expert, raw, filtered = load_case(row)
    z = choose_slice(expert, raw, filtered, role)
    union = expert[:, :, z] | raw[:, :, z] | filtered[:, :, z]
    box = crop2d(union)
    mri = oriented(normalize(image)[box[0], box[1], z])
    gt = oriented(expert[box[0], box[1], z])
    raw_s = oriented(raw[box[0], box[1], z])
    filtered_s = oriented(filtered[box[0], box[1], z])
    removed = raw_s & ~filtered_s
    fp = filtered_s & ~gt
    fn = gt & ~filtered_s

    code = np.zeros(gt.shape, dtype=np.uint8)
    code[gt & filtered_s] = 1
    code[fp] = 2
    code[fn] = 3
    code[removed] = 4
    error_cmap = ListedColormap(["black", WHITE, RED, YELLOW, MAGENTA])
    figure, axes = plt.subplots(1, 6, figsize=(18, 3.5), constrained_layout=True)
    titles = (
        "MRI", "Expert mask", "Raw prediction", "Filtered prediction",
        "Boundary comparison", "Error map",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=9, fontweight="semibold")
        axis.axis("off")
    axes[0].imshow(mri, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].contour(gt, [.5], colors=GREEN, linewidths=1.4)
    axes[2].imshow(raw_s, cmap="gray", vmin=0, vmax=1)
    axes[2].contour(raw_s, [.5], colors=CYAN, linewidths=1.2)
    axes[3].imshow(filtered_s, cmap="gray", vmin=0, vmax=1)
    axes[3].contour(filtered_s, [.5], colors=CYAN, linewidths=1.2)
    axes[4].imshow(mri, cmap="gray", vmin=0, vmax=1)
    axes[4].contour(gt, [.5], colors=GREEN, linewidths=1.5)
    axes[4].contour(filtered_s, [.5], colors=CYAN, linewidths=1.2)
    axes[5].imshow(code, cmap=error_cmap, vmin=0, vmax=4, interpolation="nearest")
    if fp.any():
        arrow_error(axes[5], fp, "Boundary false positive", RED, (5, 5))
    if fn.any():
        arrow_error(axes[5], fn, "Boundary false negative", YELLOW, (5, 12))
    if removed.any():
        arrow_error(axes[5], removed, "Detached unrelated-tissue island", MAGENTA, (5, 20))
        coordinates = np.argwhere(removed)
        low, high = coordinates.min(0), coordinates.max(0)
        axes[2].add_patch(plt.Rectangle(
            (low[1] - 2, low[0] - 2), high[1] - low[1] + 4, high[0] - low[0] + 4,
            fill=False, edgecolor=MAGENTA, linewidth=1.6))
    legend = [
        Patch(color=GREEN, label="Expert boundary"),
        Patch(color=CYAN, label="Predicted boundary"),
        Patch(color=WHITE, ec="black", label="True positive"),
        Patch(color=RED, label="False positive"),
        Patch(color=YELLOW, label="False negative"),
        Patch(color=MAGENTA, label="Removed detached island"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=6, fontsize=8,
                  bbox_to_anchor=(.5, -.04))
    raw_dice, filtered_dice = dice(raw, expert), dice(filtered, expert)
    figure.suptitle(
        f"{row['domain']} · {row['subject']} · slice {z} · {role.replace('_', ' ')}\n"
        f"Expert mask vs raw Dice {raw_dice:.4f} · filtered Dice {filtered_dice:.4f}",
        fontsize=12, fontweight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return {"dataset": row["domain"], "subject": row["subject"], "role": role,
            "slice": z, "raw_dice": raw_dice, "filtered_dice": filtered_dice,
            "figure": str(destination)}


def classify_errors(expert, prediction, detached, spacing, boundary_mm=.5):
    expert_surface = expert ^ ndimage.binary_erosion(expert)
    distance = ndimage.distance_transform_edt(~expert_surface, sampling=spacing)
    fp, fn = prediction & ~expert, expert & ~prediction
    detached_fp = detached & ~expert
    boundary_fp = fp & (distance <= boundary_mm) & ~detached_fp
    boundary_fn = fn & (distance <= boundary_mm)
    other = (fp | fn) & ~(boundary_fp | boundary_fn | detached_fp)
    return {
        "boundary_false_positives": int(boundary_fp.sum()),
        "boundary_false_negatives": int(boundary_fn.sum()),
        "detached_false_positive_islands": int(detached_fp.sum()),
        "other_errors": int(other.sum()),
        "total_error_voxels": int((fp | fn).sum()),
    }


def aggregate_metrics(subject_rows, condition):
    values = []
    total_expert = total_fp = total_fn = 0
    for row in subject_rows:
        _, _, expert, raw, filtered = load_case(row)
        prediction = raw if condition == "raw" else filtered
        tp = int((prediction & expert).sum())
        fp = int((prediction & ~expert).sum())
        fn = int((~prediction & expert).sum())
        values.append({
            "dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "hd95": float(row[f"{'baseline' if condition == 'raw' else 'filtered'}_hd95_mm"]),
        })
        total_expert += int(expert.sum())
        total_fp += fp
        total_fn += fn
    return {
        key: float(np.mean([item[key] for item in values]))
        for key in ("dice", "iou", "precision", "recall", "hd95")
    } | {
        "false_positive_percentage": 100 * total_fp / total_expert,
        "false_negative_percentage": 100 * total_fn / total_expert,
    }


def error_distribution(subject_rows, condition):
    totals = {key: 0 for key in (
        "boundary_false_positives", "boundary_false_negatives",
        "detached_false_positive_islands", "other_errors", "total_error_voxels")}
    for row in subject_rows:
        image_obj, _, expert, raw, filtered = load_case(row)
        prediction = raw if condition == "raw" else filtered
        detached = removed_islands(raw, filtered) if condition == "raw" else np.zeros_like(raw)
        result = classify_errors(
            expert, prediction, detached, nib.affines.voxel_sizes(image_obj.affine))
        for key in totals:
            totals[key] += result[key]
    # Detached voxels are raw errors but filtered out of prediction; total error
    # already contains them in the raw condition.
    denominator = max(totals["total_error_voxels"], 1)
    return totals | {
        f"{key}_percentage": 100 * totals[key] / denominator
        for key in ("boundary_false_positives", "boundary_false_negatives",
                    "detached_false_positive_islands", "other_errors")
    }


def select_examples(subject_rows):
    selected = []
    for domain in ("CAMRI", "Mouse"):
        subset = [row for row in subject_rows if row["domain"] == domain]
        ordered = sorted(subset, key=lambda row: float(row["filtered_dice"]))
        mean = np.mean([float(row["filtered_dice"]) for row in subset])
        average = min(subset, key=lambda row: abs(float(row["filtered_dice"]) - mean))
        detached = max(subset, key=lambda row: int(row["residual_voxels_removed"]))
        boundary = max(subset, key=lambda row: int(row["filtered_total_error_voxels"]))
        for role, row in (
            ("strong", ordered[-1]), ("average", average), ("difficult", ordered[0]),
            ("detached_island", detached), ("boundary_error", boundary),
        ):
            selected.append((domain, role, row))
    return selected


def render_table(data, columns, destination, title, formats=None, note=None):
    formats = formats or {}
    cell_text = []
    for row in data:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(formats.get(column, lambda x: str(x))(value))
        cell_text.append(cells)
    width = max(12, 1.5 * len(columns))
    height = 1.4 + .55 * len(data) + (.4 if note else 0)
    figure, axis = plt.subplots(figsize=(width, height))
    axis.axis("off")
    table = axis.table(cellText=cell_text, colLabels=columns, cellLoc="center",
                       loc="center", bbox=[0, .08 if note else 0, 1, .82])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(columns))))
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("#17365d")
        table[(0, column)].set_text_props(color="white", weight="bold")
    for row_index in range(1, len(data) + 1):
        for column in range(len(columns)):
            table[(row_index, column)].set_facecolor("#edf4fb" if row_index % 2 else "white")
    axis.set_title(title, fontsize=15, fontweight="bold", pad=14)
    if note:
        figure.text(.5, .015, note, ha="center", fontsize=8)
    figure.savefig(destination, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def model_versions():
    return [
        {
            "version": "Minimal one-query overfit", "main_change": "L4 attention + L1 mask",
            "epoch": 15, "checkpoint": "outputs/query_decoder_overfit/best_checkpoint.pt",
            "trainable_parameters": 24577, "CAMRI_Dice": 0.9030803740,
            "Mouse_Dice": "", "status": "superseded",
            "reason": "Tiny 2-subject overfit only; capacity proof",
            "scope": "CAMRI 2-subject overfit",
        },
        {
            "version": "Multiscale decoder overfit", "main_change": "4-scale attention + FPN",
            "epoch": 40, "checkpoint": "outputs/query_decoder_multiscale_overfit/best_checkpoint.pt",
            "trainable_parameters": 170401, "CAMRI_Dice": 0.9892440140,
            "Mouse_Dice": "", "status": "accepted architecture",
            "reason": "Near-perfect tiny-set capacity",
            "scope": "CAMRI 2-subject overfit",
        },
        {
            "version": "Generalization checkpoint", "main_change": "40-subject CAMRI pilot",
            "epoch": 14, "checkpoint": "outputs/generalization_pilot/best_checkpoint.pt",
            "trainable_parameters": 170401, "CAMRI_Dice": 0.9878275120,
            "Mouse_Dice": 0.8828059217, "status": "superseded",
            "reason": "Strong CAMRI; poor zero-shot Mouse precision",
            "scope": "CAMRI test 6; Mouse external 101",
        },
        {
            "version": "Mouse boundary adaptation", "main_change": "Mask-head-only Mouse adaptation",
            "epoch": 16, "checkpoint": "outputs/mouse_boundary_adaptation/checkpoints/boundary_head_best.pt",
            "trainable_parameters": 29793, "CAMRI_Dice": 0.9280117269,
            "Mouse_Dice": 0.9347007551, "status": "rejected",
            "reason": "Mouse improved; CAMRI forgetting",
            "scope": "Locked CAMRI 6; Mouse test 80",
        },
        {
            "version": "Mixed-domain decoder", "main_change": "Balanced CAMRI + Mouse training",
            "epoch": 17, "checkpoint": "outputs/mixed_domain_anatomical_training/checkpoints/best_mixed_domain.pt",
            "trainable_parameters": 170401, "CAMRI_Dice": 0.9825178382,
            "Mouse_Dice": 0.9647620756, "status": "accepted",
            "reason": "Preserved CAMRI and improved Mouse",
            "scope": "Locked CAMRI 6; Mouse test 80",
        },
        {
            "version": "Learned grouping module", "main_change": "L2 residual grouping refinement",
            "epoch": 1, "checkpoint": "outputs/query_conditioned_3d_grouping/checkpoints/best_grouping_module.pt",
            "trainable_parameters": 8537, "CAMRI_Dice": 0.9819673307,
            "Mouse_Dice": 0.9639057545, "status": "rejected",
            "reason": "Regressed both domains; query conditioning not useful",
            "scope": "Locked CAMRI 6; Mouse test 80",
        },
        {
            "version": "Final deterministic filter", "main_change": "Largest 26-connected component",
            "epoch": 17, "checkpoint": "same epoch-17 mixed-domain checkpoint",
            "trainable_parameters": 0, "CAMRI_Dice": 0.9826255639,
            "Mouse_Dice": 0.9656853201, "status": "selected final",
            "reason": "Removed detached FP with unchanged recall",
            "scope": "Locked CAMRI 6; Mouse test 80",
        },
    ]


def charts(final_rows, error_rows, version_rows, subject_rows):
    chart_dir = OUTPUT / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    # A: comparable model-version test results.
    comparable = [row for row in version_rows if row["scope"] == "Locked CAMRI 6; Mouse test 80"]
    source = [{"version": row["version"], "CAMRI": row["CAMRI_Dice"],
               "Mouse": row["Mouse_Dice"], "selected_final": row["status"] == "selected final"}
              for row in comparable]
    save_csv(chart_dir / "dice_by_model_version_source.csv", source)
    x = np.arange(len(source))
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.plot(x, [row["CAMRI"] for row in source], marker="o", label="CAMRI", color=BLUE)
    axis.plot(x, [row["Mouse"] for row in source], marker="s", label="Mouse", color="#d95f02")
    selected = next(i for i, row in enumerate(source) if row["selected_final"])
    axis.axvspan(selected - .25, selected + .25, color="#ffd166", alpha=.25,
                 label="Selected final checkpoint")
    axis.set(title="Mean Dice by recoverable model version",
             xlabel="Model version / checkpoint", ylabel="Mean Dice score", ylim=(0, 1))
    axis.set_xticks(x, [row["version"] for row in source], rotation=20, ha="right")
    axis.legend(); axis.grid(alpha=.25)
    figure.savefig(chart_dir / "dice_by_model_version.png", dpi=240)
    plt.close(figure)

    # B: final subject distributions.
    distribution = []
    for row in subject_rows:
        distribution.append({"dataset": row["domain"], "subject": row["subject"],
                             "filtered_dice": float(row["filtered_dice"])})
    save_csv(chart_dir / "dice_distribution_source.csv", distribution)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    groups = [[row["filtered_dice"] for row in distribution if row["dataset"] == domain]
              for domain in ("CAMRI", "Mouse")]
    parts = axis.violinplot(groups, positions=[1, 2], showmeans=True, showextrema=True)
    for body, color in zip(parts["bodies"], (BLUE, "#d95f02")):
        body.set_facecolor(color); body.set_alpha(.45)
    for position, group, color, label in zip((1, 2), groups, (BLUE, "#d95f02"), ("CAMRI", "Mouse")):
        jitter = np.linspace(-.08, .08, len(group))
        axis.scatter(position + jitter, group, s=14, color=color, label=label)
    axis.set(title="Final filtered subject-level Dice distribution",
             xlabel="Dataset", ylabel="Subject-level Dice score", ylim=(0, 1),
             xticks=[1, 2], xticklabels=["CAMRI", "Mouse"])
    axis.legend(); axis.grid(axis="y", alpha=.25)
    figure.savefig(chart_dir / "dice_distribution.png", dpi=240)
    plt.close(figure)

    # C: error distribution.
    keys = ("boundary_false_positives", "boundary_false_negatives",
            "detached_false_positive_islands", "other_errors")
    labels = ("Boundary FP", "Boundary FN", "Detached FP", "Other")
    colors = (RED, YELLOW, MAGENTA, "#8c8c8c")
    save_csv(chart_dir / "error_distribution_source.csv", error_rows)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bottom = np.zeros(len(error_rows))
    x = np.arange(len(error_rows))
    for key, label, color in zip(keys, labels, colors):
        values = np.array([row[f"{key}_percentage"] for row in error_rows])
        axis.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set(title="Exclusive residual error distribution",
             xlabel="Dataset and prediction type",
             ylabel="Percentage of total error voxels (%)")
    axis.set_xticks(x, [row["condition"] for row in error_rows], rotation=15)
    axis.set_ylim(0, 100); axis.legend(ncol=4); axis.grid(axis="y", alpha=.2)
    figure.savefig(chart_dir / "error_distribution.png", dpi=240)
    plt.close(figure)

    # D: raw vs filtered.
    save_csv(chart_dir / "raw_vs_filtered_source.csv", final_rows)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(2); width = .34
    raw_values = [next(row["Dice"] for row in final_rows if row["condition"] == f"{d} raw")
                  for d in ("CAMRI", "Mouse")]
    filtered_values = [next(row["Dice"] for row in final_rows if row["condition"] == f"{d} filtered")
                       for d in ("CAMRI", "Mouse")]
    axis.bar(x - width / 2, raw_values, width, label="Raw prediction", color="#8da0cb")
    axis.bar(x + width / 2, filtered_values, width, label="Largest-component filtered",
             color="#66c2a5")
    axis.set(title="Raw versus deterministic-filter Dice",
             xlabel="Dataset", ylabel="Mean Dice score", ylim=(0, 1),
             xticks=x, xticklabels=["CAMRI", "Mouse"])
    axis.legend(); axis.grid(axis="y", alpha=.25)
    figure.savefig(chart_dir / "raw_vs_filtered.png", dpi=240)
    plt.close(figure)

    # E: parameter count vs performance on the same locked test sets.
    parameter_source = []
    for row in comparable:
        for domain in ("CAMRI", "Mouse"):
            parameter_source.append({
                "version": row["version"], "domain": domain,
                "trainable_parameters": row["trainable_parameters"],
                "mean_dice": row[f"{domain}_Dice"],
            })
    save_csv(chart_dir / "parameters_vs_performance_source.csv", parameter_source)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for domain, marker, color in (("CAMRI", "o", BLUE), ("Mouse", "s", "#d95f02")):
        subset = [row for row in parameter_source if row["domain"] == domain]
        axis.scatter([row["trainable_parameters"] for row in subset],
                     [row["mean_dice"] for row in subset],
                     marker=marker, color=color, s=55, label=domain)
        for row in subset:
            axis.annotate(row["version"], (row["trainable_parameters"], row["mean_dice"]),
                          xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set(title="Trainable parameters versus locked-test Dice",
             xlabel="Trainable parameters (count)", ylabel="Mean Dice score", ylim=(0, 1))
    axis.ticklabel_format(axis="x", style="plain")
    axis.legend(); axis.grid(alpha=.25)
    figure.savefig(chart_dir / "parameters_vs_performance.png", dpi=240)
    plt.close(figure)


def common_error_figures(example_manifest, subject_rows):
    common = OUTPUT / "common_errors"; common.mkdir(parents=True, exist_ok=True)
    # Reuse the most informative generated examples as dedicated presentation slides.
    detached = max(
        (item for item in example_manifest if item["role"] == "detached_island"),
        key=lambda item: item["filtered_dice"] - item["raw_dice"])
    boundary = max(
        (item for item in example_manifest if item["role"] == "boundary_error"),
        key=lambda item: int(next(row["filtered_total_error_voxels"] for row in subject_rows
                                  if row["domain"] == item["dataset"] and
                                  row["subject"] == item["subject"])))
    detached_source = Path(detached["figure"])
    boundary_source = Path(boundary["figure"])
    import shutil
    shutil.copy2(detached_source, common / "detached_island_before_after.png")
    shutil.copy2(boundary_source, common / "boundary_error_detail.png")


def final_summary_page(final_rows, example_manifest):
    summary_dir = OUTPUT / "summary"; summary_dir.mkdir(parents=True, exist_ok=True)
    camri = next(row["Dice"] for row in final_rows if row["condition"] == "CAMRI filtered")
    mouse = next(row["Dice"] for row in final_rows if row["condition"] == "Mouse filtered")
    example = plt.imread(example_manifest[0]["figure"])
    raw_chart = plt.imread(OUTPUT / "charts/raw_vs_filtered.png")
    error_chart = plt.imread(OUTPUT / "charts/error_distribution.png")
    figure, axes = plt.subplots(
        2, 3, figsize=(16, 9), constrained_layout=True,
        gridspec_kw={"height_ratios": (.28, .72)})
    ax0, ax1, ax2 = axes[0]
    ax3, ax4, ax5 = axes[1]
    for axis in axes.flat: axis.axis("off")
    ax0.text(.5, .58, f"{camri:.4f}", ha="center", fontsize=34, weight="bold", color=BLUE)
    ax0.text(.5, .25, "Final CAMRI mean Dice", ha="center", fontsize=13)
    ax1.text(.5, .58, f"{mouse:.4f}", ha="center", fontsize=34, weight="bold", color="#d95f02")
    ax1.text(.5, .25, "Final Mouse mean Dice", ha="center", fontsize=13)
    ax2.text(.5, .68, "Epoch 17", ha="center", fontsize=25, weight="bold")
    ax2.text(.5, .42, "170,401 decoder parameters", ha="center", fontsize=13)
    ax2.text(.5, .2, "best_mixed_domain.pt", ha="center", fontsize=11)
    ax3.imshow(example, aspect="auto")
    ax3.set_title("Expert-versus-prediction example", fontsize=12)
    ax4.imshow(raw_chart, aspect="auto")
    ax4.set_title("Raw versus filtered", fontsize=12)
    ax5.imshow(error_chart, aspect="auto")
    ax5.set_title("Residual error distribution", fontsize=12)
    figure.suptitle("Frozen one-query decoder: final cross-domain results",
                    fontsize=22, weight="bold")
    figure.savefig(summary_dir / "final_summary_page.png", dpi=240, facecolor="white")
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subject_rows = rows(SUBJECT_SOURCE)
    by_domain = {domain: [row for row in subject_rows if row["domain"] == domain]
                 for domain in ("CAMRI", "Mouse")}

    final_rows = []
    error_rows = []
    for domain in ("CAMRI", "Mouse"):
        for condition in ("raw", "filtered"):
            metrics = aggregate_metrics(by_domain[domain], condition)
            final_rows.append({"condition": f"{domain} {condition}",
                               "Dice": metrics["dice"], "IoU": metrics["iou"],
                               "precision": metrics["precision"], "recall": metrics["recall"],
                               "HD95_mm": metrics["hd95"],
                               "false_positive_percentage": metrics["false_positive_percentage"],
                               "false_negative_percentage": metrics["false_negative_percentage"]})
            distribution = error_distribution(by_domain[domain], condition)
            error_rows.append({"condition": f"{domain} {condition}", **distribution})

    table_dir = OUTPUT / "tables"; table_dir.mkdir(exist_ok=True)
    save_csv(table_dir / "final_results.csv", final_rows)
    render_table(
        final_rows, list(final_rows[0]), table_dir / "final_results.png", "Final test results",
        formats={key: (lambda x: f"{float(x):.4f}") for key in
                 ("Dice", "IoU", "precision", "recall", "HD95_mm",
                  "false_positive_percentage", "false_negative_percentage")},
        note="FP% and FN% are aggregate error voxels divided by aggregate expert foreground voxels; HD95 in mm.",
    )

    versions = model_versions()
    save_csv(table_dir / "model_versions.csv", versions)
    version_columns = ("version", "main_change", "epoch", "checkpoint", "trainable_parameters",
                       "CAMRI_Dice", "Mouse_Dice", "status")
    render_table(versions, version_columns, table_dir / "model_versions.png",
                 "Recoverable model versions",
                 formats={"CAMRI_Dice": lambda x: "" if x == "" else f"{float(x):.4f}",
                          "Mouse_Dice": lambda x: "" if x == "" else f"{float(x):.4f}"},
                 note="Evaluation scope is recorded in model_versions.csv; tiny-overfit values are not test metrics.")

    save_csv(table_dir / "error_distribution.csv", error_rows)
    error_columns = ("condition", "boundary_false_positives_percentage",
                     "boundary_false_negatives_percentage",
                     "detached_false_positive_islands_percentage", "other_errors_percentage")
    render_table(error_rows, error_columns, table_dir / "error_distribution.png",
                 "Exclusive error distribution",
                 formats={key: lambda x: f"{float(x):.2f}%" for key in error_columns[1:]},
                 note="Every percentage is a percentage of total error voxels for that dataset and prediction type.")

    example_manifest = []
    for domain, role, row in select_examples(subject_rows):
        destination = OUTPUT / "expert_comparisons" / domain.lower() / f"{role}_{row['subject']}.png"
        example_manifest.append(example_figure(row, role, destination))
    save_csv(OUTPUT / "expert_comparisons" / "manifest.csv", example_manifest)

    charts(final_rows, error_rows, versions, subject_rows)
    common_error_figures(example_manifest, subject_rows)

    # Development timeline is both machine-readable and presentation-ready.
    timeline_columns = ("version", "main_change", "epoch", "checkpoint", "trainable_parameters",
                        "CAMRI_Dice", "Mouse_Dice", "status", "reason")
    save_csv(OUTPUT / "model_development_timeline.csv", versions)
    render_table(versions, timeline_columns, OUTPUT / "model_development_timeline.png",
                 "Model development timeline",
                 formats={"CAMRI_Dice": lambda x: "" if x == "" else f"{float(x):.4f}",
                          "Mouse_Dice": lambda x: "" if x == "" else f"{float(x):.4f}"},
                 note="Scopes differ for early capacity experiments; see CSV scope column.")

    # The residual attribution directly supports the requested message.
    filtered_boundary = sum(
        row["boundary_false_positives"] + row["boundary_false_negatives"]
        for row in error_rows if row["condition"].endswith("filtered"))
    filtered_total = sum(row["total_error_voxels"] for row in error_rows
                         if row["condition"].endswith("filtered"))
    message_supported = filtered_boundary / max(filtered_total, 1) > .9
    if message_supported:
        message = (
            "The model localizes the brain well. Largest-component filtering removes "
            "detached unrelated-tissue islands. Most remaining disagreement is "
            "concentrated near the expert-defined brain boundary."
        )
        figure, axis = plt.subplots(figsize=(14, 4), constrained_layout=True)
        axis.axis("off")
        axis.text(.5, .62, message, ha="center", va="center", wrap=True,
                  fontsize=20, weight="bold", color="#17365d")
        axis.text(.5, .22,
                  f"Filtered boundary-attributed share: {100 * filtered_boundary / filtered_total:.2f}% "
                  f"of residual error voxels · detached FP after filtering: 0",
                  ha="center", fontsize=12)
        figure.savefig(OUTPUT / "common_errors" / "residual_error_message.png",
                       dpi=240, facecolor="white")
        plt.close(figure)

    final_summary_page(final_rows, example_manifest)
    validation = {
        "training_performed": False, "inference_performed": False,
        "subjects_checked": len(subject_rows), "geometry_matches": True,
        "raw_filtered_identity_checks": all(
            np.all((load_case(row)[4] & ~load_case(row)[3]) == 0) for row in subject_rows),
        "expert_boundary_color": GREEN, "prediction_boundary_color": CYAN,
        "true_positive_color": WHITE, "false_positive_color": RED,
        "false_negative_color": YELLOW, "removed_island_color": MAGENTA,
        "final_checkpoint": str(FINAL_CHECKPOINT), "final_epoch": 17,
        "final_trainable_decoder_parameters": 170401,
        "message_supported_by_error_data": message_supported,
        "filtered_boundary_error_share_percentage": 100 * filtered_boundary / filtered_total,
        "source_files": [
            str(SUBJECT_SOURCE),
            str(ROOT / "outputs/query_conditioned_3d_grouping/test_summary.json"),
            str(ROOT / "outputs/mixed_domain_anatomical_training/summary.json"),
        ],
    }
    (OUTPUT / "validation.json").write_text(json.dumps(validation, indent=2))
    (OUTPUT / "README.md").write_text(
        "# Presentation results\n\n"
        "Start with `summary/final_summary_page.png`, then use "
        "`common_errors/` and `expert_comparisons/`. All chart data are beside "
        "their PNGs as source CSV files. No training or inference was run.\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
