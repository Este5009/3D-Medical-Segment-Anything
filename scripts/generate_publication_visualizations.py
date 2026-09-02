#!/usr/bin/env python3
"""Generate publication figures from existing CAMRI and Mouse predictions.

No inference or training occurs here. Subjects are selected deterministically
from the existing filtered test metrics: minimum, median, and maximum Dice in
each domain. The saved largest-26-component predictions are reused verbatim.
"""
from __future__ import annotations

import argparse
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
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure


SOURCE_METRICS = ROOT / "outputs/filtered_residual_failure_analysis/per_subject_metrics.csv"
CHECKPOINT = ROOT / "outputs/mixed_domain_anatomical_training/checkpoints/best_mixed_domain.pt"
OUTPUT = ROOT / "outputs/publication_visualizations"
ACCENT = "#00d5e7"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image)]
    low, high = np.percentile(values, (1, 99.5))
    return np.clip((image - low) / max(float(high - low), 1e-8), 0, 1)


def display_slice(array: np.ndarray, index: int) -> np.ndarray:
    """Display native axis-2 slices with a consistent radiological layout."""
    return np.rot90(array[:, :, index])


def crop_box(mask: np.ndarray, margin: int = 8) -> tuple[slice, slice]:
    projection = mask.any(axis=2)
    coordinates = np.argwhere(projection)
    if not len(coordinates):
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    low = np.maximum(coordinates.min(0) - margin, 0)
    high = np.minimum(coordinates.max(0) + margin + 1, mask.shape[:2])
    return slice(int(low[0]), int(high[0])), slice(int(low[1]), int(high[1]))


def save_contact_pages(
    image: np.ndarray, mask: np.ndarray, destination: Path, subject: str, kind: str,
    panels_per_page: int = 30,
) -> int:
    """Save all native slices across one or more dense, consistently cropped pages."""
    destination.mkdir(parents=True, exist_ok=True)
    image_n = normalized(image)
    box = crop_box(mask)
    page_count = 0
    for start in range(0, image.shape[2], panels_per_page):
        indices = list(range(start, min(start + panels_per_page, image.shape[2])))
        figure, axes = plt.subplots(5, 6, figsize=(12, 10), constrained_layout=True)
        for axis, z in zip(axes.flat, indices):
            mri = display_slice(image_n[box[0], box[1], :], z)
            prediction = display_slice(mask[box[0], box[1], :], z)
            if kind in ("original", "overlay", "brain_only"):
                shown = mri if kind != "brain_only" else mri * prediction
                axis.imshow(shown, cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(prediction, cmap="gray", vmin=0, vmax=1)
            if kind == "overlay" and prediction.any():
                axis.imshow(np.ma.masked_where(~prediction, prediction), cmap="winter",
                            vmin=0, vmax=1, alpha=.22)
                axis.contour(prediction, levels=[.5], colors=ACCENT, linewidths=.75)
            axis.set_title(f"slice {z}", fontsize=7)
            axis.axis("off")
        for axis in axes.flat[len(indices):]:
            axis.axis("off")
        readable = kind.replace("_", " ").title()
        figure.suptitle(f"{subject} · {readable} · slices {indices[0]}–{indices[-1]}",
                       fontsize=13, fontweight="semibold")
        page_count += 1
        figure.savefig(destination / f"page_{page_count:02d}.png", dpi=220,
                       facecolor="white")
        plt.close(figure)
    return page_count


def brain_center(mask: np.ndarray) -> np.ndarray:
    return np.round(np.argwhere(mask).mean(0)).astype(int) if mask.any() else np.array(mask.shape) // 2


def plane(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(array, index, axis=axis))


def save_orthogonal(image, mask, destination, subject):
    center = brain_center(mask)
    image_n = normalized(image)
    views = ((2, center[2], "Axial"), (1, center[1], "Coronal"), (0, center[0], "Sagittal"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.3), constrained_layout=True)
    for axis, (dimension, index, title) in zip(axes, views):
        mri, prediction = plane(image_n, dimension, int(index)), plane(mask, dimension, int(index))
        axis.imshow(mri, cmap="gray", vmin=0, vmax=1)
        if prediction.any():
            axis.contour(prediction, [.5], colors=ACCENT, linewidths=1.4)
        axis.set_title(f"{title} · slice {index}", fontsize=11)
        axis.axis("off")
    figure.suptitle(f"{subject} · orthogonal prediction contours", fontsize=14,
                   fontweight="semibold")
    figure.savefig(destination, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def surface_mesh(mask: np.ndarray):
    # Smooth only the display isosurface, never the saved prediction. This
    # suppresses voxel stair-steps while preserving the binary output itself.
    field = ndimage.gaussian_filter(mask.astype(np.float32), sigma=1.0)
    return measure.marching_cubes(field, .5, step_size=1)


SURFACE_VIEWS = {
    "left": (0, False), "right": (0, True), "anterior": (1, False),
    "posterior": (1, True), "inferior": (2, False), "superior": (2, True),
}


def save_surface_views(mask: np.ndarray, destination: Path, subject: str) -> int:
    """Render six opaque depth-shaded cardinal surfaces.

    Cardinal depth rendering avoids the polygon sorting artifacts that
    matplotlib's 3D engine produces for large medical meshes.
    """
    destination.mkdir(parents=True, exist_ok=True)
    smooth = ndimage.gaussian_filter(mask.astype(np.float32), sigma=.8) > .5
    for name, (view_axis, reverse) in SURFACE_VIEWS.items():
        stack = np.moveaxis(np.flip(smooth, view_axis) if reverse else smooth, view_axis, 0)
        silhouette = stack.any(0)
        depth = np.argmax(stack, axis=0).astype(np.float32)
        depth[~silhouette] = np.nan
        filled = np.nan_to_num(depth, nan=float(np.nanmax(depth) if silhouette.any() else 0))
        # Low-frequency depth normals give smooth anatomical lighting rather
        # than emphasizing individual voxel terraces.
        filled = ndimage.gaussian_filter(filled, sigma=3.0)
        gradient_y, gradient_x = np.gradient(filled)
        normal_z = 1 / np.sqrt(1 + gradient_x ** 2 + gradient_y ** 2)
        normal_x, normal_y = -gradient_x * normal_z, -gradient_y * normal_z
        light = np.array([-.45, -.35, .82])
        intensity = np.clip(normal_x * light[0] + normal_y * light[1] + normal_z * light[2], 0, 1)
        intensity = .28 + .72 * intensity
        rgb = np.ones((*silhouette.shape, 3), dtype=np.float32)
        base = np.array([.08, .55, .70])
        rgb[silhouette] = base * intensity[silhouette, None] + .12 * (1 - intensity[silhouette, None])
        rgb = np.rot90(rgb)
        figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
        axis.imshow(rgb, interpolation="bicubic")
        short_subject = subject if len(subject) < 38 else subject[:35] + "…"
        axis.set_title(f"{short_subject}\n{name}", fontsize=10, fontweight="semibold")
        axis.axis("off")
        figure.savefig(destination / f"{name}.png", dpi=300, bbox_inches="tight",
                       pad_inches=.05, facecolor="white")
        plt.close(figure)
    return len(SURFACE_VIEWS)


def alpha_composite(volume: np.ndarray, mask: np.ndarray, axis: int, reverse=False) -> np.ndarray:
    """Front-to-back intensity rendering with masked background transparency."""
    values = normalized(volume)
    if reverse:
        values, mask = np.flip(values, axis=axis), np.flip(mask, axis=axis)
    values, mask = np.moveaxis(values, axis, 0), np.moveaxis(mask, axis, 0)
    alpha = np.where(mask, .04 + .15 * values ** 1.35, 0)
    transmission = np.cumprod(1 - alpha + 1e-8, axis=0)
    weights = alpha * np.concatenate((np.ones_like(transmission[:1]), transmission[:-1]), axis=0)
    rendered = (weights * values).sum(0)
    rendered /= max(float(rendered.max()), 1e-8)
    return np.rot90(rendered)


VOLUME_VIEWS = {
    "left": (0, False), "right": (0, True), "anterior": (1, False),
    "posterior": (1, True), "inferior": (2, False), "superior": (2, True),
}


def save_volume_views(image: np.ndarray, mask: np.ndarray, destination: Path, subject: str) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    renderings = {
        name: alpha_composite(image, mask, axis, reverse)
        for name, (axis, reverse) in VOLUME_VIEWS.items()
    }
    rotated_image = ndimage.rotate(image, 32, axes=(0, 1), reshape=False, order=1)
    rotated_mask = ndimage.rotate(mask.astype(np.uint8), 32, axes=(0, 1),
                                  reshape=False, order=0) > 0
    renderings["oblique"] = alpha_composite(rotated_image, rotated_mask, 0, False)
    for name, rendering in renderings.items():
        foreground = rendering > .01
        if foreground.any():
            coordinates = np.argwhere(foreground)
            low = np.maximum(coordinates.min(0) - 5, 0)
            high = np.minimum(coordinates.max(0) + 6, rendering.shape)
            rendering = rendering[low[0]:high[0], low[1]:high[1]]
        figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
        axis.imshow(rendering, cmap="bone", vmin=0, vmax=1)
        short_subject = subject if len(subject) < 38 else subject[:35] + "…"
        axis.set_title(f"{short_subject}\nintensity volume · {name}", fontsize=10,
                       fontweight="semibold", color="white")
        axis.axis("off")
        figure.savefig(destination / f"{name}.png", dpi=300, facecolor="black",
                       bbox_inches="tight")
        plt.close(figure)
    return len(renderings)


def save_comparison(image, target, prediction, destination, subject, dice):
    z = int(np.argmax(target.sum((0, 1))))
    box = crop_box(target | prediction, margin=10)
    image_n = display_slice(normalized(image)[box[0], box[1], :], z)
    target_s = display_slice(target[box[0], box[1], :], z)
    prediction_s = display_slice(prediction[box[0], box[1], :], z)
    difference = np.zeros((*target_s.shape, 3), dtype=np.float32)
    difference[target_s & prediction_s] = (.2, .75, .35)
    difference[prediction_s & ~target_s] = (1, .25, .2)
    difference[target_s & ~prediction_s] = (.2, .55, 1)
    figure, axes = plt.subplots(1, 5, figsize=(17, 3.7), constrained_layout=True)
    axes[0].imshow(image_n, cmap="gray"); axes[0].set_title("MRI")
    axes[1].imshow(target_s, cmap="gray"); axes[1].set_title("Ground truth")
    axes[2].imshow(prediction_s, cmap="gray"); axes[2].set_title("Prediction")
    axes[3].imshow(image_n, cmap="gray")
    axes[3].contour(target_s, [.5], colors="#ffd166", linewidths=1.2)
    axes[3].contour(prediction_s, [.5], colors=ACCENT, linewidths=1.2)
    axes[3].set_title("GT (yellow) · Pred (cyan)")
    axes[4].imshow(image_n, cmap="gray", alpha=.55)
    axes[4].imshow(difference, alpha=.8)
    axes[4].set_title("TP green · FP red · FN blue")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{subject} · slice {z} · volumetric Dice {dice:.4f}",
                   fontsize=14, fontweight="semibold")
    figure.savefig(destination, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def ffmpeg_path() -> Path:
    candidates = [
        Path(sys.executable).parent / "ffmpeg",
        Path("/opt/homebrew/Caskroom/miniforge/base/envs/rs2/bin/ffmpeg"),
        Path("/opt/homebrew/Caskroom/miniforge/base/pkgs/ffmpeg-4.3.2-h38cfed3_3/bin/ffmpeg"),
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise RuntimeError("ffmpeg executable not found")
    return path


def save_movie(image, mask, destination, subject):
    matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg_path())
    image_n = normalized(image)
    box = crop_box(mask, margin=10)
    figure, axes = plt.subplots(1, 3, figsize=(9, 3.3), constrained_layout=True)
    writer = FFMpegWriter(fps=5, metadata={"title": subject})
    with writer.saving(figure, str(destination), dpi=140):
        for z in range(image.shape[2]):
            mri = display_slice(image_n[box[0], box[1], :], z)
            prediction = display_slice(mask[box[0], box[1], :], z)
            for axis in axes:
                axis.clear()
                axis.axis("off")
            axes[0].imshow(mri, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title(f"MRI · slice {z}")
            axes[1].imshow(mri, cmap="gray", vmin=0, vmax=1)
            if prediction.any():
                axes[1].contour(prediction, [.5], colors=ACCENT, linewidths=1)
            axes[1].set_title("Prediction contour")
            axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Predicted mask")
            writer.grab_frame()
    plt.close(figure)


def export_brain_only(image_obj, image, mask, destination):
    brain = (image * mask).astype(np.float32)
    nib.save(nib.Nifti1Image(brain, image_obj.affine, image_obj.header),
             destination / "brain_only.nii.gz")
    try:
        import SimpleITK as sitk
        sitk.WriteImage(sitk.ReadImage(str(destination / "brain_only.nii.gz")),
                        str(destination / "brain_only.mha"), True)
        mha = True
    except Exception:
        mha = False
    return mha


def selected_rows() -> list[dict]:
    rows = read_csv(SOURCE_METRICS)
    selected = []
    for domain in ("CAMRI", "Mouse"):
        subset = sorted((row for row in rows if row["domain"] == domain),
                        key=lambda row: float(row["filtered_dice"]))
        indices = (0, len(subset) // 2, len(subset) - 1)
        for rank, index in zip(("difficult", "median", "best"), indices):
            selected.append({**subset[index], "representative_rank": rank})
    return selected


def save_summary_figure(records: list[dict], destination: Path):
    figure, axes = plt.subplots(len(records), 4, figsize=(13, 3 * len(records)),
                               constrained_layout=True)
    for row_index, row in enumerate(records):
        image = row["_image"]; prediction = row["_prediction"]
        z = int(np.argmax(prediction.sum((0, 1))))
        box = crop_box(prediction, 10)
        mri = display_slice(normalized(image)[box[0], box[1], :], z)
        mask = display_slice(prediction[box[0], box[1], :], z)
        panels = (mri, mri, mask, mri * mask)
        for column, panel in enumerate(panels):
            axes[row_index, column].imshow(panel, cmap="gray", vmin=0, vmax=1)
            axes[row_index, column].axis("off")
        axes[row_index, 1].imshow(np.ma.masked_where(~mask, mask), cmap="winter",
                                 alpha=.2, vmin=0, vmax=1)
        axes[row_index, 1].contour(mask, [.5], colors=ACCENT, linewidths=1)
        axes[row_index, 0].set_ylabel(
            f"{row['domain']} · {row['representative_rank']}\n{row['subject']}\n"
            f"Dice {float(row['filtered_dice']):.4f}", fontsize=9)
    for axis, title in zip(axes[0], ("MRI", "Prediction overlay", "Brain mask", "Skull-stripped brain")):
        axis.set_title(title, fontsize=12, fontweight="semibold")
    figure.suptitle("Frozen learned-query brain segmentation · representative test cases",
                   fontsize=16, fontweight="bold")
    figure.savefig(destination, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT.relative_to(ROOT)))
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    records, manifest = [], []
    for row in selected_rows():
        subject = row["subject"]
        destination = output / row["domain"].lower() / f"{row['representative_rank']}_{subject}"
        destination.mkdir(parents=True, exist_ok=True)
        image_obj = nib.load(row["image_path"])
        image = np.asarray(image_obj.dataobj, dtype=np.float32)
        target = np.asarray(nib.load(row["ground_truth_path"]).dataobj) > 0
        prediction = np.asarray(nib.load(row["filtered_prediction_path"]).dataobj) > 0
        if image.shape != target.shape or image.shape != prediction.shape:
            raise ValueError(f"geometry mismatch for {row['domain']} {subject}")

        page_counts = {}
        for kind in ("original", "overlay", "brain_only", "mask"):
            page_counts[kind] = save_contact_pages(
                image, prediction, destination / "contact_sheets" / kind,
                f"{row['domain']} · {subject}", kind,
            )
        save_orthogonal(image, prediction, destination / "orthogonal_views.png",
                        f"{row['domain']} · {subject}")
        # Render in isotropic physical space so slice thickness cannot visually
        # flatten or elongate the brain.
        spacing = nib.affines.voxel_sizes(image_obj.affine)
        isotropic_zoom = spacing / spacing.min()
        render_image = ndimage.zoom(image, isotropic_zoom, order=1)
        render_mask = ndimage.zoom(prediction.astype(np.uint8), isotropic_zoom, order=0) > 0
        surface_count = save_surface_views(
            render_mask, destination / "surface_renderings", f"{row['domain']} · {subject}")
        volume_count = save_volume_views(
            render_image, render_mask, destination / "volume_renderings", f"{row['domain']} · {subject}")
        save_comparison(image, target, prediction, destination / "ground_truth_comparison.png",
                        f"{row['domain']} · {subject}", float(row["filtered_dice"]))
        save_movie(image, prediction, destination / "slice_review.mp4",
                   f"{row['domain']} · {subject}")
        mha = export_brain_only(image_obj, image, prediction, destination)
        # Orthogonal PNG, comparison PNG, MP4, and NIfTI are always present;
        # MHA is the optional fifth non-render artifact.
        saved_count = sum(page_counts.values()) + surface_count + volume_count + 4 + int(mha)
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        manifest.append({
            "domain": row["domain"], "subject": subject,
            "representative_rank": row["representative_rank"],
            "filtered_dice": row["filtered_dice"], "visualization_files": saved_count,
            "subject_output": str(destination),
        })
        records.append({**clean, "_image": image, "_prediction": prediction})
        print(f"{row['domain']} {row['representative_rank']} {subject}: "
              f"{saved_count} files", flush=True)

    save_summary_figure(records, output / "publication_summary_figure.png")
    write_csv(output / "manifest.csv", manifest)
    total = sum(int(row["visualization_files"]) for row in manifest) + 1
    checkpoint_epoch = 17
    lines = [
        "# Publication visualization summary", "",
        "These figures reuse existing test MRI, expert masks, and deterministic",
        "largest-26-component predictions. No training or inference was run.",
        "The model, checkpoint, preprocessing, and threshold were not changed.", "",
        f"- Checkpoint: `{CHECKPOINT.relative_to(ROOT)}` (epoch {checkpoint_epoch})",
        f"- Subjects: {len(manifest)} (best, median, and difficult in each domain)",
        f"- Visualization/export files generated: {total}", "",
        "## Subjects", "",
        "| Dataset | Rank | Subject | Existing filtered Dice | Output |",
        "|---|---|---|---:|---|",
    ]
    for row in manifest:
        relative = Path(row["subject_output"]).relative_to(output)
        lines.append(
            f"| {row['domain']} | {row['representative_rank']} | `{row['subject']}` | "
            f"{float(row['filtered_dice']):.4f} | `{relative}` |"
        )
    lines += [
        "", "## Primary figure", "",
        "`publication_summary_figure.png` contains all six representative cases.",
        "Each subject directory contains full-volume contact sheets, six surface",
        "views, seven intensity-volume views, orthogonal contours, an MP4, an",
        "expert comparison, and skull-stripped NIfTI/MHA exports.",
    ]
    (output / "publication_summary.md").write_text("\n".join(lines) + "\n")
    (output / "configuration.json").write_text(json.dumps({
        "training_performed": False, "inference_performed": False,
        "selection": "minimum, median, maximum existing filtered Dice per domain",
        "source_metrics": str(SOURCE_METRICS), "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint_epoch, "prediction_type":
            "existing deterministic largest-26-connected-component output",
    }, indent=2))
    print(f"Generated {total} files under {output}")


if __name__ == "__main__":
    main()
