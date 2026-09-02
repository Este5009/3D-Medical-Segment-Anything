#!/usr/bin/env python3
"""FAST diagnostic: does the frozen RS2-Net encoder use finer input detail?

Read-only / inference-only. No training, no weight changes, no decoder. Same
3 Mouse subjects as outputs/boundary_stage_localization/.

IMPORTANT COST FINDING (see run log / findings.md): a throwaway sanity check
constructing the encoder at a modestly larger input (192x128x160, 1.5x the
control voxel count) and running ONE forward pass took **173 seconds** on
this CPU-only machine (MPS has no Conv3d support here), versus ~1 second for
the control 128x128x160 size -- a ~170x slowdown for a 1.5x voxel increase.
This is grossly disproportionate to voxel count and was confirmed to not be
a resource-contention fluke (no other heavy process was running). Given this,
the original plan (3 subjects x 3 finer candidates through the encoder) was
cut down to keep this diagnostic within a practical runtime budget:

  - The CHEAP measurement (expert-mask-only corrected-NN resampling; no
    network) is run for ALL candidates, including a substantial and a
    near-native candidate, since it costs a fraction of a second each.
  - The EXPENSIVE measurement (actual frozen-encoder level0 forward pass) is
    run only for control and ONE finer candidate ("modest"), each for all 3
    subjects, since even that alone costs ~3 minutes/subject. "substantial"
    and "near-native" encoder costs are extrapolated from the measured
    control->modest scaling and reported, not run.

Model-axis convention throughout (unchanged from the prior diagnostic):
axis0 -> native Y (biggest measured native-detail loss, ~2.5-3.6x), axis1 ->
native Z (already finer than native -- left unchanged), axis2 -> native X
(~1.6-2.3x loss). "modest" only refines axis0 (the single biggest offender)
to keep the one expensive run as cheap as possible while still testing the
real question.

Reused, unmodified: DefaultPreprocessor, the corrected NN mask-resampling
call pattern (from corrected_label_preprocessing.py), digital_contour_statistics
/ feature_image (from diagnose_model_spatial_resolution.py), RS2NetEncoderAdapter.
Only the preprocessing spacing is overridden (in-memory dict mutation on the
loaded plans configuration, not a file edit).
"""
from __future__ import annotations
import csv, json, resource, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

from diagnose_model_spatial_resolution import digital_contour_statistics, feature_image
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import _pad_and_center_crop, load_json

OUT = ROOT / "outputs/higher_resolution_encoder_diagnostic"
LEVEL0_DIR = ROOT / "outputs/true_full_resolution_level0_decoder"
BOUNDARY_STAGE = ROOT / "outputs/boundary_stage_localization"

SUBJECTS = [
    {"role": "strongly_pixelated", "subject": "POLYIC_20190517_mouse39__E12_P1"},
    {"role": "median", "subject": "POLYIC_20190510_mouse37__E2_P1"},
    {"role": "best_true_level0", "subject": "POLYIC_20190517_mouse37__E3_P1"},
]

# Cheap (expert-mask-only, no encoder) candidates: control + every finer
# option we considered, so the geometric-ceiling numbers are complete.
ALL_CANDIDATES = [
    {"name": "control", "spacing": (0.25, 0.20, 0.16), "tile": (128, 128, 160), "run_encoder": True},
    {"name": "modest_axis0_only", "spacing": (0.16667, 0.20, 0.16), "tile": (192, 128, 160), "run_encoder": True},
    {"name": "substantial_axis0+2", "spacing": (0.142857, 0.20, 0.1), "tile": (224, 128, 256), "run_encoder": False},
    {"name": "near_native", "spacing": (0.0714, 0.20, 0.0667), "tile": (448, 128, 384), "run_encoder": False},
]


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def preprocess_at_spacing(image_path, mask_path, paths, spacing, tile_size, need_image=True):
    """Same crop/transpose/resample pipeline as corrected_label_preprocessing.py,
    with configuration.spacing overridden in-memory to the candidate spacing.
    If need_image is False, only the (cheap) expert-mask resample runs --
    the image is still cropped/resampled by run_case since RS2 does image and
    seg together, but the caller can skip building the padded image tensor."""
    from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from RS2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    from RS2.utilities.plans_handling.plans_handler import PlansManager
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

    root = paths.baseline_project / "RS2/jsons"
    manager = PlansManager(load_json(root / "plans.json"))
    configuration = manager.get_configuration("3d_fullres")
    configuration.configuration["spacing"] = list(spacing)  # in-memory only; no file touched

    data, _, properties = DefaultPreprocessor(False).run_case(
        [str(image_path)], str(mask_path), manager, configuration, load_json(root / "dataset.json"))
    properties = dict(properties)
    resampled_shape = tuple(int(x) for x in data.shape[1:])
    properties["shape_after_resampling"] = resampled_shape

    reader = manager.image_reader_writer_class()
    segmentation, _ = reader.read_seg(str(mask_path))
    segmentation = segmentation.transpose([0, *[axis + 1 for axis in manager.transpose_forward]])
    segmentation = segmentation[(slice(None), *bounding_box_to_slice(properties["bbox_used_for_cropping"]))]
    source_spacing = [properties["spacing"][axis] for axis in manager.transpose_forward]
    categorical = resample_data_or_seg_to_shape(
        (segmentation > 0).astype(np.uint8), resampled_shape, source_spacing, list(spacing),
        is_seg=True, order=0, order_z=0, force_separate_z=None)
    target = _pad_and_center_crop(torch.from_numpy(categorical.astype(np.float32)).unsqueeze(0), tile_size)

    image = None
    if need_image:
        image = _pad_and_center_crop(torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0), tile_size)
        if image.shape != target.shape:
            raise ValueError(f"Image/target mismatch: {image.shape} vs {target.shape}")
    return image, target, resampled_shape, properties


def boundary_band(mask2d, iterations=2):
    return ndimage.binary_dilation(mask2d, iterations=iterations) & ~ndimage.binary_erosion(mask2d, iterations=iterations)


def gradient_contrast(activation2d, boundary_band2d):
    gy, gx = np.gradient(activation2d.astype(np.float64))
    magnitude = np.sqrt(gy ** 2 + gx ** 2)
    inside = magnitude[boundary_band2d]
    outside = magnitude[~boundary_band2d]
    if inside.size == 0 or outside.size == 0 or outside.mean() < 1e-8:
        return float("nan")
    return float(inside.mean() / outside.mean())


def native_downsample_factors(spacing_model, native_spacing_xyz):
    """Mirrors effective_native_footprint's axis convention (transpose [1,0,2]):
    model axis0 -> native y, axis1 -> native z, axis2 -> native x."""
    model = np.asarray(spacing_model)
    native_order = np.asarray([model[2], model[0], model[1]])  # x, y, z
    return native_order / np.asarray(native_spacing_xyz)


def expert_ceiling_row(case, candidate, paths):
    """Cheap: resample only the expert mask (corrected NN) at this spacing
    and measure its own boundary articulation -- the geometric ceiling."""
    start = time.time()
    _, target, resampled_shape, properties = preprocess_at_spacing(
        Path(case["image_path"]), Path(case["ground_truth_path"]), paths,
        candidate["spacing"], candidate["tile"], need_image=False)
    elapsed = time.time() - start
    model_expert = target[0, 0].numpy() > 0.5
    content = model_expert.sum(axis=(0, 2))
    k = int(np.argmax(content)) if content.max() >= 20 else model_expert.shape[1] // 2
    contour = digital_contour_statistics(model_expert[:, k, :])
    native_spacing = tuple(float(x) for x in nib.load(case["ground_truth_path"]).header.get_zooms()[:3])
    downsample = native_downsample_factors(candidate["spacing"], native_spacing)
    return {
        "subject": case["subject"], "role": case["role"], "candidate": candidate["name"],
        "tile_shape": str(candidate["tile"]), "resampled_shape": str(resampled_shape),
        "voxel_count": int(np.prod(candidate["tile"])),
        "voxel_ratio_vs_control": int(np.prod(candidate["tile"])) / int(np.prod(ALL_CANDIDATES[0]["tile"])),
        "native_downsample_factor_x": downsample[0], "native_downsample_factor_y": downsample[1], "native_downsample_factor_z": downsample[2],
        "expert_direction_changes_per_100_at_this_grid": contour["direction_changes_per_100_steps"] if contour else None,
        "expert_axis_run_max_at_this_grid": contour["axis_run_max"] if contour else None,
        "cheap_resample_seconds": elapsed,
    }


def main():
    start_all = time.time()
    config = load_json(ROOT / "configs/true_full_resolution_level0_decoder.yaml")
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    per_subject = {r["subject"]: r for r in csv.DictReader(open(LEVEL0_DIR / "per_subject_comparison.csv")) if r["domain"] == "Mouse"}
    native_stats = {r["subject"]: r for r in csv.DictReader(open(BOUNDARY_STAGE / "stage_metrics.csv"))}
    cases = []
    for s in SUBJECTS:
        row = dict(per_subject[s["subject"]]); row.update(s); cases.append(row)

    OUT.mkdir(parents=True, exist_ok=True)

    # --- Cheap pass: expert-mask-only geometric ceiling, ALL candidates ---
    ceiling_rows = []
    for candidate in ALL_CANDIDATES:
        for case in cases:
            row = expert_ceiling_row(case, candidate, paths)
            ceiling_rows.append(row)
            print(f"[ceiling] {row['subject']} {candidate['name']} dc/100={row['expert_direction_changes_per_100_at_this_grid']:.1f} "
                  f"shape={row['resampled_shape']} ({row['cheap_resample_seconds']:.2f}s)", flush=True)
    with (OUT / "expert_ceiling_all_candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ceiling_rows[0])); w.writeheader(); w.writerows(ceiling_rows)

    # --- Expensive pass: real frozen-encoder level0, control + modest only ---
    encoder_rows = []
    figures_cache = {}
    for candidate in ALL_CANDIDATES:
        if not candidate["run_encoder"]:
            continue
        t0 = time.time()
        model = RS2NetEncoderAdapter(paths, image_size=candidate["tile"], in_channels=1, out_channels=1, feature_size=48).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        build_seconds = time.time() - t0
        print(f"[encoder build] {candidate['name']} tile={candidate['tile']} strict-load OK in {build_seconds:.1f}s", flush=True)
        for case in cases:
            before = peak_mib(); t0 = time.time()
            with torch.inference_mode():
                image, target, resampled_shape, properties = preprocess_at_spacing(
                    Path(case["image_path"]), Path(case["ground_truth_path"]), paths,
                    candidate["spacing"], candidate["tile"], need_image=True)
                features = model(image)
            elapsed = time.time() - t0
            peak = peak_mib()
            activation = feature_image(features["level0"])
            model_expert = target[0, 0].numpy() > 0.5
            content = model_expert.sum(axis=(0, 2))
            k = int(np.argmax(content)) if content.max() >= 20 else model_expert.shape[1] // 2
            expert_slice = model_expert[:, k, :]
            activation_slice = activation[:, k, :]
            contour = digital_contour_statistics(expert_slice)
            band = boundary_band(expert_slice)
            contrast = gradient_contrast(activation_slice, band)
            row = {
                "subject": case["subject"], "role": case["role"], "candidate": candidate["name"],
                "tile_shape": str(candidate["tile"]), "resampled_shape": str(resampled_shape),
                "voxel_ratio_vs_control": int(np.prod(candidate["tile"])) / int(np.prod(ALL_CANDIDATES[0]["tile"])),
                "expert_direction_changes_per_100_at_this_grid": contour["direction_changes_per_100_steps"] if contour else None,
                "level0_boundary_gradient_contrast": contrast,
                "elapsed_seconds": elapsed, "peak_process_memory_mib": peak, "incremental_memory_mib": peak - before,
            }
            encoder_rows.append(row)
            figures_cache.setdefault(case["subject"], {})[candidate["name"]] = {
                "activation_slice": activation_slice, "expert_slice": expert_slice,
            }
            print(f"[encoder] {case['subject']} [{candidate['name']}] dc/100={row['expert_direction_changes_per_100_at_this_grid']:.1f} "
                  f"contrast={contrast:.3f} t={elapsed:.1f}s peak={peak:.0f}MiB", flush=True)
        del model

    with (OUT / "encoder_level0_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(encoder_rows[0])); w.writeheader(); w.writerows(encoder_rows)

    # --- Extrapolate substantial/near-native encoder cost from measured scaling ---
    control_time = next(r["elapsed_seconds"] for r in encoder_rows if r["candidate"] == "control")
    modest_time = next(r["elapsed_seconds"] for r in encoder_rows if r["candidate"] == "modest_axis0_only")
    modest_ratio = next(c for c in ALL_CANDIDATES if c["name"] == "modest_axis0_only")
    voxel_ratio_modest = int(np.prod(modest_ratio["tile"])) / int(np.prod(ALL_CANDIDATES[0]["tile"]))
    time_per_voxel_ratio_unit = (modest_time - control_time) / voxel_ratio_modest  # crude linear-in-slowdown extrapolation base
    slowdown_factor = modest_time / max(control_time, 1e-6)
    extrapolation = {}
    for candidate in ALL_CANDIDATES:
        if candidate["run_encoder"]:
            continue
        vr = int(np.prod(candidate["tile"])) / int(np.prod(ALL_CANDIDATES[0]["tile"]))
        # Conservative: assume the same super-linear slowdown-per-voxel-ratio
        # observed from control->modest holds; report as an order-of-magnitude
        # estimate, not a precise prediction.
        est_seconds = control_time * (slowdown_factor ** (vr / voxel_ratio_modest))
        extrapolation[candidate["name"]] = {
            "voxel_ratio_vs_control": vr, "estimated_seconds_per_sample": est_seconds,
            "estimated_minutes_per_sample": est_seconds / 60,
        }

    elapsed_all = time.time() - start_all
    (OUT / "run_meta.json").write_text(json.dumps({
        "elapsed_seconds_this_script": elapsed_all,
        "control_encoder_seconds_per_sample": control_time,
        "modest_encoder_seconds_per_sample": modest_time,
        "measured_slowdown_factor_for_1.5x_voxels": slowdown_factor,
        "extrapolated_unrun_candidates": extrapolation,
    }, indent=2))
    print(f"\ncontrol={control_time:.2f}s modest={modest_time:.2f}s slowdown_factor={slowdown_factor:.1f}x for {voxel_ratio_modest:.2f}x voxels")
    print(json.dumps(extrapolation, indent=2))

    # --- One compact figure per subject: native / control+level0 / modest+level0 ---
    manifest = []
    for case in cases:
        subject = case["subject"]
        native_expert = np.asarray(nib.load(case["ground_truth_path"]).dataobj) > 0
        z_native = int(np.argmax(native_expert.sum(axis=(0, 1))))
        native_slice = native_expert[:, :, z_native]
        native_mri = np.asarray(nib.load(case["image_path"]).dataobj, dtype=np.float32)[:, :, z_native]

        fig, ax = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
        p1, p99 = np.percentile(native_mri, [1, 99])
        ax[0, 0].imshow(np.clip((native_mri - p1) / max(p99 - p1, 1e-8), 0, 1).T, cmap="gray", origin="lower")
        ax[0, 0].set_title("Native MRI (max-content slice)")
        ax[0, 1].imshow(native_slice.T, cmap="gray", origin="lower")
        ax[0, 1].set_title(f"Native expert\n(dc/100={float(native_stats[subject]['native_expert_direction_changes_per_100']):.1f})")

        for col, name in enumerate(["control", "modest_axis0_only"]):
            cache = figures_cache[subject][name]
            row = next(r for r in encoder_rows if r["subject"] == subject and r["candidate"] == name)
            a = ax[1, col]
            a.imshow(cache["activation_slice"].T, cmap="magma", origin="lower")
            a.contour(cache["expert_slice"].T, levels=[0.5], colors="cyan", linewidths=1.2)
            a.set_title(f"{name}: level0 |activation|\ndc/100={row['expert_direction_changes_per_100_at_this_grid']:.1f}, "
                        f"contrast={row['level0_boundary_gradient_contrast']:.2f}\n"
                        f"tile={row['tile_shape']} t={row['elapsed_seconds']:.1f}s mem={row['peak_process_memory_mib']:.0f}MiB")
        for a in ax.flat:
            a.axis("off")
        fig.suptitle(f"Mouse {subject} | {case['role']} | control vs. modest (finer axis0 only)")
        fig_path = OUT / f"{case['role']}_{subject}.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        manifest.append({"subject": subject, "role": case["role"], "figure_path": str(fig_path)})
        print(f"figure: {subject}", flush=True)

    with (OUT / "figure_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)

    print(f"\nTotal elapsed (this script): {time.time() - start_all:.1f} s")


if __name__ == "__main__":
    main()
