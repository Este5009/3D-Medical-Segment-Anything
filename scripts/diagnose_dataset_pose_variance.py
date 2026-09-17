#!/usr/bin/env python3
"""How much does the RAW anatomy's pose actually vary across and within the
three real datasets, before any of this project's own preprocessing touches
it? Answers the standing question from the space-invariance tests: is the
sensitivity we measured a real-world risk, or an artefact of testing angles
this data never actually contains?

Two things are measured per subject, directly from the expert mask in its
own native physical (mm) space -- no resampling, no model:

  orientation -- the mask's dominant physical axis (first principal
    component of its foreground-voxel point cloud), reported as an angle
    to a fixed lab-frame reference direction. Spread within a dataset =
    how much any two same-dataset subjects could differ in pose; the gap
    between datasets' means = a systematic between-dataset offset.
  position    -- the mask's centroid, in mm, relative to the physical
    center of its own image volume. Same within/between logic.

Also records each file's raw NIfTI orientation code (e.g. "RAS", "LPI") --
a free, near-zero-cost check for a trivial axis-convention mismatch between
datasets before trusting the geometric numbers at all.

CPU-only: nibabel + numpy + scipy, no torch, no GPU, no model forward pass.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

OUT = ROOT / "outputs" / "dataset_pose_variance"
OUT.mkdir(parents=True, exist_ok=True)

REFERENCE_AXIS = np.array([0.0, 0.0, 1.0])  # fixed lab-frame direction every subject's principal axis is measured against


def rows(p):
    return list(csv.DictReader(open(p)))


def load_json(p):
    return json.loads(Path(p).read_text())


def subject_list(limit_per_dataset):
    """Real subjects actually used by this project's own splits/metrics --
    same files the invariance tests drew from, so this diagnosis speaks
    directly to those results."""
    out = []

    cam = rows(ROOT / "outputs/generalization_pilot/metrics.csv")
    seen = set()
    for r in cam:
        if r["subject"] in seen:
            continue
        seen.add(r["subject"])
        out.append({"dataset": "CAMRI (rat brain)", "subject": r["subject"], "mask_path": r["mask_path"]})
    if limit_per_dataset:
        out = [r for r in out if r["dataset"] != "CAMRI (rat brain)"] + \
              [r for r in out if r["dataset"] == "CAMRI (rat brain)"][:limit_per_dataset]

    mouse = rows(ROOT / "outputs/mouse_external_evaluation/subject_metrics.csv")
    mouse_rows = [{"dataset": "Mouse (POLYIC brain)", "subject": r["scan_id"], "mask_path": r["ground_truth_path"]} for r in mouse]
    if limit_per_dataset:
        mouse_rows = mouse_rows[:limit_per_dataset]
    out += mouse_rows

    stroke = rows(ROOT / "outputs/mouse_stroke_lesion/metrics.csv")
    stroke_rows = [{"dataset": "Mouse (stroke lesion)", "subject": r["subject"], "mask_path": r["mask_path"]} for r in stroke]
    if limit_per_dataset:
        stroke_rows = stroke_rows[:limit_per_dataset]
    out += stroke_rows

    return [r for r in out if Path(r["mask_path"]).exists()]


def principal_axis_and_centroid(mask_path):
    """PCA on the foreground voxel cloud in PHYSICAL mm space (via the
    NIfTI affine, not raw voxel indices -- so this is comparable across
    datasets even if they have different voxel sizes)."""
    img = nib.load(str(mask_path))
    data = np.asarray(img.dataobj) > 0
    if data.sum() < 20:
        return None
    # argwhere on the array nibabel hands back already yields (i,j,k) voxel
    # indices in the exact order img.affine expects -- apply it directly.
    voxel_idx = np.argwhere(data)
    ones = np.ones((voxel_idx.shape[0], 1))
    homog = np.hstack([voxel_idx.astype(np.float64), ones])
    physical = (img.affine @ homog.T).T[:, :3]  # (N,3) mm coordinates

    centroid = physical.mean(axis=0)
    centered = physical - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]  # largest-variance direction
    if principal[2] < 0:  # fix sign ambiguity of an eigenvector
        principal = -principal
    principal = principal / np.linalg.norm(principal)

    extent_mm = physical.max(axis=0) - physical.min(axis=0)
    volume_center = nib.affines.apply_affine(img.affine, np.array(data.shape) / 2.0)
    centroid_offset = centroid - volume_center

    axcodes = "".join(nib.aff2axcodes(img.affine))
    angle_deg = np.degrees(np.arccos(np.clip(abs(np.dot(principal, REFERENCE_AXIS)), -1, 1)))
    return {
        "principal_axis": principal, "centroid_offset_mm": centroid_offset,
        "axis_angle_deg": angle_deg, "axcodes": axcodes,
        "extent_mm": extent_mm, "voxel_count": int(data.sum()),
    }


def main():
    ap_limit = 60
    records = subject_list(ap_limit)
    per_subject, axis_by_subject = [], {}
    for r in records:
        res = principal_axis_and_centroid(r["mask_path"])
        if res is None:
            continue
        key = (r["dataset"], r["subject"])
        axis_by_subject[key] = res["principal_axis"]
        per_subject.append({
            "dataset": r["dataset"], "subject": r["subject"],
            "axis_angle_to_fixed_reference_deg": res["axis_angle_deg"],
            "centroid_offset_x_mm": res["centroid_offset_mm"][0],
            "centroid_offset_y_mm": res["centroid_offset_mm"][1],
            "centroid_offset_z_mm": res["centroid_offset_mm"][2],
            "centroid_offset_mag_mm": float(np.linalg.norm(res["centroid_offset_mm"])),
            "extent_x_mm": res["extent_mm"][0], "extent_y_mm": res["extent_mm"][1], "extent_z_mm": res["extent_mm"][2],
            "axcodes": res["axcodes"], "voxel_count": res["voxel_count"],
        })
        print(f"{r['dataset']:24s} {r['subject']:32s} centroid-offset={np.linalg.norm(res['centroid_offset_mm']):6.2f} mm   {res['axcodes']}", flush=True)

    # Angle-to-a-single-fixed-reference conflates two different things:
    # real physical pose AND each dataset's own NIfTI storage-orientation
    # convention (confirmed different across these three -- see axcodes
    # above), so it is not a fair between-dataset comparison, and is kept
    # in the per-subject CSV for reference only. What IS fair, and what
    # actually answers "how much does pose vary in practice", is each
    # subject's deviation from ITS OWN dataset's mean axis -- that cancels
    # out any fixed storage-convention offset and isolates real
    # within-dataset spread, which is the number the summary uses.
    # A principal axis has a sign ambiguity (v and -v are the same line) --
    # my earlier "force z-component positive" heuristic is numerically
    # unstable whenever the true axis is close to perpendicular to z
    # (exactly CAMRI's case), spuriously flipping some subjects and
    # corrupting a naive vector mean. The correct, sign-invariant way to
    # average axes: sum each axis's outer product v @ v.T (sign-invariant,
    # since (-v)@(-v).T == v@v.T) and take that matrix's dominant
    # eigenvector as the mean direction.
    datasets = sorted(set(r["dataset"] for r in per_subject))
    mean_axis = {}
    for d in datasets:
        vecs = np.array([v for (dd, s), v in axis_by_subject.items() if dd == d])
        outer_sum = (vecs[:, :, None] * vecs[:, None, :]).sum(axis=0)
        eigvals, eigvecs = np.linalg.eigh(outer_sum)
        mean_axis[d] = eigvecs[:, np.argmax(eigvals)]
    for r in per_subject:
        v = axis_by_subject[(r["dataset"], r["subject"])]
        r["axis_angle_within_dataset_deg"] = float(np.degrees(np.arccos(
            np.clip(abs(np.dot(v, mean_axis[r["dataset"]])), -1, 1))))

    with (OUT / "per_subject_pose.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_subject[0]))
        w.writeheader()
        w.writerows(per_subject)

    summary = []
    for d in datasets:
        q = [r for r in per_subject if r["dataset"] == d]
        angles = np.array([r["axis_angle_within_dataset_deg"] for r in q])
        offsets = np.array([r["centroid_offset_mag_mm"] for r in q])
        codes = sorted(set(r["axcodes"] for r in q))
        summary.append({
            "dataset": d, "n": len(q),
            "axis_angle_mean_deg": float(angles.mean()), "axis_angle_sd_deg": float(angles.std()),
            "axis_angle_min_deg": float(angles.min()), "axis_angle_max_deg": float(angles.max()),
            "axis_angle_range_deg": float(angles.max() - angles.min()),
            "centroid_offset_mean_mm": float(offsets.mean()), "centroid_offset_sd_mm": float(offsets.std()),
            "centroid_offset_max_mm": float(offsets.max()),
            "nifti_orientation_codes": ",".join(codes),
        })
    with (OUT / "summary_by_dataset.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))

    # ---- figure: within- vs between-dataset spread ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    positions = range(len(datasets))
    angle_groups = [[r["axis_angle_within_dataset_deg"] for r in per_subject if r["dataset"] == d] for d in datasets]
    offset_groups = [[r["centroid_offset_mag_mm"] for r in per_subject if r["dataset"] == d] for d in datasets]
    for ax, groups, title, ylabel in (
        (axes[0], angle_groups, "Mask principal-axis deviation from its own dataset's mean pose", "degrees"),
        (axes[1], offset_groups, "Mask centroid offset from volume center", "mm"),
    ):
        bp = ax.boxplot(groups, positions=positions, widths=0.55, patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#EDF1F2", edgecolor="#55636F")
        for i, g in enumerate(groups):
            ax.scatter([i] * len(g), g, s=14, alpha=0.55, color="#C2620E", zorder=3)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(datasets, rotation=12, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.3)
    fig.suptitle("How much does the real, unprocessed anatomy's pose vary -- within and across datasets?", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "pose_variance.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUT}/pose_variance.png")


if __name__ == "__main__":
    main()
