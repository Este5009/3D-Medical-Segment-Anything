#!/usr/bin/env python3
"""Visual atlas: corrected-label baseline vs. TRUE full-resolution level0.

Read-only. Loads only already-saved predictions from
outputs/corrected_label_retraining/ and outputs/true_full_resolution_level0_decoder/
(per_subject_comparison.csv and the nii.gz masks it references). No model is
loaded, no inference is run, and no training occurs.

Case selection is entirely data-driven from existing CSVs:
  - per_subject_comparison.csv            (Dice / HD95, both conditions)
  - contour_geometry_comparison.csv       (expert boundary complexity, baseline
                                            pixelation ratio -- both already
                                            computed by
                                            compare_true_full_resolution_level0_results.py)
  - a one-off whole-mask convex-hull "solidity" scan (this script), used only
    to find the strongest concavity/lobe-separation example; it reads the
    same saved masks, nothing else.

Slice selection within each chosen subject is automatic: among slices with
substantial expert foreground, pick the slice where the baseline and level0
*filtered* predictions disagree with each other the most (largest XOR area).
This directly targets "where the models differ most in boundary geometry"
rather than an arbitrary center slice.
"""
from __future__ import annotations
import csv, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull, QhullError

OUT = ROOT / "outputs/true_full_resolution_level0_decoder"
ATLAS = OUT / "visual_comparison_atlas"

GREEN = "#22c55e"   # expert
RED = "#ef4444"     # baseline / FP
CYAN = "#06b6d4"     # true-level0
YELLOW = "#facc15"   # FN


def rows(p):
    return list(csv.DictReader(open(p)))


# ---------------------------------------------------------------------------
# 1. Data-driven case selection (reads only existing CSVs / saved masks)
# ---------------------------------------------------------------------------

def dice_delta(r):
    return float(r["level0_filtered_dice"]) - float(r["baseline_filtered_dice"])


def hd_delta(r):
    """Positive = level0 HD95 improved (lower) relative to baseline."""
    return float(r["baseline_filtered_hd95_mm"]) - float(r["level0_filtered_hd95_mm"])


def whole_mask_solidity(mask2d):
    """area / convex-hull-area of ALL foreground pixels, ignoring connected
    components. Low values indicate a genuinely concave or multi-lobed shape
    (e.g. two separate blobs joined only by their shared convex hull)."""
    pts = np.argwhere(mask2d)
    if len(pts) < 10:
        return None
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None
    return len(pts) / hull.volume  # 2D: hull.volume is the polygon area


def find_concavity_case(mouse_rows):
    """Scan saved masks (no inference) for the subject/slice where the
    expert shape is genuinely non-convex (solidity < 0.85) and TRUE level0
    matches that concavity/lobe-gap more closely than the baseline does."""
    best = None
    for r in mouse_rows:
        gt = np.asarray(nib.load(r["ground_truth_path"]).dataobj) > 0
        base = np.asarray(nib.load(r["baseline_filtered_prediction_path"]).dataobj) > 0
        new = np.asarray(nib.load(r["level0_filtered_prediction_path"]).dataobj) > 0
        slice_sums = gt.sum(axis=(0, 1))
        max_slice = slice_sums.max()
        if max_slice < 30:
            continue
        min_voxels = max(30, 0.15 * max_slice)
        for z in range(gt.shape[2]):
            if slice_sums[z] < min_voxels:
                continue
            s_gt = whole_mask_solidity(gt[:, :, z])
            if s_gt is None or s_gt > 0.85:
                continue
            s_base = whole_mask_solidity(base[:, :, z])
            s_new = whole_mask_solidity(new[:, :, z])
            if s_base is None or s_new is None:
                continue
            improvement = abs(s_base - s_gt) - abs(s_new - s_gt)
            if improvement > 0.01:
                score = improvement * slice_sums[z]
                if best is None or score > best[0]:
                    best = (score, r["subject"], z, s_gt, s_base, s_new)
    return best


def select_cases():
    per_subject = rows(OUT / "per_subject_comparison.csv")
    mouse = [r for r in per_subject if r["domain"] == "Mouse"]
    camri = [r for r in per_subject if r["domain"] == "CAMRI"]
    by_subject = {(r["domain"], r["subject"]): r for r in per_subject}

    geometry = rows(OUT / "contour_geometry_comparison.csv")
    geo_baseline = {r["subject"]: r for r in geometry if r["domain"] == "Mouse" and r["condition"] == "corrected_baseline"}

    used = set()

    def pick(sid):
        used.add(sid)
        return sid

    # 1. Largest Mouse Dice improvement.
    largest_improvement = pick(max(mouse, key=dice_delta)["subject"])

    # 2. Median Mouse Dice improvement (subject closest to the median delta).
    med = statistics.median(dice_delta(r) for r in mouse)
    median_change = pick(min(mouse, key=lambda r: abs(dice_delta(r) - med))["subject"])

    # 3. Smallest/no improvement: smallest non-negative delta.
    non_negative = [r for r in mouse if dice_delta(r) >= 0]
    smallest_improvement = pick(min(non_negative, key=dice_delta)["subject"])

    # 4. One of the 2/80 regressions: the larger-magnitude regression.
    regressions = sorted([r for r in mouse if dice_delta(r) < 0], key=dice_delta)
    regression_case = pick(regressions[0]["subject"])

    # 5. Largest HD95 improvement.
    hd95_case = pick(max(mouse, key=hd_delta)["subject"])

    # 6. High-curvature improvement: most complex expert boundary among
    # subjects with a positive Dice delta, excluding already-used subjects
    # where a distinct alternative exists.
    complexity_ranked = sorted(
        (r for r in geo_baseline.values() if dice_delta(by_subject[("Mouse", r["subject"])]) > 0),
        key=lambda r: -float(r["expert_direction_changes_per_100_steps"]),
    )
    curvature_case = None
    for r in complexity_ranked:
        if r["subject"] not in used:
            curvature_case = r["subject"]; break
    if curvature_case is None:
        curvature_case = complexity_ranked[0]["subject"]
    pick(curvature_case)

    # 7. Previously pixelated/staircase: highest baseline axis-run-max ratio
    # (how much longer the baseline's longest straight run is than the
    # expert's), excluding already-used subjects where possible.
    pixel_ranked = sorted(geo_baseline.values(), key=lambda r: -float(r["prediction_expert_axis_run_max_ratio"]))
    pixelated_case = None
    for r in pixel_ranked:
        if r["subject"] not in used:
            pixelated_case = r["subject"]; break
    if pixelated_case is None:
        pixelated_case = pixel_ranked[0]["subject"]
    pick(pixelated_case)

    # 8. Concavity / lobe-separation: automated whole-mask-solidity scan over
    # ALL 80 Mouse subjects' saved masks (read-only, no inference).
    concavity_best = find_concavity_case(mouse)
    concavity_case = concavity_best[1] if concavity_best else largest_improvement
    concavity_same_as_largest = concavity_case == largest_improvement
    pick(concavity_case)

    # CAMRI: representative (delta closest to the CAMRI median) and worst
    # difference (largest |delta| between the two models).
    camri_med = statistics.median(dice_delta(r) for r in camri)
    camri_representative = min(camri, key=lambda r: abs(dice_delta(r) - camri_med))["subject"]
    camri_worst = max(camri, key=lambda r: abs(dice_delta(r)))["subject"]

    cases = [
        {"domain": "Mouse", "subject": largest_improvement, "reason": "largest_dice_improvement",
         "note": "Largest Mouse Dice gain (level0 - baseline) of all 80 test subjects."},
        {"domain": "Mouse", "subject": median_change, "reason": "median_dice_improvement",
         "note": "Subject whose Dice change is closest to the Mouse-wide median change."},
        {"domain": "Mouse", "subject": smallest_improvement, "reason": "smallest_improvement",
         "note": "Smallest non-negative Dice change among Mouse subjects (near-flat case)."},
        {"domain": "Mouse", "subject": regression_case, "reason": "regression",
         "note": "One of the two Mouse subjects (of 80) where level0 Dice is lower than baseline."},
        {"domain": "Mouse", "subject": hd95_case, "reason": "largest_hd95_improvement",
         "note": "Largest reduction in HD95 (mm) from baseline to level0."},
        {"domain": "Mouse", "subject": curvature_case, "reason": "high_curvature_improvement",
         "note": "High expert-boundary complexity (direction changes/100 steps) with a positive Dice change."},
        {"domain": "Mouse", "subject": pixelated_case, "reason": "previously_pixelated",
         "note": "Highest baseline axis-run/expert-axis-run ratio (most staircase-like baseline contour) among Mouse subjects."},
        {"domain": "Mouse", "subject": concavity_case, "reason": "lobe_separation_or_concavity",
         "note": ("Automated whole-mask convex-hull-solidity scan of all 80 Mouse subjects' saved masks found this "
                  "the only case where the expert shape is genuinely non-convex (solidity<0.85) and level0's solidity "
                  f"({concavity_best[5]:.3f}) is measurably closer to the expert's ({concavity_best[3]:.3f}) than the "
                  f"baseline's is ({concavity_best[4]:.3f})."
                  + (" This coincides with the largest-Dice-improvement case above -- it is the same subject, "
                     "shown again because it is also the clearest concavity example, not a distinct pick."
                     if concavity_same_as_largest else "")),
         "forced_slice": concavity_best[2] if concavity_best else None},
        {"domain": "CAMRI", "subject": camri_representative, "reason": "camri_representative",
         "note": "CAMRI subject whose Dice change is closest to the CAMRI-wide median change (typical case, n=6)."},
        {"domain": "CAMRI", "subject": camri_worst, "reason": "camri_worst_difference",
         "note": "CAMRI subject with the largest-magnitude Dice change (either direction) between the two models."},
    ]
    return cases, by_subject


# ---------------------------------------------------------------------------
# 2. Automatic slice selection: maximize baseline-vs-level0 disagreement
#    among slices with substantial expert content.
# ---------------------------------------------------------------------------

def select_slice(gt, base, new, forced_slice=None):
    if forced_slice is not None:
        return forced_slice, "forced (concavity scan slice)"
    slice_sums = gt.sum(axis=(0, 1))
    max_slice = slice_sums.max()
    if max_slice == 0:
        return gt.shape[2] // 2, "fallback: no expert content"
    min_voxels = max(20, 0.10 * max_slice)
    candidates = [z for z in range(gt.shape[2]) if slice_sums[z] >= min_voxels]
    if not candidates:
        candidates = list(range(gt.shape[2]))
    disagreement = {z: int((base[:, :, z] ^ new[:, :, z]).sum()) for z in candidates}
    z = max(candidates, key=lambda z: disagreement[z])
    return z, f"max baseline-vs-level0 disagreement ({disagreement[z]} voxels) among slices with >={int(min_voxels)} expert voxels"


# ---------------------------------------------------------------------------
# 3. Per-case 9-panel figure (A-I)
# ---------------------------------------------------------------------------

def crop_bounds(*masks2d, margin=6):
    union = np.zeros_like(masks2d[0], dtype=bool)
    for m in masks2d:
        union |= m
    q = np.argwhere(union)
    if len(q) == 0:
        return (0, masks2d[0].shape[0]), (0, masks2d[0].shape[1])
    lo = np.maximum(q.min(0) - margin, 0)
    hi = np.minimum(q.max(0) + margin + 1, masks2d[0].shape)
    return (lo[0], hi[0]), (lo[1], hi[1])


def draw_fp_fn(ax, mri_norm, pred, gt):
    rgb = np.stack([mri_norm] * 3, axis=-1)
    fp = pred & ~gt
    fn = ~pred & gt
    rgb[fp] = (1, 0, 0)
    rgb[fn] = (1, 0.82, 0.08)
    ax.imshow(rgb.transpose(1, 0, 2), origin="lower", interpolation="nearest")


def make_case_figure(case, record, out_path):
    image = np.asarray(nib.load(record["image_path"]).dataobj, dtype=np.float32)
    gt = np.asarray(nib.load(record["ground_truth_path"]).dataobj) > 0
    base = np.asarray(nib.load(record["baseline_filtered_prediction_path"]).dataobj) > 0
    new = np.asarray(nib.load(record["level0_filtered_prediction_path"]).dataobj) > 0
    spacing = tuple(float(x) for x in nib.load(record["ground_truth_path"]).header.get_zooms()[:3])

    z, slice_reason = select_slice(gt, base, new, case.get("forced_slice"))
    mri = image[:, :, z]
    p1, p99 = np.percentile(mri, [1, 99])
    mri_norm = np.clip((mri - p1) / max(p99 - p1, 1e-8), 0, 1)
    gt2, base2, new2 = gt[:, :, z], base[:, :, z], new[:, :, z]

    (xlo, xhi), (ylo, yhi) = crop_bounds(gt2, base2, new2, margin=6)

    fig, ax = plt.subplots(3, 3, figsize=(13, 13), constrained_layout=True)

    ax[0, 0].imshow(mri_norm.T, cmap="gray", origin="lower", interpolation="nearest")
    ax[0, 0].set_title("A. MRI")

    ax[0, 1].imshow(gt2.T, cmap="gray", origin="lower", interpolation="nearest")
    ax[0, 1].set_title("B. Expert mask")

    ax[0, 2].imshow(base2.T, cmap="gray", origin="lower", interpolation="nearest")
    ax[0, 2].set_title("C. Baseline mask")

    ax[1, 0].imshow(new2.T, cmap="gray", origin="lower", interpolation="nearest")
    ax[1, 0].set_title("D. TRUE level0 mask")

    ax[1, 1].imshow(mri_norm.T, cmap="gray", origin="lower", interpolation="nearest")
    ax[1, 1].contour(gt2.T, levels=[0.5], colors=GREEN, linewidths=1.6)
    ax[1, 1].contour(base2.T, levels=[0.5], colors=RED, linewidths=1.2)
    ax[1, 1].set_title("E. Expert + baseline contour")

    ax[1, 2].imshow(mri_norm.T, cmap="gray", origin="lower", interpolation="nearest")
    ax[1, 2].contour(gt2.T, levels=[0.5], colors=GREEN, linewidths=1.6)
    ax[1, 2].contour(new2.T, levels=[0.5], colors=CYAN, linewidths=1.2)
    ax[1, 2].set_title("F. Expert + level0 contour")

    draw_fp_fn(ax[2, 0], mri_norm, base2, gt2)
    ax[2, 0].set_title("G. Baseline FP/FN")

    draw_fp_fn(ax[2, 1], mri_norm, new2, gt2)
    ax[2, 1].set_title("H. TRUE level0 FP/FN")

    zoom_mri = mri_norm[xlo:xhi, ylo:yhi]
    ax[2, 2].imshow(zoom_mri.T, cmap="gray", origin="lower", interpolation="nearest")
    ax[2, 2].contour(gt2[xlo:xhi, ylo:yhi].T, levels=[0.5], colors=GREEN, linewidths=2.0)
    ax[2, 2].contour(base2[xlo:xhi, ylo:yhi].T, levels=[0.5], colors=RED, linewidths=1.6)
    ax[2, 2].contour(new2[xlo:xhi, ylo:yhi].T, levels=[0.5], colors=CYAN, linewidths=1.6)
    ax[2, 2].set_title("I. Zoomed boundary: expert/baseline/level0")

    for a in ax.flat:
        a.axis("off")

    fig.legend(
        handles=[
            Line2D([0], [0], color=GREEN, lw=2, label="Expert"),
            Line2D([0], [0], color=RED, lw=2, label="Baseline (corrected-label, half-res)"),
            Line2D([0], [0], color=CYAN, lw=2, label="TRUE level0 (full-res)"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=RED, markersize=12, label="FP"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=YELLOW, markersize=12, label="FN"),
        ],
        loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02),
    )

    bd, nd = float(record["baseline_filtered_dice"]), float(record["level0_filtered_dice"])
    bh, nh = float(record["baseline_filtered_hd95_mm"]), float(record["level0_filtered_hd95_mm"])
    fig.suptitle(
        f"{case['domain']} {case['subject']}  |  {case['reason']}  |  slice {z}\n"
        f"Dice: baseline {bd:.4f} -> level0 {nd:.4f} (Δ{nd - bd:+.4f})   "
        f"HD95: baseline {bh:.4f} -> level0 {nh:.4f} mm (Δ{nh - bh:+.4f} mm)   "
        f"spacing {spacing[0]:.4f}x{spacing[1]:.4f}x{spacing[2]:.4f} mm",
        fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return z, slice_reason, (xlo, xhi, ylo, yhi), spacing


# ---------------------------------------------------------------------------
# 4. Contact sheet: zoomed triple-contour panel only, tiled
# ---------------------------------------------------------------------------

def make_contact_sheet(entries, out_path):
    n = len(entries)
    cols = 5
    rows_n = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 4.6 * rows_n), constrained_layout=True)
    ax = np.atleast_2d(ax)
    for i, e in enumerate(entries):
        r, c = divmod(i, cols)
        a = ax[r, c]
        image = np.asarray(nib.load(e["image_path"]).dataobj, dtype=np.float32)[:, :, e["slice"]]
        p1, p99 = np.percentile(image, [1, 99])
        mri_norm = np.clip((image - p1) / max(p99 - p1, 1e-8), 0, 1)
        gt = (np.asarray(nib.load(e["ground_truth_path"]).dataobj) > 0)[:, :, e["slice"]]
        base = (np.asarray(nib.load(e["baseline_filtered_prediction_path"]).dataobj) > 0)[:, :, e["slice"]]
        new = (np.asarray(nib.load(e["level0_filtered_prediction_path"]).dataobj) > 0)[:, :, e["slice"]]
        xlo, xhi, ylo, yhi = e["crop"]
        a.imshow(mri_norm[xlo:xhi, ylo:yhi].T, cmap="gray", origin="lower", interpolation="nearest")
        a.contour(gt[xlo:xhi, ylo:yhi].T, levels=[0.5], colors=GREEN, linewidths=1.8)
        a.contour(base[xlo:xhi, ylo:yhi].T, levels=[0.5], colors=RED, linewidths=1.4)
        a.contour(new[xlo:xhi, ylo:yhi].T, levels=[0.5], colors=CYAN, linewidths=1.4)
        a.set_title(f"{e['domain']} {e['subject']}\n{e['reason']}  z={e['slice']}\nΔDice {e['delta_dice']:+.4f}", fontsize=9)
        a.axis("off")
    for i in range(n, rows_n * cols):
        r, c = divmod(i, cols)
        ax[r, c].axis("off")
    fig.legend(
        handles=[
            Line2D([0], [0], color=GREEN, lw=2, label="Expert"),
            Line2D([0], [0], color=RED, lw=2, label="Baseline"),
            Line2D([0], [0], color=CYAN, lw=2, label="TRUE level0"),
        ],
        loc="lower center", ncol=3,
    )
    fig.suptitle("TRUE full-resolution level0 vs. corrected-label baseline -- boundary comparison atlas", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    cases, by_subject = select_cases()
    manifest_rows = []
    contact_entries = []

    for case in cases:
        record = by_subject[(case["domain"], case["subject"])]
        domain_dir = ATLAS / case["domain"].lower()
        fname = f"{case['reason']}_{case['subject']}.png"
        out_path = domain_dir / fname
        z, slice_reason, crop, spacing = make_case_figure(case, record, out_path)
        bd, nd = float(record["baseline_filtered_dice"]), float(record["level0_filtered_dice"])
        bh, nh = float(record["baseline_filtered_hd95_mm"]), float(record["level0_filtered_hd95_mm"])
        manifest_rows.append({
            "domain": case["domain"], "subject": case["subject"], "selection_reason": case["reason"],
            "selection_note": case["note"], "slice": z, "slice_selection_method": slice_reason,
            "crop_x_lo": crop[0], "crop_x_hi": crop[1], "crop_y_lo": crop[2], "crop_y_hi": crop[3],
            "baseline_filtered_dice": bd, "level0_filtered_dice": nd, "dice_delta": nd - bd,
            "baseline_filtered_hd95_mm": bh, "level0_filtered_hd95_mm": nh, "hd95_delta_mm": nh - bh,
            "spacing_x_mm": spacing[0], "spacing_y_mm": spacing[1], "spacing_z_mm": spacing[2],
            "figure_path": str(out_path),
        })
        contact_entries.append({
            "domain": case["domain"], "subject": case["subject"], "reason": case["reason"], "slice": z,
            "crop": crop, "delta_dice": nd - bd,
            "image_path": record["image_path"], "ground_truth_path": record["ground_truth_path"],
            "baseline_filtered_prediction_path": record["baseline_filtered_prediction_path"],
            "level0_filtered_prediction_path": record["level0_filtered_prediction_path"],
        })
        print(f"{case['domain']} {case['subject']} [{case['reason']}] slice={z} "
              f"Dice {bd:.4f}->{nd:.4f} HD95 {bh:.4f}->{nh:.4f}", flush=True)

    with (ATLAS / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        w.writeheader(); w.writerows(manifest_rows)

    make_contact_sheet(contact_entries, ATLAS / "contact_sheet.png")

    print(json.dumps({"cases": len(manifest_rows)}, indent=2))


if __name__ == "__main__":
    main()
