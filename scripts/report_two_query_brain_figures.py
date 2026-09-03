#!/usr/bin/env python3
"""Brain-side visual comparison for the two-query experiment: single-task
brain decoder (outputs/paper_width_level0_decoder) vs the joint two-query
model's brain-task predictions (outputs/two_task_joint_decoder), per
domain (CAMRI, Mouse). Companion to report_two_query_experiment.py, which
only covered the lesion side.

Uses each model's RAW (unfiltered) prediction -- the joint model's
evaluation did not save a largest-connected-component-filtered NIfTI to
disk (only the filtered Dice number), and raw vs. filtered is visually
near-identical for a well-trained model at this quality level.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np

SINGLE_TASK = ROOT / "outputs/paper_width_level0_decoder/per_subject_comparison.csv"
JOINT = ROOT / "outputs/two_task_joint_decoder/brain_native_metrics_per_subject.csv"
OUT = ROOT / "outputs/two_query_experiment_figures/brain"


def rows(p):
    return list(csv.DictReader(open(p)))


def figure(sid, domain, image_path, gt_path, single_pred_path, joint_pred_path, out_path, dice_single, dice_joint):
    im = np.asarray(nib.load(image_path).dataobj, float)
    gt = np.asarray(nib.load(gt_path).dataobj) > 0
    sp = np.asarray(nib.load(single_pred_path).dataobj) > 0
    jp = np.asarray(nib.load(joint_pred_path).dataobj) > 0
    z = int(gt.sum((0, 1)).argmax())
    a, b = np.percentile(im[:, :, z], [1, 99]); m = np.clip((im[:, :, z] - a) / max(b - a, 1e-8), 0, 1)

    def err(p):
        x = np.stack([m] * 3, -1); pz = p[:, :, z]; x[pz & ~gt[:, :, z]] = (1, 0, 0); x[~pz & gt[:, :, z]] = (1, 1, 0); return x

    fig, ax = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = [m, gt[:, :, z], m, err(sp), sp[:, :, z], jp[:, :, z]]
    titles = ["MRI", "Expert brain mask", "(unused)", f"Single-task FP/FN (Dice {dice_single:.4f})", "Single-task prediction", "Joint two-query prediction"]
    for a0, x, t in zip(ax.flat, panels, titles):
        a0.imshow(x.T if x.ndim == 2 else x.transpose(1, 0, 2), cmap="gray" if x.ndim == 2 else None, origin="lower")
        a0.set_title(t); a0.axis("off")
    ax.flat[2].axis("off")
    ax.flat[5].contour(gt[:, :, z].T, colors="#24c96b", linewidths=1)
    ax.flat[5].contour(jp[:, :, z].T, colors="#00d5e7", linewidths=1)
    ax.flat[5].set_title(f"Joint vs expert (Dice {dice_joint:.4f})")
    fig.legend(handles=[Line2D([0], [0], color="#24c96b", label="Expert contour"), Line2D([0], [0], color="#00d5e7", label="Joint prediction contour")],
               loc="lower center", ncol=2)
    fig.suptitle(f"{domain} {sid} | slice {z} | single-task {dice_single:.4f} -> joint {dice_joint:.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170); plt.close(fig)
    return z


def main():
    single = {(r["domain"], r["subject"]): r for r in rows(SINGLE_TASK)}
    joint = {(r["domain"], r["scan_id"]): r for r in rows(JOINT)}
    manifest = []
    for domain in ("CAMRI", "Mouse"):
        common = sorted(sid for (d, sid) in joint if d == domain and (d, sid) in single)
        by_joint_dice = sorted(common, key=lambda sid: float(joint[(domain, sid)]["dice"]))
        roles = {"worst_joint_dice": by_joint_dice[0], "median_joint_dice": by_joint_dice[len(by_joint_dice) // 2], "best_joint_dice": by_joint_dice[-1]}
        for role, sid in roles.items():
            s, j = single[(domain, sid)], joint[(domain, sid)]
            p = OUT / domain.lower() / f"{role}_{sid}.png"
            z = figure(sid, domain, j["image_path"], j["ground_truth_path"],
                       s["paper_width_filtered_prediction_path"], j["prediction_path"], p,
                       float(s["paper_width_filtered_dice"]), float(j["dice"]))
            manifest.append({"domain": domain, "role": role, "subject": sid, "slice": z,
                              "single_task_dice": s["paper_width_filtered_dice"], "joint_dice": j["dice"], "path": str(p)})
    p = OUT / "manifest.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
    print(f"wrote {len(manifest)} figures to {OUT}")


if __name__ == "__main__":
    main()
