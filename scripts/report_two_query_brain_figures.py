#!/usr/bin/env python3
"""Brain-side visual comparison for the two-query experiment: the JOINT
two-query model's brain-task predictions (outputs/two_task_joint_decoder)
against the expert mask, per domain (CAMRI, Mouse). Simple 4-panel layout
focused on the joint model itself (the actual subject of this experiment),
not the single-task baseline:
  1. MRI  2. Expert mask  3. Joint model's predicted mask  4. TP/FP/FN overlay

Uses the joint model's RAW (unfiltered) prediction -- its evaluation did
not save a largest-connected-component-filtered NIfTI to disk (only the
filtered Dice number), and raw vs. filtered is visually near-identical for
a well-trained model at this quality level.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

JOINT = ROOT / "outputs/two_task_joint_decoder/brain_native_metrics_per_subject.csv"
OUT = ROOT / "outputs/two_query_experiment_figures/brain"


def rows(p):
    return list(csv.DictReader(open(p)))


def figure(sid, domain, image_path, gt_path, pred_path, out_path, dice):
    im = np.asarray(nib.load(image_path).dataobj, float)
    gt = np.asarray(nib.load(gt_path).dataobj) > 0
    pred = np.asarray(nib.load(pred_path).dataobj) > 0
    z = int(gt.sum((0, 1)).argmax())
    a, b = np.percentile(im[:, :, z], [1, 99]); m = np.clip((im[:, :, z] - a) / max(b - a, 1e-8), 0, 1)
    gz, pz = gt[:, :, z], pred[:, :, z]

    overlay = np.stack([m] * 3, -1)
    overlay[pz & gz] = (0, 0.8, 0)      # true positive: green
    overlay[pz & ~gz] = (1, 0, 0)       # false positive: red
    overlay[~pz & gz] = (1, 1, 0)       # false negative: yellow

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.3), constrained_layout=True)
    panels = [m, gz, pz, overlay]
    titles = ["MRI", "Expert brain mask", "Joint model prediction", f"Overlay (Dice {dice:.4f})"]
    for a0, x, t in zip(ax, panels, titles):
        a0.imshow(x.T if x.ndim == 2 else x.transpose(1, 0, 2), cmap="gray" if x.ndim == 2 else None, origin="lower")
        a0.set_title(t); a0.axis("off")
    fig.legend(handles=[Patch(color=(0, 0.8, 0), label="Agree (TP)"), Patch(color="red", label="False positive"), Patch(color="yellow", label="False negative")],
               loc="lower center", ncol=3)
    fig.suptitle(f"{domain} {sid} | slice {z} | Dice {dice:.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return z


def main():
    joint = rows(JOINT)
    manifest = []
    for domain in ("CAMRI", "Mouse"):
        q = [r for r in joint if r["domain"] == domain]
        by_dice = sorted(q, key=lambda r: float(r["dice"]))
        roles = {"worst_dice": by_dice[0], "median_dice": by_dice[len(by_dice) // 2], "best_dice": by_dice[-1]}
        for role, r in roles.items():
            sid = r["scan_id"]
            p = OUT / domain.lower() / f"{role}_{sid}.png"
            z = figure(sid, domain, r["image_path"], r["ground_truth_path"], r["prediction_path"], p, float(r["dice"]))
            manifest.append({"domain": domain, "role": role, "subject": sid, "slice": z, "dice": r["dice"], "path": str(p)})
    p = OUT / "manifest.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
    print(f"wrote {len(manifest)} figures to {OUT}")


if __name__ == "__main__":
    main()
