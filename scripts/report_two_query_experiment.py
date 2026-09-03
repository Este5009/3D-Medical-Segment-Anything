#!/usr/bin/env python3
"""Learning curves + visual comparison figures for the two-query
generalization experiment: lesion-only control, and the joint two-query
model on both its tasks. Run after both training runs and both evaluate
scripts have completed.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

LESION_OUT = ROOT / "outputs/stroke_lesion_only_decoder"
JOINT_OUT = ROOT / "outputs/two_task_joint_decoder"


def rows(p):
    return list(csv.DictReader(open(p)))


def lesion_only_curve():
    h = rows(LESION_OUT / "training/history.csv")
    summary = json.loads((LESION_OUT / "training/summary.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([int(x["epoch"]) for x in h], [float(x["validation_dice"]) for x in h], label="lesion validation", color="tab:red")
    ax.axvline(summary["selected_epoch"], color="black", ls="--", label=f"selected epoch {summary['selected_epoch']}")
    ax.set(title="Lesion-only decoder validation history\n(single query, no shared weights with brain task)",
           xlabel="Epoch", ylabel="Volumetric Dice")
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(LESION_OUT / "training/learning_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", LESION_OUT / "training/learning_curves.png")


def joint_curve():
    h = rows(JOINT_OUT / "training/history.csv")
    summary = json.loads((JOINT_OUT / "training/summary.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([int(x["epoch"]) for x in h], [float(x["camri_validation_dice"]) for x in h], label="CAMRI (brain)")
    ax.plot([int(x["epoch"]) for x in h], [float(x["mouse_validation_dice"]) for x in h], label="Mouse (brain)")
    ax.plot([int(x["epoch"]) for x in h], [float(x["lesion_validation_dice"]) for x in h], label="Stroke (lesion)", color="tab:red")
    ax.plot([int(x["epoch"]) for x in h], [float(x["balanced_validation_dice"]) for x in h], label="Balanced (3-way, selection score)", ls=":", color="black")
    ax.axvline(summary["selected_epoch"], color="black", ls="--", lw=1, label=f"selected epoch {summary['selected_epoch']}")
    ax.set(title="TwoTaskLevel0OneQueryMaskDecoder validation history\n(one shared trunk, two independent queries: query_brain, query_lesion)",
           xlabel="Epoch", ylabel="Volumetric Dice")
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(JOINT_OUT / "training/learning_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", JOINT_OUT / "training/learning_curves.png")


def lesion_figure(scan_id, image_path, gt_path, lesion_only_pred_path, joint_pred_path, out_path, dice_lo, dice_jt):
    im = np.asarray(nib.load(image_path).dataobj, float)
    gt = np.asarray(nib.load(gt_path).dataobj) > 0
    lo = np.asarray(nib.load(lesion_only_pred_path).dataobj) > 0
    jt = np.asarray(nib.load(joint_pred_path).dataobj) > 0
    if not gt.any():
        return None
    z_scores = gt.sum((0, 1))
    z = int(z_scores.argmax())
    a, b = np.percentile(im[:, :, z], [1, 99]); m = np.clip((im[:, :, z] - a) / max(b - a, 1e-8), 0, 1)

    def err(p):
        x = np.stack([m] * 3, -1); p = p[:, :, z]; x[p & ~gt[:, :, z]] = (1, 0, 0); x[~p & gt[:, :, z]] = (1, 1, 0); return x

    fig, ax = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = [m, gt[:, :, z], m, err(lo), lo[:, :, z], jt[:, :, z]]
    titles = ["T2 MRI", "Expert lesion mask", "(unused)", f"Lesion-only FP/FN (Dice {dice_lo:.3f})", "Lesion-only prediction", f"Joint two-query prediction"]
    for a0, x, t in zip(ax.flat, panels, titles):
        a0.imshow(x.T if x.ndim == 2 else x.transpose(1, 0, 2), cmap="gray" if x.ndim == 2 else None, origin="lower")
        a0.set_title(t); a0.axis("off")
    ax.flat[2].axis("off")
    ax.flat[5].contour(gt[:, :, z].T, colors="#24c96b", linewidths=1)
    ax.flat[5].contour(jt[:, :, z].T, colors="#00d5e7", linewidths=1)
    ax.flat[5].set_title(f"Joint prediction vs expert (Dice {dice_jt:.3f})")
    fig.suptitle(f"{scan_id} | slice {z}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170); plt.close(fig)
    return z


def lesion_figures():
    lo_rows = {r["scan_id"]: r for r in rows(LESION_OUT / "native_metrics_per_subject.csv")}
    jt_rows = {r["scan_id"]: r for r in rows(JOINT_OUT / "lesion_native_metrics_per_subject.csv")}
    common = sorted(set(lo_rows) & set(jt_rows))
    if not common:
        print("no common test subjects between lesion-only and joint eval; skipping figures")
        return
    by_joint_dice = sorted(common, key=lambda s: float(jt_rows[s]["dice"]))
    roles = {"worst_joint_dice": by_joint_dice[0], "median_joint_dice": by_joint_dice[len(by_joint_dice) // 2], "best_joint_dice": by_joint_dice[-1]}
    manifest = []
    for role, sid in roles.items():
        lo, jt = lo_rows[sid], jt_rows[sid]
        p = LESION_OUT.parent / "two_query_experiment_figures" / f"{role}_{sid}.png"
        z = lesion_figure(sid, jt["image_path"], jt["ground_truth_path"], lo["prediction_path"], jt["prediction_path"], p, float(lo["dice"]), float(jt["dice"]))
        if z is not None:
            manifest.append({"role": role, "scan_id": sid, "slice": z, "lesion_only_dice": lo["dice"], "joint_dice": jt["dice"], "path": str(p)})
    if manifest:
        p = LESION_OUT.parent / "two_query_experiment_figures/manifest.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
        print("wrote", len(manifest), "figures to", p.parent)


if __name__ == "__main__":
    lesion_only_curve()
    joint_curve()
    lesion_figures()
