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
from matplotlib.patches import Patch
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


def lesion_figure(scan_id, image_path, gt_path, joint_pred_path, out_path, dice_jt):
    """4-panel layout focused on the JOINT model itself (the actual subject
    of this experiment): MRI, expert mask, joint prediction, TP/FP/FN
    overlay -- matching report_two_query_brain_figures.py's layout."""
    im = np.asarray(nib.load(image_path).dataobj, float)
    gt = np.asarray(nib.load(gt_path).dataobj) > 0
    jt = np.asarray(nib.load(joint_pred_path).dataobj) > 0
    if not gt.any():
        return None
    z = int(gt.sum((0, 1)).argmax())
    a, b = np.percentile(im[:, :, z], [1, 99]); m = np.clip((im[:, :, z] - a) / max(b - a, 1e-8), 0, 1)
    gz, jz = gt[:, :, z], jt[:, :, z]

    overlay = np.stack([m] * 3, -1)
    overlay[jz & gz] = (0, 0.8, 0)
    overlay[jz & ~gz] = (1, 0, 0)
    overlay[~jz & gz] = (1, 1, 0)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.3), constrained_layout=True)
    panels = [m, gz, jz, overlay]
    titles = ["T2 MRI", "Expert lesion mask", "Joint model prediction", f"Overlay (Dice {dice_jt:.4f})"]
    for a0, x, t in zip(ax, panels, titles):
        a0.imshow(x.T if x.ndim == 2 else x.transpose(1, 0, 2), cmap="gray" if x.ndim == 2 else None, origin="lower")
        a0.set_title(t); a0.axis("off")
    fig.legend(handles=[Patch(color=(0, 0.8, 0), label="Agree (TP)"), Patch(color="red", label="False positive"), Patch(color="yellow", label="False negative")],
               loc="lower center", ncol=3)
    fig.suptitle(f"{scan_id} | slice {z} | Dice {dice_jt:.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return z


def lesion_figures():
    jt_rows = {r["scan_id"]: r for r in rows(JOINT_OUT / "lesion_native_metrics_per_subject.csv")}
    if not jt_rows:
        print("no joint lesion eval rows found; skipping figures")
        return
    by_joint_dice = sorted(jt_rows, key=lambda s: float(jt_rows[s]["dice"]))
    roles = {"worst_dice": by_joint_dice[0], "median_dice": by_joint_dice[len(by_joint_dice) // 2], "best_dice": by_joint_dice[-1]}
    manifest = []
    for role, sid in roles.items():
        jt = jt_rows[sid]
        p = LESION_OUT.parent / "two_query_experiment_figures" / f"{role}_{sid}.png"
        z = lesion_figure(sid, jt["image_path"], jt["ground_truth_path"], jt["prediction_path"], p, float(jt["dice"]))
        if z is not None:
            manifest.append({"role": role, "scan_id": sid, "slice": z, "dice": jt["dice"], "path": str(p)})
    if manifest:
        p = LESION_OUT.parent / "two_query_experiment_figures/manifest.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
        print("wrote", len(manifest), "figures to", p.parent)


if __name__ == "__main__":
    lesion_only_curve()
    joint_curve()
    lesion_figures()
