#!/usr/bin/env python3
"""3D surface renders of the joint two-query model's segmentations, from
multiple viewing angles, plus the source MRI for comparison.

Brain and tumor renders necessarily come from DIFFERENT scans: no subject
in this project has both a real brain mask AND a real lesion mask (CAMRI/
Mouse have brain masks only; the stroke dataset has lesion masks only, no
brain masks). Each render uses a scan from the domain its query was
actually trained/evaluated on:
  - Brain: CAMRI subject 078 (query_brain), the joint model's own
    best-Dice CAMRI test case (0.9928).
  - Tumor: stroke subject 20191030CH_Exp8_M20 (query_lesion), the same
    median-Dice case (0.8753) already used in the 2D comparison figures.

NOT included: applying query_brain to a stroke image. That was tried and
checked directly -- the resulting mask has 99.8% IoU with the SAME
subject's query_lesion prediction (i.e. query_brain, applied out of the
only domain it was ever trained/evaluated on, just re-finds the lesion
instead of the whole brain). Rendering that would misleadingly imply a
working "brain outline" where none exists; it is a real, useful negative
finding about domain generalization, not a rendering worth publishing as
a segmentation result.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np
from skimage import measure
from scipy import ndimage

ROOT = Path("/Users/estebanfelixmoran/Desktop/Purdue/Medical_Imaging/3D-Medical-Segment-Anything")
OUT = ROOT / "outputs/two_query_experiment_figures/3d"
STRUCT = np.ones((3, 3, 3), bool)

BRAIN_IMAGE = "/Users/estebanfelixmoran/Desktop/Purdue/Medical_Imaging/Datasets/Image_Database/CAMRI Rat Brain MRI Data/sub-078/ses-1/anat/sub-078_ses-1_acq-RARE_T2w.nii.gz"
BRAIN_PRED = ROOT / "outputs/two_task_joint_decoder/native_predictions/brain_camri/078_prediction.nii.gz"
BRAIN_GT = None  # filled in from generalization_pilot/metrics.csv below

LESION_SID = "20191030CH_Exp8_M20"
LESION_IMAGE = f"/private/tmp/claude-501/-Users-estebanfelixmoran-Desktop-Purdue-Medical-Imaging-3D-Medical-Segment-Anything/b61a1f45-3d04-4f74-bf67-a7cfbabb173f/scratchpad/stroke_figure_subjects/{LESION_SID}/t2.nii"
LESION_PRED = ROOT / f"outputs/two_task_joint_decoder/native_predictions/lesion/{LESION_SID}_prediction.nii.gz"
LESION_GT = f"/private/tmp/claude-501/-Users-estebanfelixmoran-Desktop-Purdue-Medical-Imaging-3D-Medical-Segment-Anything/b61a1f45-3d04-4f74-bf67-a7cfbabb173f/scratchpad/stroke_figure_subjects/{LESION_SID}/masklesion_manual.nii"


def largest_component(mask):
    lab, n = ndimage.label(mask, STRUCT)
    if n == 0:
        return mask
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1)


def mesh_from_mask(mask, spacing):
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0.5, spacing=spacing)
    return verts, faces


def render_mesh_multi_angle(verts, faces, out_path, color, title, angles):
    fig = plt.figure(figsize=(4.2 * len(angles), 4.6))
    for i, (elev, azim, name) in enumerate(angles, 1):
        ax = fig.add_subplot(1, len(angles), i, projection="3d")
        mesh = Poly3DCollection(verts[faces], alpha=0.95, facecolor=color, edgecolor="none")
        ax.add_collection3d(mesh)
        ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
        ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
        ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, fontsize=11)
        ax.set_box_aspect((np.ptp(verts[:, 0]), np.ptp(verts[:, 1]), np.ptp(verts[:, 2])))
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close(fig)


def render_mri_slices(image_path, out_path, title, n=4):
    obj = nib.load(image_path)
    im = np.asarray(obj.dataobj, float)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4))
    zs = np.linspace(int(im.shape[2] * 0.2), int(im.shape[2] * 0.8), n).astype(int)
    for ax, z in zip(axes, zs):
        a, b = np.percentile(im[:, :, z], [1, 99])
        m = np.clip((im[:, :, z] - a) / max(b - a, 1e-8), 0, 1)
        ax.imshow(m.T, cmap="gray", origin="lower"); ax.set_title(f"slice {z}"); ax.axis("off")
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close(fig)


ANGLES = [(20, -60, "3/4 view"), (0, 0, "front"), (0, 90, "side"), (90, -90, "top")]


def main():
    import csv
    cam = {r["subject"]: r for r in csv.DictReader(open(ROOT / "outputs/generalization_pilot/metrics.csv"))}
    brain_gt_path = cam["078"]["mask_path"]

    # --- brain (CAMRI 078, query_brain, validated domain) ---
    obj = nib.load(BRAIN_PRED)
    spacing = tuple(map(float, obj.header.get_zooms()[:3]))
    pred = largest_component(np.asarray(obj.dataobj) > 0)
    gt = np.asarray(nib.load(brain_gt_path).dataobj) > 0
    v, f = mesh_from_mask(pred, spacing)
    render_mesh_multi_angle(v, f, OUT / "brain_3d_prediction.png", "#e0b090",
                             f"Joint model -- predicted brain surface (CAMRI 078, Dice 0.9928)", ANGLES)
    v, f = mesh_from_mask(gt, spacing)
    render_mesh_multi_angle(v, f, OUT / "brain_3d_expert.png", "#90c0e0",
                             "Expert brain surface (CAMRI 078)", ANGLES)
    render_mri_slices(BRAIN_IMAGE, OUT / "brain_mri_slices.png", "Source MRI -- CAMRI 078")

    # --- tumor (stroke subject, query_lesion, validated domain) ---
    obj = nib.load(LESION_PRED)
    spacing = tuple(map(float, obj.header.get_zooms()[:3]))
    pred = largest_component(np.asarray(obj.dataobj) > 0)
    gt = np.asarray(nib.load(LESION_GT).dataobj) > 0
    v, f = mesh_from_mask(pred, spacing)
    render_mesh_multi_angle(v, f, OUT / "tumor_3d_prediction.png", "#e04040",
                             f"Joint model -- predicted lesion surface ({LESION_SID}, Dice 0.8753)", ANGLES)
    v, f = mesh_from_mask(gt, spacing)
    render_mesh_multi_angle(v, f, OUT / "tumor_3d_expert.png", "#e0a040",
                             f"Expert lesion surface ({LESION_SID})", ANGLES)
    render_mri_slices(LESION_IMAGE, OUT / "tumor_mri_slices.png", f"Source MRI -- {LESION_SID}")

    print("wrote 3D renders to", OUT)


if __name__ == "__main__":
    main()
