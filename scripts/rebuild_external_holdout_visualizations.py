#!/usr/bin/env python3
"""Rebuild external-holdout figures from saved artifacts only.

This module deliberately imports no model code and performs no inference.  It
reads the already-saved native MRI, expert mask, prediction, and CSV metrics,
reuses the strict geometry/metric audit, and renders a separate Matplotlib
visualization package without overwriting the earlier figures.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_external_holdout")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from visualize_external_holdout_results import (
    SOURCE, acquisition_group, audit_and_load, brain_crop, contour,
    informative_slices, recompute_binary_metrics, recompute_slices, robust_mri,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = SOURCE / "visualizations_fixed"
DPI = 180
COLORS = {
    "fn": "#2166e6", "fp": "#e52d2d", "overlap": "#23be50",
    "expert": "#00d9e8", "prediction": "#ed25ce",
    "twelve": "#0878b8", "sixty_four": "#d4512d",
}


def _masked(mask):
    """Return a masked array so false voxels remain fully transparent."""
    return np.ma.masked_where(~mask.astype(bool), mask.astype(float))


def _slice_arrays(record, index, crop):
    xs, ys = crop
    mri = robust_mri(record["image"][:, :, index], record["image"])[xs, ys]
    gt = record["target"][:, :, index][xs, ys]
    pred = record["prediction"][:, :, index][xs, ys]
    return mri, gt, pred


def _imshow_anatomy(ax, mri):
    ax.imshow(mri.T, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])


def _draw_panel(ax, kind, mri, gt, pred):
    """Render one comparison panel; masks use nearest-neighbor interpolation."""
    _imshow_anatomy(ax, mri)
    if kind == "expert":
        ax.imshow(_masked(gt.T), cmap=ListedColormap([COLORS["expert"]]), alpha=.34,
                  origin="lower", interpolation="nearest")
        ax.contour(gt.T, levels=[.5], colors=[COLORS["expert"]], linewidths=1.6)
    elif kind == "prediction":
        ax.imshow(_masked(pred.T), cmap=ListedColormap([COLORS["prediction"]]), alpha=.34,
                  origin="lower", interpolation="nearest")
        ax.contour(pred.T, levels=[.5], colors=[COLORS["prediction"]], linewidths=1.6)
    elif kind == "contours":
        if gt.any(): ax.contour(gt.T, levels=[.5], colors=[COLORS["expert"]], linewidths=1.8)
        if pred.any(): ax.contour(pred.T, levels=[.5], colors=[COLORS["prediction"]], linewidths=1.4)
    elif kind == "errors":
        label = np.zeros(gt.shape, dtype=np.uint8)
        label[gt & pred] = 1; label[pred & ~gt] = 2; label[gt & ~pred] = 3
        shown = np.ma.masked_where(label.T == 0, label.T)
        ax.imshow(shown, cmap=ListedColormap([COLORS["overlap"], COLORS["fp"], COLORS["fn"]]),
                  vmin=1, vmax=3, alpha=.68, origin="lower", interpolation="nearest")


LEGEND_HANDLES = [
    Patch(color=COLORS["overlap"], label="Overlap"), Patch(color=COLORS["fp"], label="False positive"),
    Patch(color=COLORS["fn"], label="False negative"), Line2D([0], [0], color=COLORS["expert"], lw=2, label="Expert contour"),
    Line2D([0], [0], color=COLORS["prediction"], lw=2, label="Prediction contour"),
]


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def subject_overview(record, path):
    selected = informative_slices(record)
    crop = brain_crop(record["target"].any(axis=2), margin=7)
    fig, axes = plt.subplots(6, 5, figsize=(16.5, 15.5), constrained_layout=True)
    columns = ["MRI", "Expert mask", "Decoder mask", "Expert vs prediction\ncontours", "Error map"]
    for col, title in enumerate(columns): axes[0, col].set_title(title, fontsize=14, fontweight="bold", pad=10)
    for row, index in enumerate(selected):
        mri, gt, pred = _slice_arrays(record, index, crop)
        for col, kind in enumerate(("mri", "expert", "prediction", "contours", "errors")):
            _draw_panel(axes[row, col], kind, mri, gt, pred)
        metric = record["slices"][index]
        axes[row, 0].set_ylabel(
            f"Slice {index}\nDice {metric['dice']:.4f}\nFP {metric['false_positives']} | FN {metric['false_negatives']}",
            fontsize=12, fontweight="bold", rotation=0, labelpad=67, va="center",
        )
    spacing = " × ".join(f"{x:.3g}" for x in record["spacing"])
    fig.suptitle(f"sub-{record['subject']} | Volumetric Dice = {record['dice']:.4f} | spacing = {spacing} mm | slices = {record['slice_count']}", fontsize=18, fontweight="bold")
    fig.legend(handles=LEGEND_HANDLES, loc="outside lower center", ncol=5, fontsize=12, frameon=True)
    save_figure(fig, path)


def hard_slice_figure(record, row, path):
    index = row["slice_index"]; crop = brain_crop(record["target"][:, :, index], margin=9)
    mri, gt, pred = _slice_arrays(record, index, crop)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.3), constrained_layout=True)
    titles = ["MRI", "Expert overlay", "Prediction overlay", "Contour comparison", "Error map"]
    for ax, title, kind in zip(axes, titles, ("mri", "expert", "prediction", "contours", "errors")):
        ax.set_title(title, fontsize=14, fontweight="bold"); _draw_panel(ax, kind, mri, gt, pred)
    spacing = " × ".join(f"{x:.3g}" for x in record["spacing"])
    fig.suptitle(
        f"sub-{record['subject']} | slice {index} | slice Dice {row['dice']:.4f} | volume Dice {record['dice']:.4f}\n"
        f"FP {row['false_positives']} | FN {row['false_negatives']} | spacing {spacing} mm | {acquisition_group(record)}",
        fontsize=16, fontweight="bold",
    )
    fig.legend(handles=LEGEND_HANDLES, loc="outside lower center", ncol=5, fontsize=11, frameon=True)
    save_figure(fig, path)


def per_slice_chart(record, path):
    rows = record["slices"]; x = np.arange(len(rows)); brain = [r["slice_index"] for r in rows if r["brain_voxels"] > 0]
    first, last = brain[0], brain[-1]; worst = min(brain, key=lambda i: rows[i]["dice"])
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11), sharex=True, constrained_layout=True)
    definitions = [
        ("Slice Dice", "Dice", [r["dice"] for r in rows], "#1268b3", (0, 1), True),
        ("Slice IoU", "IoU", [r["iou"] for r in rows], "#008d70", (0, 1), True),
        ("False-positive voxels", "FP voxels", [r["false_positives"] for r in rows], COLORS["fp"], None, False),
        ("False-negative voxels", "FN voxels", [r["false_negatives"] for r in rows], COLORS["fn"], None, False),
    ]
    empty = np.array([r["brain_voxels"] == 0 for r in rows])
    for ax, (title, ylabel, values, color, ylim, markers) in zip(axes, definitions):
        for index in np.where(empty)[0]: ax.axvspan(index-.5, index+.5, color="0.92", zorder=0)
        ax.plot(x, values, color=color, lw=1.8, marker="o" if markers else None, markersize=3.2, label=title)
        ax.axvline(first, color=COLORS["expert"], ls="--", lw=1.8, label=f"First expert slice ({first})")
        ax.axvline(last, color=COLORS["prediction"], ls="--", lw=1.8, label=f"Last expert slice ({last})")
        ax.plot(worst, values[worst], "s", color="red", ms=7, label=f"Worst non-empty ({worst})")
        ax.set_title(title, fontsize=14, fontweight="bold", loc="left"); ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(-.5, len(rows)-.5)
        if ylim: ax.set_ylim(*ylim)
        else: ax.set_ylim(bottom=0)
        ax.grid(True, color="0.84", lw=.7)
        ax.tick_params(labelsize=10); ax.legend(loc="best", fontsize=9, ncol=2)
    axes[0].annotate(f"Worst slice {worst}\nDice {rows[worst]['dice']:.4f}", xy=(worst, rows[worst]["dice"]), xytext=(max(0,worst-15), .18), arrowprops={"arrowstyle":"->","color":"red"}, fontsize=10, color="red", fontweight="bold")
    axes[-1].set_xlabel("Native slice index", fontsize=13, fontweight="bold")
    fig.suptitle(f"sub-{record['subject']} | Per-slice performance | Volumetric Dice = {record['dice']:.4f}", fontsize=18, fontweight="bold")
    save_figure(fig, path)


def select_hard(records):
    candidates = [(record, row) for record in records for row in record["slices"] if row["brain_voxels"] > 0]
    selected = {}
    groups = [sorted(candidates, key=lambda x:x[1]["dice"])[:20], sorted(candidates, key=lambda x:x[1]["false_negatives"], reverse=True)[:10], sorted(candidates, key=lambda x:x[1]["false_positives"], reverse=True)[:10]]
    for group in groups:
        for record, row in group: selected[(record["subject"], row["slice_index"])] = (record, row)
    return list(selected.values())


def contact_sheet(selected, path):
    chosen = sorted(selected, key=lambda x:(x[1]["dice"], -(x[1]["false_positives"]+x[1]["false_negatives"])))[:12]
    fig, axes = plt.subplots(12, 5, figsize=(14, 28), constrained_layout=True)
    titles = ["MRI", "Expert", "Prediction", "Contours", "Error map"]
    for col, title in enumerate(titles): axes[0,col].set_title(title, fontsize=14, fontweight="bold")
    for row_index, (record, row) in enumerate(chosen):
        crop=brain_crop(record["target"][:,:,row["slice_index"]],margin=8);mri,gt,pred=_slice_arrays(record,row["slice_index"],crop)
        for ax,kind in zip(axes[row_index],("mri","expert","prediction","contours","errors")): _draw_panel(ax,kind,mri,gt,pred)
        axes[row_index,0].set_ylabel(f"sub-{record['subject']} | slice {row['slice_index']}\nDice {row['dice']:.3f} | FP {row['false_positives']} | FN {row['false_negatives']}", fontsize=10, fontweight="bold", rotation=0, labelpad=78, va="center")
    fig.suptitle("Twelve most informative external-holdout hard slices", fontsize=19, fontweight="bold")
    fig.legend(handles=LEGEND_HANDLES, loc="outside lower center", ncol=5, fontsize=11)
    save_figure(fig,path)


def dataset_charts(records, directory, data_directory):
    rows=[]
    for r in records: rows.append({"subject":r["subject"],"dice":r["dice"],"precision":r["precision"],"recall":r["recall"],"slice_count":r["slice_count"],"brain_volume_mm3":r["brain_volume_mm3"],"group":"12-slice" if r["slice_count"]==12 else "64-slice"})
    directory.mkdir(parents=True,exist_ok=True);data_directory.mkdir(parents=True,exist_ok=True)
    with (data_directory/"dataset_metrics_plotted.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    mean=np.mean([r["dice"] for r in rows]);median=np.median([r["dice"] for r in rows]);best=max(rows,key=lambda r:r["dice"]);worst=min(rows,key=lambda r:r["dice"])
    made=[]
    # Sorted horizontal bars.
    ordered=sorted(rows,key=lambda r:r["dice"]);fig,ax=plt.subplots(figsize=(10,22),constrained_layout=True);colors=[COLORS["twelve"] if r["group"]=="12-slice" else COLORS["sixty_four"] for r in ordered]
    ax.barh([f"sub-{r['subject']}" for r in ordered],[r["dice"] for r in ordered],color=colors);ax.axvline(mean,color="red",ls="--",label=f"Mean {mean:.4f}");ax.axvline(median,color="purple",ls=":",label=f"Median {median:.4f}");ax.set_xlim(min(r["dice"] for r in rows)-.002,1);ax.set_xlabel("Volumetric Dice");ax.set_ylabel("Subject ID");ax.set_title(f"External-holdout Dice by subject (N={len(rows)})",fontweight="bold");ax.grid(axis="x",alpha=.35);ax.legend();ax.annotate(f"Worst: sub-{worst['subject']}",(worst["dice"],0),xytext=(8,5),textcoords="offset points");ax.annotate(f"Best: sub-{best['subject']}",(best["dice"],len(rows)-1),xytext=(-100,-12),textcoords="offset points");p=directory/"sorted_dice_by_subject.png";save_figure(fig,p);made.append(p)
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True);ax.hist([r["dice"] for r in rows],bins=16,color=COLORS["twelve"],edgecolor="white");ax.axvline(mean,color="red",ls="--",label=f"Mean {mean:.4f}");ax.axvline(median,color="purple",ls=":",label=f"Median {median:.4f}");ax.set(xlabel="Volumetric Dice",ylabel="Number of subjects",title=f"External-holdout Dice distribution (N={len(rows)})");ax.grid(axis="y",alpha=.3);ax.legend();p=directory/"dice_histogram.png";save_figure(fig,p);made.append(p)
    groups=[[r["dice"] for r in rows if r["group"]==name] for name in ("12-slice","64-slice")];fig,ax=plt.subplots(figsize=(8,6),constrained_layout=True);bp=ax.boxplot(groups,tick_labels=[f"12-slice\n(n={len(groups[0])})",f"64-slice\n(n={len(groups[1])})"],patch_artist=True);[patch.set_facecolor(c) for patch,c in zip(bp["boxes"],[COLORS["twelve"],COLORS["sixty_four"]])];ax.set(xlabel="Acquisition group",ylabel="Volumetric Dice",title="Dice by acquisition group");ax.grid(axis="y",alpha=.3);ax.legend(handles=[Patch(color=COLORS["twelve"],label="12-slice"),Patch(color=COLORS["sixty_four"],label="64-slice")]);p=directory/"dice_by_acquisition_group.png";save_figure(fig,p);made.append(p)
    specs=[("brain_volume_mm3","Expert brain volume (mm³)","dice_vs_brain_volume.png"),("slice_count","Native slice count","dice_vs_slice_count.png"),("recall","Recall","recall_vs_dice.png"),("precision","Precision","precision_vs_dice.png")]
    for key,xlabel,name in specs:
        fig,ax=plt.subplots(figsize=(8.5,6),constrained_layout=True)
        for group,color in (("12-slice",COLORS["twelve"]),("64-slice",COLORS["sixty_four"])):
            subset=[r for r in rows if r["group"]==group];ax.scatter([r[key] for r in subset],[r["dice"] for r in subset],s=42,color=color,label=f"{group} (n={len(subset)})",alpha=.9)
        corr=float(np.corrcoef([r[key] for r in rows],[r["dice"] for r in rows])[0,1]);ax.set_xlabel(xlabel,fontsize=12);ax.set_ylabel("Volumetric Dice",fontsize=12);ax.set_title(f"Dice vs {xlabel.lower()} | Pearson r = {corr:.3f}",fontsize=15,fontweight="bold");ax.tick_params(labelsize=10);ax.grid(alpha=.3);ax.legend(handles=[Line2D([0],[0],marker="o",linestyle="",color=COLORS["twelve"],label=f"12-slice group, n={sum(r['group']=='12-slice' for r in rows)}"),Line2D([0],[0],marker="o",linestyle="",color=COLORS["sixty_four"],label=f"64-slice group, n={sum(r['group']=='64-slice' for r in rows)}")],loc="lower left",fontsize=10,frameon=True);ax.annotate(f"Worst sub-{worst['subject']}",(worst[key],worst["dice"]),xytext=(8,8),textcoords="offset points");ax.annotate(f"Best sub-{best['subject']}",(best[key],best["dice"]),xytext=(8,-15),textcoords="offset points");p=directory/name;save_figure(fig,p);made.append(p)
    return made


def validate_images(paths):
    checks=[]
    for path in paths:
        image=Image.open(path).convert("RGB");a=np.asarray(image);nonwhite=float(np.mean(np.any(a<245,axis=2)));border=np.concatenate([a[:3].reshape(-1,3),a[-3:].reshape(-1,3),a[:,:3].reshape(-1,3),a[:,-3:].reshape(-1,3)]);border_ink=float(np.mean(np.any(border<220,axis=1)))
        ok=image.width>=900 and image.height>=600 and nonwhite>.025 and border_ink<.10 and float(a.std())>8
        try: displayed_path = str(path.relative_to(ROOT))
        except ValueError: displayed_path = str(path)
        checks.append({"path":displayed_path,"width":image.width,"height":image.height,"nonwhite_fraction":nonwhite,"border_ink_fraction":border_ink,"passed":ok})
    failed=[x for x in checks if not x["passed"]]
    if failed: raise RuntimeError("Figure quality validation failed:\n"+json.dumps(failed[:10],indent=2))
    return checks


def main():
    records,audit=audit_and_load();ranked=sorted(records,key=lambda r:r["dice"]);best=list(reversed(ranked[-5:]));median=ranked[len(ranked)//2-2:len(ranked)//2+3];worst=ranked[:10]
    paths=[]
    for record in best+median+worst:
        p=OUT/"subject_overviews"/f"sub-{record['subject']}_overview.png";subject_overview(record,p);paths.append(p)
    for record in records:
        p=OUT/"per_slice_charts"/f"sub-{record['subject']}_per_slice_metrics.png";per_slice_chart(record,p);paths.append(p)
    hard=select_hard(records)
    for record,row in hard:
        p=OUT/"hard_slices"/f"sub-{record['subject']}_slice-{row['slice_index']:03d}.png";hard_slice_figure(record,row,p);paths.append(p)
    contact=OUT/"hard_slices"/"hard_slice_contact_sheet_top12.png";contact_sheet(hard,contact);paths.append(contact)
    dataset=dataset_charts(records,OUT/"dataset_charts",OUT/"plotted_data");paths.extend(dataset)
    checks=validate_images(paths)
    # Required representative sample is explicitly recorded for subsequent visual inspection.
    sample=[OUT/"subject_overviews"/f"sub-{r['subject']}_overview.png" for r in (best[:2]+worst[:2])]
    sample += [OUT/"per_slice_charts"/f"sub-{r['subject']}_per_slice_metrics.png" for r in (best[0],worst[0],median[2])]
    sample += [OUT/"hard_slices"/f"sub-{r['subject']}_slice-{row['slice_index']:03d}.png" for r,row in sorted(hard,key=lambda x:x[1]["dice"])[:3]] + dataset
    summary={"figures_regenerated":len(paths),"subjects_validated":audit["subjects_validated"],"blank_figures":0,"clipped_labels_detected":0,"best_subjects":[r["subject"] for r in best],"median_subjects":[r["subject"] for r in median],"worst_subjects":[r["subject"] for r in worst],"hard_slices":len(hard),"representative_sample":[str(p.relative_to(ROOT)) for p in sample],"image_checks":checks}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"visualization_quality_checks.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("image_checks","representative_sample")},indent=2))


if __name__ == "__main__": main()
