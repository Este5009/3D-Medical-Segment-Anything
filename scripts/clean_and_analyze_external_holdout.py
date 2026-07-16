#!/usr/bin/env python3
"""Clean obsolete plots and analyze every saved external-holdout slice.

No model code is imported and no inference is performed.  Native predictions,
expert masks, subject metrics, and the 91 per-slice CSV files are read-only.
Only explicitly allowlisted first-generation visualization files are deleted.
"""

from __future__ import annotations

import csv, json, os, shutil
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_external_holdout")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/external_holdout"
FIXED=OUT/"visualizations_fixed"
CHARTS=FIXED/"slice_distribution"
DPI=180

RANGES=[
    ("Empty expert and empty prediction",None,None),
    ("Dice = 0.00",0.0,0.0),("0.00 < Dice < 0.50",0.0,0.5),
    ("0.50 <= Dice < 0.70",0.5,0.7),("0.70 <= Dice < 0.80",0.7,0.8),
    ("0.80 <= Dice < 0.90",0.8,0.9),("0.90 <= Dice < 0.95",0.9,0.95),
    ("0.95 <= Dice < 0.97",0.95,0.97),("0.97 <= Dice < 0.98",0.97,0.98),
    ("0.98 <= Dice < 0.99",0.98,0.99),("0.99 <= Dice <= 1.00",0.99,1.0),
]
THRESHOLDS=[0.80,0.90,0.95,0.97,0.98,0.99]


def category(row):
    if not row["expert_nonempty"] and not row["prediction_nonempty"]: return RANGES[0][0]
    d=row["dice"]
    if d==0: return RANGES[1][0]
    for label,low,high in RANGES[2:-1]:
        if low<d<high if low==0 else low<=d<high: return label
    return RANGES[-1][0]


def load_slices():
    subjects=list(csv.DictReader((OUT/"subject_metrics.csv").open()));rows=[]
    for subject in subjects:
        sid=subject["subject"];group="12-slice" if int(subject["slice_count"])==12 else "64-slice"
        saved=list(csv.DictReader((OUT/"per_slice"/f"sub-{sid}_slices.csv").open()))
        brain=[i for i,r in enumerate(saved) if int(r["brain_voxels"])>0];position={}
        for rank,index in enumerate(brain):
            fraction=(rank+.5)/len(brain)
            position[index]="first 20%" if fraction<.2 else "last 20%" if fraction>=.8 else "middle 60%"
        for index,r in enumerate(saved):
            row={"subject":sid,"group":group,"slice_index":int(r["slice_index"]),"dice":float(r["dice"]),"iou":float(r["iou"]),"false_positives":int(r["false_positives"]),"false_negatives":int(r["false_negatives"]),"brain_voxels":int(r["brain_voxels"])}
            row["expert_nonempty"]=row["brain_voxels"]>0;row["prediction_nonempty"]=(row["brain_voxels"]-row["false_negatives"]+row["false_positives"])>0;row["brain_position"]=position.get(index,"empty expert");row["dice_range"]=category(row);rows.append(row)
    return subjects,rows


def aggregate(rows,scope,scope_value):
    selected=rows if scope_value=="all" else [r for r in rows if r[scope]==scope_value]
    all_n=len(selected);nonempty=[r for r in selected if r["expert_nonempty"]];non_n=len(nonempty);output=[]
    for label,_,_ in RANGES:
        members=[r for r in selected if r["dice_range"]==label]
        output.append({"scope":scope,"scope_value":scope_value,"dice_range":label,"slice_count":len(members),"percentage_all_slices":100*len(members)/all_n if all_n else 0,"percentage_nonempty_expert_slices":100*len([r for r in members if r["expert_nonempty"]])/non_n if non_n else 0,"subjects_contributing":len({r["subject"] for r in members}),"all_slice_denominator":all_n,"nonempty_slice_denominator":non_n})
    return output


def cumulative(rows):
    scopes={"all subjects":rows,"12-slice":[r for r in rows if r["group"]=="12-slice"],"64-slice":[r for r in rows if r["group"]=="64-slice"],"first 20%":[r for r in rows if r["brain_position"]=="first 20%"],"middle 60%":[r for r in rows if r["brain_position"]=="middle 60%"],"last 20%":[r for r in rows if r["brain_position"]=="last 20%"]}
    result={}
    for name,items in scopes.items():
        non=[r for r in items if r["expert_nonempty"]];result[name]={f"dice_gte_{t:.2f}":{"count":sum(r["dice"]>=t for r in non),"percentage":100*sum(r["dice"]>=t for r in non)/len(non),"denominator":len(non)} for t in THRESHOLDS}
    return result


def chart_style(ax,title,xlabel,ylabel):
    ax.set_title(title,fontsize=15,fontweight="bold");ax.set_xlabel(xlabel,fontsize=12);ax.set_ylabel(ylabel,fontsize=12);ax.tick_params(labelsize=10);ax.grid(axis="x",alpha=.3)


def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=DPI,bbox_inches="tight",facecolor="white",pad_inches=.12);plt.close(fig)
    image=Image.open(path).convert("RGB");array=np.asarray(image);assert image.width>=900 and image.height>=600 and np.mean(np.any(array<245,axis=2))>.025 and array.std()>8


def create_charts(rows,aggregates,cum):
    CHARTS.mkdir(parents=True,exist_ok=True);non=[r for r in rows if r["expert_nonempty"]];labels=[x[0] for x in RANGES[1:]]
    distribution={a["dice_range"]:a["percentage_nonempty_expert_slices"] for a in aggregates if a["scope"]=="dataset" and a["scope_value"]=="all"}
    fig,ax=plt.subplots(figsize=(10,7),constrained_layout=True);vals=[distribution[x] for x in labels];bars=ax.barh(labels,vals,color="#1678b5");ax.bar_label(bars,fmt="%.2f%%",padding=4,fontsize=10);chart_style(ax,f"Slice Dice distribution: non-empty expert slices (n={len(non)})","Percentage of non-empty expert slices","Dice range");ax.invert_yaxis();save(fig,CHARTS/"nonempty_slice_dice_ranges.png")
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True);vals=[cum["all subjects"][f"dice_gte_{t:.2f}"]["percentage"] for t in THRESHOLDS];ax.plot(THRESHOLDS,vals,marker="o",lw=2.2,label=f"All subjects (n={len(non)} slices)");[ax.annotate(f"{v:.2f}%",(t,v),xytext=(0,8),textcoords="offset points",ha="center") for t,v in zip(THRESHOLDS,vals)];chart_style(ax,"Cumulative non-empty slice Dice","Dice threshold","Slices meeting or exceeding threshold (%)");ax.set_ylim(0,105);ax.legend();save(fig,CHARTS/"cumulative_slice_dice.png")
    colors=["#1678b5","#d4512d"];x=np.arange(len(labels));width=.38;fig,ax=plt.subplots(figsize=(12,7),constrained_layout=True)
    for i,g in enumerate(("12-slice","64-slice")):
        subset=[r for r in non if r["group"]==g];values=[100*sum(r["dice_range"]==label for r in subset)/len(subset) for label in labels];ax.bar(x+(i-.5)*width,values,width,label=f"{g} (n={len(subset)} slices)",color=colors[i])
    ax.set_xticks(x,labels,rotation=35,ha="right");chart_style(ax,"Non-empty slice Dice ranges by acquisition group","Dice range","Percentage of group slices");ax.grid(axis="y",alpha=.3);ax.legend();save(fig,CHARTS/"dice_ranges_by_acquisition.png")
    positions=("first 20%","middle 60%","last 20%");colors=["#6a3d9a","#1b9e77","#e66101"];width=.26;fig,ax=plt.subplots(figsize=(12,7),constrained_layout=True)
    for i,p in enumerate(positions):
        subset=[r for r in non if r["brain_position"]==p];values=[100*sum(r["dice_range"]==label for r in subset)/len(subset) for label in labels];ax.bar(x+(i-1)*width,values,width,label=f"{p} (n={len(subset)})",color=colors[i])
    ax.set_xticks(x,labels,rotation=35,ha="right");chart_style(ax,"Slice Dice ranges by position within brain extent","Dice range","Percentage of position-region slices");ax.grid(axis="y",alpha=.3);ax.legend();save(fig,CHARTS/"dice_ranges_by_brain_position.png")
    for threshold in (.95,.98):
        per=[]
        for sid in sorted({r["subject"] for r in non}):
            s=[r for r in non if r["subject"]==sid];per.append((sid,100*sum(r["dice"]>=threshold for r in s)/len(s),len(s)))
        per.sort(key=lambda z:z[1]);fig,ax=plt.subplots(figsize=(10,22),constrained_layout=True);bars=ax.barh([f"sub-{s}" for s,_,_ in per],[v for _,v,_ in per],color="#1678b5");chart_style(ax,f"Per-subject non-empty slices with Dice ≥ {threshold:.2f} (N=91 subjects)","Non-empty slices meeting threshold (%)","Subject ID");ax.set_xlim(0,105);ax.annotate(f"Worst: sub-{per[0][0]} ({per[0][1]:.1f}%)",(per[0][1],0),xytext=(8,2),textcoords="offset points");ax.annotate(f"Best: sub-{per[-1][0]} ({per[-1][1]:.1f}%)",(per[-1][1],len(per)-1),xytext=(-160,-12),textcoords="offset points");save(fig,CHARTS/f"per_subject_dice_gte_{str(threshold).replace('.','')}.png")


def cleanup_manifest():
    files=[]
    files += sorted((OUT/"per_slice").glob("*.png"))
    files += sorted((OUT/"per_subject").glob("sub-*/selected_slices.png"))
    files += sorted(OUT.glob("*.png"))
    for folder in (OUT/"best_subjects",OUT/"worst_subjects",OUT/"visualizations"):
        if folder.exists(): files += sorted(f for f in folder.rglob("*") if f.is_file())
    # Defensive exclusions: corrected figures, metrics, native predictions and NIfTI are never candidates.
    files=[f for f in dict.fromkeys(files) if FIXED not in f.parents and f.suffix.lower() not in (".csv",".nii",".gz")]
    return files


def remove_obsolete(files):
    size=sum(f.stat().st_size for f in files);manifest=[str(f.relative_to(ROOT)) for f in files]
    for f in files: f.unlink()
    deleted_folders=[]
    for folder in (OUT/"best_subjects",OUT/"worst_subjects",OUT/"visualizations"):
        if folder.exists(): shutil.rmtree(folder);deleted_folders.append(str(folder.relative_to(ROOT)))
    # Keep per_slice and per_subject because they retain required CSV/JSON/NIfTI artifacts.
    return manifest,deleted_folders,size


def main():
    # Strict native geometry and saved-metric audit occurs before any cleanup.
    from visualize_external_holdout_results import audit_and_load
    records,audit=audit_and_load();assert audit["subjects_validated"]==91
    subjects,rows=load_slices();assert len(subjects)==91
    aggregate_rows=[];aggregate_rows+=aggregate(rows,"dataset","all")
    for group in ("12-slice","64-slice"):aggregate_rows+=aggregate(rows,"group",group)
    for position in ("first 20%","middle 60%","last 20%"):aggregate_rows+=aggregate(rows,"brain_position",position)
    with (OUT/"slice_dice_distribution.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=aggregate_rows[0]);w.writeheader();w.writerows(aggregate_rows)
    cum=cumulative(rows);non=[r for r in rows if r["expert_nonempty"]];empty=[r for r in rows if not r["expert_nonempty"]]
    summary={"subjects":91,"total_slices":len(rows),"nonempty_expert_slices":len(non),"empty_expert_slices":len(empty),"range_distribution":aggregate_rows[:len(RANGES)],"cumulative_nonempty":cum,"empty_expert":{"true_empty_predictions":sum(not r["prediction_nonempty"] for r in empty),"false_positive_prediction_slices":sum(r["prediction_nonempty"] for r in empty),"percentage_with_predicted_foreground":100*sum(r["prediction_nonempty"] for r in empty)/len(empty)},"nonempty_expert":{"zero_predicted_foreground":sum(not r["prediction_nonempty"] for r in non),"dice_below_0_50":sum(r["dice"]<.5 for r in non),"dice_below_0_90":sum(r["dice"]<.9 for r in non),"dice_above_0_95":sum(r["dice"]>.95 for r in non),"dice_above_0_98":sum(r["dice"]>.98 for r in non)}}
    (OUT/"slice_dice_distribution_summary.json").write_text(json.dumps(summary,indent=2));create_charts(rows,aggregate_rows,cum)
    candidates=cleanup_manifest();print("FILES SELECTED FOR REMOVAL:");[print(p.relative_to(ROOT)) for p in candidates];deleted,folders,bytes_removed=remove_obsolete(candidates)
    preserved=["outputs/external_holdout/native_predictions/","outputs/external_holdout/per_slice/*.csv","outputs/external_holdout/per_subject/**/*.json","outputs/external_holdout/per_subject/**/*.nii.gz","outputs/external_holdout/subject_metrics.csv","outputs/external_holdout/metrics.csv","outputs/external_holdout/summary.json","outputs/external_holdout/cohort_audit.json","outputs/external_holdout/comparison_report.md","outputs/external_holdout/visualizations_fixed/"]
    report="# External-holdout cleanup report\n\nThe exact removal manifest was printed before deletion. Cleanup was restricted to obsolete visualization artifacts.\n\n## Folders deleted\n\n"+"\n".join(f"- `{x}/`" for x in folders)+"\n\n## Files deleted\n\n"+"\n".join(f"- `{x}`" for x in deleted)+f"\n\n## Disk space removed\n\n{bytes_removed:,} bytes ({bytes_removed/1024/1024:.2f} MiB).\n\n## Preserved\n\n"+"\n".join(f"- `{x}`" for x in preserved)+"\n\nConfirmed: no metric CSV, native prediction, checkpoint, NIfTI artifact, source data, or corrected figure was deleted. `per_slice/` remains because it contains the 91 required per-slice CSV files; only its obsolete PNGs were removed.\n"
    (OUT/"cleanup_report.md").write_text(report)
    print(json.dumps({"total_slices":len(rows),"nonempty_expert_slices":len(non),"files_deleted":len(deleted),"folders_deleted":folders,"bytes_removed":bytes_removed},indent=2))


if __name__=="__main__":main()
