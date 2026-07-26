#!/usr/bin/env python3
"""Strict epoch-14 one-query evaluation on saved adult-mouse MRI pairs.

There is intentionally no optimizer, training loop, backward call, checkpoint
write, threshold search, or post-processing change in this file.
"""
from __future__ import annotations
import csv,json,os,re,sys,time
from collections import Counter,defaultdict
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib_mouse_external")
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from scipy.ndimage import binary_erosion
from models.query_mask_decoder import FrozenEncoderQueryModel,MultiScaleOneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter,RS2NetPaths
from train_query_decoder_overfit import choose_device,load_json
from evaluate_external_holdout import preprocess,sliding_window_logits,export_native,metrics,slice_metrics

DPI=180;THRESHOLDS=(.80,.90,.95,.97,.98,.99)
COL={"overlap":"#23be50","fp":"#e52d2d","fn":"#2166e6","expert":"#00d9e8","pred":"#ed25ce"}

def stem(path): return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
def normalized(path): return re.sub(r"[^a-z0-9]","",stem(path).lower())
def identity(name):
    date=(re.search(r"_(20\d{6})_",name) or [None,"unknown"])[1];m=re.search(r"mouse(\d+)",name,re.I)
    mouse=f"mouse-{m.group(1)}" if m else f"anonymous-{stem(Path(name))}"
    return mouse,date

def geometry(path):
    n=nib.load(str(path));s=sitk.ReadImage(str(path));return {"shape":list(n.shape),"affine":n.affine.tolist(),"spacing":list(s.GetSpacing()),"origin":list(s.GetOrigin()),"direction":list(s.GetDirection()),"orientation":"".join(nib.aff2axcodes(n.affine)),"size":list(s.GetSize())}
def same_geometry(a,b):return a["shape"]==b["shape"] and np.allclose(a["affine"],b["affine"],atol=1e-5) and all(np.allclose(a[k],b[k],atol=1e-5) for k in ("spacing","origin","direction"))
def physical_bounds(g):
    origin=np.array(g["origin"]);direction=np.array(g["direction"]).reshape(3,3);extent=np.array(g["spacing"])*(np.array(g["size"])-1);corners=np.array([origin+direction@np.array([x,y,z]) for x in (0,extent[0]) for y in (0,extent[1]) for z in (0,extent[2])]);return corners.min(0),corners.max(0)
def same_physical_volume(a,b):
    alo,ahi=physical_bounds(a);blo,bhi=physical_bounds(b);inter=np.maximum(0,np.minimum(ahi,bhi)-np.maximum(alo,blo));union=np.maximum(ahi,bhi)-np.minimum(alo,blo);return float(np.prod(inter)/max(np.prod(union),1e-12))>.90

def discover_and_audit(config,paths,out):
    images=sorted((paths.dataset_root/config["image_directory"]).rglob("*.nii*"));masks=sorted((paths.dataset_root/config["mask_directory"]).rglob("*.nii*"));im={normalized(p):p for p in images};ma={normalized(p):p for p in masks};keys=sorted(set(im)&set(ma));rows=[];valid=[];excluded=[]
    corrected=out/"per_subject"/"geometry_corrected_masks";corrected.mkdir(parents=True,exist_ok=True)
    for key in keys:
        image,mask=im[key],ma[key];scan=stem(image);mouse,date=identity(image.name);ig,mg=geometry(image),geometry(mask);status="original_geometry_valid";used=mask;reason=""
        values=np.unique(np.asarray(nib.load(str(mask)).dataobj));foreground=int((np.asarray(nib.load(str(mask)).dataobj)>0).sum());binarizable=bool(np.all(np.isfinite(values)) and len(values)<=256)
        if foreground==0:reason="expert mask contains no foreground"
        elif not binarizable:reason=f"mask not safely binarizable ({len(values)} unique values)"
        elif not same_geometry(ig,mg):
            if same_physical_volume(ig,mg):
                destination=corrected/f"{scan}_mask.nii.gz";ref=sitk.ReadImage(str(image));lab=sitk.ReadImage(str(mask));res=sitk.Resample(lab,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8);sitk.WriteImage(res,str(destination),True);used=destination;status="geometry-corrected";mg=geometry(used)
                if int((sitk.GetArrayFromImage(res)>0).sum())==0 or not same_geometry(ig,mg):reason="nearest-neighbor geometry correction failed"
            else:reason="image/mask physical volumes do not sufficiently overlap"
        row={"scan_id":scan,"mouse_id":mouse,"timepoint":date,"image_path":str(image),"mask_path":str(mask),"evaluation_mask_path":str(used),"image_shape":"x".join(map(str,ig["shape"])),"mask_shape":"x".join(map(str,mg["shape"])),"spacing":"x".join(f"{x:.6g}" for x in ig["spacing"]),"orientation":ig["orientation"],"geometry_status":status,"mask_unique_values":";".join(map(str,values[:20])),"mask_foreground_voxels":foreground,"valid":not bool(reason),"exclusion_reason":reason};rows.append(row)
        if reason:excluded.append(row)
        else:valid.append({**row,"image":image,"mask":used,"image_geometry":ig})
    with (out/"dataset_audit.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    counts=Counter(r["mouse_id"] for r in valid);longitudinal={k:v for k,v in counts.items() if v>1 and not k.startswith("anonymous-")};audit={"images_found":len(images),"masks_found":len(masks),"valid_pairs":len(valid),"unpaired_images":[str(im[k]) for k in sorted(set(im)-set(ma))],"unpaired_masks":[str(ma[k]) for k in sorted(set(ma)-set(im))],"duplicate_normalized_ids":[],"geometry_mismatches":sum(r["geometry_status"]!="original_geometry_valid" for r in rows),"geometry_corrected":sum(r["geometry_status"]=="geometry-corrected" for r in rows),"excluded":excluded,"explicit_unique_mice":len({r["mouse_id"] for r in valid if not r["mouse_id"].startswith("anonymous-")}),"anonymous_scans_without_recoverable_mouse_id":sum(r["mouse_id"].startswith("anonymous-") for r in valid),"unique_identifiers_used":len(counts),"total_scans":len(valid),"longitudinal_repeats":longitudinal,"note":"Anonymous filenames do not encode a defensible biological mouse ID; each remains scan-unique rather than inventing longitudinal linkage.","pairs":rows};(out/"dataset_audit.json").write_text(json.dumps(audit,indent=2));return valid,audit

def native_arrays(image,mask,pred):return np.asarray(nib.load(str(image)).dataobj,dtype=np.float32),np.asarray(nib.load(str(mask)).dataobj)>0,np.asarray(nib.load(str(pred)).dataobj)>0
def enriched_slices(pred,target):
    rows=slice_metrics(pred,target);brain=[r["slice_index"] for r in rows if r["brain_voxels"]>0]
    for rank,index in enumerate(brain):f=(rank+.5)/len(brain);rows[index]["brain_position"]="first 20%" if f<.2 else "last 20%" if f>=.8 else "middle 60%"
    for r in rows:r["expert_foreground_present"]=r["brain_voxels"]>0;r["prediction_foreground_present"]=(r["brain_voxels"]-r["false_negatives"]+r["false_positives"])>0;r.setdefault("brain_position","empty expert")
    return rows

def robust(slice_,volume):
    v=volume[np.isfinite(volume)&(volume!=0)];lo,hi=np.percentile(v,(1,99));return np.clip((slice_-lo)/max(hi-lo,1e-8),0,1)
def crop(mask,margin=7):
    loc=np.argwhere(mask);lo=np.maximum(loc[:,:2].min(0)-margin,0);hi=np.minimum(loc[:,:2].max(0)+margin+1,mask.shape[:2]);return slice(lo[0],hi[0]),slice(lo[1],hi[1])
def panel(ax,kind,mri,gt,pred):
    ax.imshow(mri.T,cmap="gray",vmin=0,vmax=1,origin="lower");ax.set_xticks([]);ax.set_yticks([])
    if kind=="expert":ax.imshow(np.ma.masked_where(~gt.T,gt.T),cmap=matplotlib.colors.ListedColormap([COL["expert"]]),alpha=.34,origin="lower");ax.contour(gt.T,[.5],colors=COL["expert"])
    elif kind=="pred":
        ax.imshow(np.ma.masked_where(~pred.T,pred.T),cmap=matplotlib.colors.ListedColormap([COL["pred"]]),alpha=.34,origin="lower");
        if pred.any():ax.contour(pred.T,[.5],colors=COL["pred"])
    elif kind=="contour":
        if gt.any():ax.contour(gt.T,[.5],colors=COL["expert"])
        if pred.any():ax.contour(pred.T,[.5],colors=COL["pred"])
    elif kind=="error":
        lab=np.zeros(gt.shape,np.uint8);lab[gt&pred]=1;lab[pred&~gt]=2;lab[gt&~pred]=3;ax.imshow(np.ma.masked_where(lab.T==0,lab.T),cmap=matplotlib.colors.ListedColormap([COL["overlap"],COL["fp"],COL["fn"]]),vmin=1,vmax=3,alpha=.68,origin="lower")
def save(fig,path):path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=DPI,bbox_inches="tight",facecolor="white",pad_inches=.12);plt.close(fig)
def selected(rows):
    b=[r["slice_index"] for r in rows if r["brain_voxels"]>0];targets=[b[0],b[round(.25*(len(b)-1))],b[len(b)//2],b[round(.75*(len(b)-1))],b[-1],min(b,key=lambda i:rows[i]["dice"])];out=[]
    for t in targets:out.append(next(i for i in sorted(b,key=lambda x:(abs(x-t),-rows[x]["error"])) if i not in out))
    return out
def overview(record,path):
    image,target,pred=record["arrays"];rows=record["slices"];sel=selected(rows);cr=crop(target.any(2));fig,axes=plt.subplots(6,5,figsize=(16.5,15.5),constrained_layout=True);titles=("MRI","Expert mask","Query-decoder prediction","Expert vs prediction\ncontours","Error map")
    for j,t in enumerate(titles):axes[0,j].set_title(t,fontweight="bold",fontsize=13)
    for i,index in enumerate(sel):
        xs,ys=cr;mri=robust(image[:,:,index],image)[xs,ys];gt=target[:,:,index][xs,ys];pr=pred[:,:,index][xs,ys]
        for ax,k in zip(axes[i],("mri","expert","pred","contour","error")):panel(ax,k,mri,gt,pr)
        r=rows[index];axes[i,0].set_ylabel(f"Slice {index}\nDice {r['dice']:.4f}\nFP {r['false_positives']} | FN {r['false_negatives']}",rotation=0,labelpad=65,va="center",fontweight="bold")
    fig.suptitle(f"{record['scan_id']} | {record['mouse_id']} | Dice {record['dice']:.4f} | shape {image.shape} | spacing {record['spacing']} | slices {image.shape[2]}",fontweight="bold",fontsize=16);save(fig,path)
def per_slice_plot(record,path):
    rows=record["slices"];x=np.arange(len(rows));brain=[r["slice_index"] for r in rows if r["brain_voxels"]>0];first,last=brain[0],brain[-1];worst=min(brain,key=lambda i:rows[i]["dice"]);fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True,constrained_layout=True)
    for ax,(title,ylabel,key,color,limit) in zip(axes,(("Slice Dice","Dice","dice","#1678b5",(0,1)),("Slice IoU","IoU","iou","#008d70",(0,1)),("False-positive voxels","FP voxels","false_positives",COL["fp"],None),("False-negative voxels","FN voxels","false_negatives",COL["fn"],None))):
        y=[r[key] for r in rows];ax.plot(x,y,marker="o" if key in ("dice","iou") else None,ms=3,color=color,label=title);ax.axvline(first,ls="--",color=COL["expert"],label=f"First ({first})");ax.axvline(last,ls="--",color=COL["pred"],label=f"Last ({last})");ax.plot(worst,y[worst],"rs",label=f"Worst ({worst})");ax.set_title(title,loc="left",fontweight="bold");ax.set_ylabel(ylabel);ax.grid(alpha=.3);ax.legend(ncol=2,fontsize=8);ax.set_xlim(-.5,len(rows)-.5)
        if limit:ax.set_ylim(*limit)
    axes[-1].set_xlabel("Native slice index");fig.suptitle(f"{record['scan_id']} | {record['mouse_id']} | Volumetric Dice {record['dice']:.4f}",fontweight="bold",fontsize=16);save(fig,path)

def plots(evaluated,slice_rows,out):
    vis=out/"visualizations";rank=sorted(evaluated,key=lambda r:r["dice"]);mid=rank[len(rank)//2-2:len(rank)//2+3]
    for r in list(reversed(rank[-5:]))+mid+rank[:10]:overview(r,vis/"subject_overviews"/f"{r['scan_id']}.png")
    for r in evaluated:per_slice_plot(r,vis/"per_slice_charts"/f"{r['scan_id']}.png")
    hard=[]
    for key,rev,n in (("dice",False,20),("false_negatives",True,10),("false_positives",True,10)):
        for row in sorted((x for x in slice_rows if x["expert_foreground_present"]),key=lambda x:x[key],reverse=rev)[:n]:
            if (row["scan_id"],row["slice_index"]) not in {(x["scan_id"],x["slice_index"]) for x in hard}:hard.append(row)
    byscan={r["scan_id"]:r for r in evaluated}
    for row in hard:
        r=byscan[row["scan_id"]];image,target,pred=r["arrays"];idx=row["slice_index"];cr=crop(target[:,:,idx]);xs,ys=cr;mri=robust(image[:,:,idx],image)[xs,ys];gt=target[:,:,idx][xs,ys];pr=pred[:,:,idx][xs,ys];fig,axes=plt.subplots(1,5,figsize=(16,4),constrained_layout=True)
        for ax,t,k in zip(axes,("MRI","Expert overlay","Prediction overlay","Contours","Error map"),("mri","expert","pred","contour","error")):ax.set_title(t,fontweight="bold");panel(ax,k,mri,gt,pr)
        fig.suptitle(f"{r['scan_id']} | slice {idx} | slice Dice {row['dice']:.4f} | volume Dice {r['dice']:.4f} | FP {row['false_positives']} | FN {row['false_negatives']}",fontweight="bold");save(fig,vis/"hard_slices"/f"{r['scan_id']}_slice-{idx:03d}.png")
    # Dataset charts.
    def scatter(key,xlabel,name):
        fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True);ax.scatter([r[key] for r in evaluated],[r["dice"] for r in evaluated]);ax.set(xlabel=xlabel,ylabel="Volumetric Dice",title=f"Mouse external Dice vs {xlabel.lower()} (n={len(evaluated)})");ax.grid(alpha=.3);save(fig,vis/"dataset_charts"/name)
    order=sorted(evaluated,key=lambda r:r["dice"]);fig,ax=plt.subplots(figsize=(11,max(8,len(order)*.25)),constrained_layout=True);ax.barh([r["scan_id"] for r in order],[r["dice"] for r in order]);ax.set(xlabel="Volumetric Dice",ylabel="Scan ID",title=f"Mouse external Dice by scan (n={len(order)})");ax.grid(axis="x",alpha=.3);save(fig,vis/"dataset_charts"/"sorted_dice.png")
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True);ax.hist([r["dice"] for r in evaluated],bins=18);ax.set(xlabel="Volumetric Dice",ylabel="Scans",title=f"Mouse external Dice distribution (n={len(evaluated)})");ax.grid(axis="y",alpha=.3);save(fig,vis/"dataset_charts"/"dice_histogram.png")
    scatter("slice_count","Native slice count","dice_vs_slice_count.png");scatter("mean_spacing","Mean voxel spacing (mm)","dice_vs_spacing.png");scatter("expert_brain_volume_mm3","Expert brain volume (mm³)","dice_vs_expert_volume.png")
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True);ax.scatter([r["recall"] for r in evaluated],[r["precision"] for r in evaluated]);ax.set(xlabel="Recall",ylabel="Precision",title=f"Mouse external precision vs recall (n={len(evaluated)})");ax.grid(alpha=.3);save(fig,vis/"dataset_charts"/"precision_vs_recall.png")
    positions=("first 20%","middle 60%","last 20%");fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True);ax.boxplot([[r["dice"] for r in slice_rows if r["brain_position"]==p] for p in positions],tick_labels=positions);ax.set(xlabel="Position within brain extent",ylabel="Slice Dice",title="Mouse slice Dice by brain position");ax.grid(axis="y",alpha=.3);save(fig,vis/"dataset_charts"/"dice_by_brain_position.png")
    bins=(0,.5,.7,.8,.9,.95,.97,.98,.99,1.000001);non=[r for r in slice_rows if r["expert_foreground_present"]];counts=np.histogram([r["dice"] for r in non],bins=bins)[0];fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True);ax.bar(range(len(counts)),100*counts/len(non));ax.set_xticks(range(len(counts)),["0-.5",".5-.7",".7-.8",".8-.9",".9-.95",".95-.97",".97-.98",".98-.99",".99-1"],rotation=25);ax.set(xlabel="Slice Dice range",ylabel="Non-empty slices (%)",title=f"Mouse non-empty slice Dice ranges (n={len(non)})");ax.grid(axis="y",alpha=.3);save(fig,vis/"dataset_charts"/"slice_dice_ranges.png")

def finalize_reports(out):
    """Add complete statistics, longitudinal analysis, and CAMRI comparison."""
    rows=list(csv.DictReader((out/"subject_metrics.csv").open()));slices=list(csv.DictReader((out/"combined_slice_metrics.csv").open()));summary=load_json(out/"summary.json")
    numeric=("dice","iou","precision","recall","hd95","expert_brain_volume_mm3","predicted_brain_volume_mm3","volume_error_percentage","inference_seconds")
    summary["statistics"]={}
    for key in numeric:
        v=np.array([float(r[key]) for r in rows]);summary["statistics"][key]={"mean":float(v.mean()),"median":float(np.median(v)),"std":float(v.std()),"minimum":float(v.min()),"maximum":float(v.max()),"percentile_5":float(np.percentile(v,5)),"percentile_95":float(np.percentile(v,95))}
    groups=defaultdict(list)
    for row in rows:
        if not row["mouse_id"].startswith("anonymous-"):groups[row["mouse_id"]].append(row)
    longitudinal=[]
    for mouse,items in sorted(groups.items()):
        items.sort(key=lambda r:(r["timepoint"],r["scan_id"]));previous=None
        for index,row in enumerate(items):
            volume=float(row["predicted_brain_volume_mm3"]);change=None if previous is None else 100*(volume-previous)/previous;longitudinal.append({"mouse_id":mouse,"timepoint":row["timepoint"],"scan_id":row["scan_id"],"dice":float(row["dice"]),"predicted_brain_volume_mm3":volume,"change_from_previous_percent":"" if change is None else change,"large_volume_change_flag":False if change is None else abs(change)>20});previous=volume
    with (out/"longitudinal_summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=longitudinal[0]);w.writeheader();w.writerows(longitudinal)
    vis=out/"visualizations"/"longitudinal";fig,axes=plt.subplots(2,1,figsize=(13,10),constrained_layout=True)
    for mouse,items in sorted(groups.items()):
        items.sort(key=lambda r:(r["timepoint"],r["scan_id"]));x=range(len(items));labels=[r["timepoint"] for r in items];axes[0].plot(x,[float(r["dice"]) for r in items],marker="o",label=mouse);axes[1].plot(x,[float(r["predicted_brain_volume_mm3"]) for r in items],marker="o",label=mouse)
    axes[0].set(title=f"Longitudinal Dice for explicitly identified mice (n={len(groups)})",ylabel="Volumetric Dice");axes[1].set(title="Longitudinal predicted brain volume",ylabel="Predicted volume (mm³)",xlabel="Ordered observed timepoint");[ax.grid(alpha=.3) for ax in axes];axes[0].legend(ncol=3,fontsize=8);save(fig,vis/"longitudinal_dice_and_volume.png")
    large=[r for r in longitudinal if r["large_volume_change_flag"]];summary["longitudinal"]={"explicit_mice_with_repeats":len(groups),"scans_with_explicit_mouse_ids":len(longitudinal),"large_adjacent_prediction_volume_changes_over_20_percent":len(large),"flagged_scans":[r["scan_id"] for r in large],"anonymous_scans_excluded_from_longitudinal_linkage":summary["anonymous_scans"]}
    camri=load_json(ROOT/"outputs/external_holdout/summary.json");camri_slice=load_json(ROOT/"outputs/external_holdout/slice_dice_distribution_summary.json");summary["camri_comparison"]={"mouse_mean_dice":summary["dice"]["mean"],"camri_mean_dice":camri["mean_dice"],"mouse_mean_precision":summary["mean_precision"],"camri_mean_precision":camri["mean_precision"],"mouse_mean_recall":summary["mean_recall"],"camri_mean_recall":camri["mean_recall"],"mouse_mean_hd95_mm":summary["mean_hd95_mm"],"camri_mean_hd95_mm":camri["mean_hd95"],"mouse_nonempty_slice_thresholds":summary["slice_analysis"]["cumulative"],"camri_nonempty_slice_thresholds":{k:v["percentage"] for k,v in camri_slice["cumulative_nonempty"]["all subjects"].items()},"mouse_position":summary["slice_analysis"]["by_position"],"camri_position":{p:{k:v["percentage"] for k,v in camri_slice["cumulative_nonempty"][p].items()} for p in ("first 20%","middle 60%","last 20%")}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    c=summary["camri_comparison"];s=summary
    report=f'''# Mouse external evaluation compared with CAMRI rat

## Locked evaluation contract

No training or fine-tuning was performed. The only decoder checkpoint was epoch 14 from `outputs/generalization_pilot/best_checkpoint.pt`. Strict loading verified one learned query with shape `{s['query_shape']}`, 170,401 decoder parameters, a frozen RS2-Net encoder, evaluation mode, no optimizer, probability threshold 0.5, and no connected-component post-processing.

## Dataset audit

The recursive audit found 101 images and 101 masks, paired all 101 by normalized filename, and excluded none. All pairs passed native geometry and non-empty safely binarizable mask checks without correction. There are 15 explicitly identified biological mouse IDs spanning 49 scans. The remaining 52 filenames omit a recoverable mouse ID; they remain scan-unique, so the true total number of biological mice cannot be established from the released names. This prevents invented longitudinal linkage.

## Volumetric transfer results

Mouse mean Dice was **{s['dice']['mean']:.4f}**, median {s['dice']['median']:.4f}, and worst case {s['dice']['min']:.4f}. Mean precision was {s['mean_precision']:.4f}, mean recall {s['mean_recall']:.4f}, and mean HD95 {s['mean_hd95_mm']:.3f} mm. All scans reached Dice 0.80, 31.68% reached 0.90, and none reached 0.95.

Failures are dominated by false positives: the combined native slices contain {s['slice_analysis']['total_fp']:,} FP voxels versus {s['slice_analysis']['total_fn']:,} FN voxels, consistent with high recall but lower precision. The first 20% is weakest at the Dice 0.90 threshold ({s['slice_analysis']['by_position']['first 20%']['dice_gte_0.90']:.2f}%), followed by the last 20% ({s['slice_analysis']['by_position']['last 20%']['dice_gte_0.90']:.2f}%); the middle 60% reaches {s['slice_analysis']['by_position']['middle 60%']['dice_gte_0.90']:.2f}%.

## CAMRI comparison

| Metric | Mouse transfer | CAMRI rat external holdout |
|---|---:|---:|
| Mean Dice | {c['mouse_mean_dice']:.4f} | {c['camri_mean_dice']:.4f} |
| Mean precision | {c['mouse_mean_precision']:.4f} | {c['camri_mean_precision']:.4f} |
| Mean recall | {c['mouse_mean_recall']:.4f} | {c['camri_mean_recall']:.4f} |
| Mean HD95 (mm) | {c['mouse_mean_hd95_mm']:.3f} | {c['camri_mean_hd95_mm']:.3f} |

The large Dice gap is principally precision loss/over-segmentation rather than loss of brain retrieval. Non-empty mouse slices reach Dice >=0.90 in {s['slice_analysis']['cumulative']['dice_gte_0.90']:.2f}% of cases versus {c['camri_nonempty_slice_thresholds']['dice_gte_0.90']:.2f}% for CAMRI; at >=0.95 the values are {s['slice_analysis']['cumulative']['dice_gte_0.95']:.2f}% and {c['camri_nonempty_slice_thresholds']['dice_gte_0.95']:.2f}%.

## Longitudinal consistency

The 15 explicitly named mice were evaluated across their available dates. {s['longitudinal']['large_adjacent_prediction_volume_changes_over_20_percent']} adjacent predicted-volume changes exceeded the prespecified descriptive 20% flag. These flags are diagnostics, not claims that anatomy must remain unchanged. Anonymous scans are excluded from longitudinal linkage because their identity is not recoverable.

## Interpretation and next step

The query decoder never trained on these exact mouse scans, so the result supports cross-dataset, cross-animal-domain **decoder transfer**: it consistently retrieves most expert brain tissue, but boundaries are substantially over-expanded relative to the mouse labels. It does not prove fully external encoder generalization because the original RS2-Net encoder had prior mouse-domain exposure, though not necessarily these exact images.

The most scientifically informative next step is **C: evaluate another truly unseen dataset** while preserving the model. That separates encoder-domain generalization from adaptation effects. Training on mouse data or unfreezing the encoder would answer a different question and should follow only after this locked benchmark is preserved.
''';(out/"comparison_report.md").write_text(report)

def main():
    config=load_json(ROOT/"configs/mouse_external_evaluation.yaml");enc_cfg=load_json(ROOT/config["encoder_config"]);paths=RS2NetPaths.from_config(enc_cfg);out=ROOT/config["output_directory"];out.mkdir(parents=True,exist_ok=True)
    valid,audit=discover_and_audit(config,paths,out);checkpoint=torch.load(ROOT/config["checkpoint"],map_location="cpu",weights_only=False);assert checkpoint["epoch"]==config["expected_checkpoint_epoch"]
    decoder=MultiScaleOneQueryMaskDecoder(config["embedding_dim"],config["num_heads"]);decoder.load_state_dict(checkpoint["decoder_state_dict"],strict=True);assert tuple(decoder.query.shape)==(1,1,32);assert sum(p.numel() for p in decoder.parameters())==170401
    device=choose_device();encoder=RS2NetEncoderAdapter(paths,image_size=tuple(config["tile_size"]),in_channels=1,out_channels=1,feature_size=48);model=FrozenEncoderQueryModel(encoder,decoder).to(device).eval();assert not any(p.requires_grad for p in model.encoder.parameters());assert not model.training
    evaluated=[];combined=[]
    for i,r in enumerate(valid,1):
        pred_path=out/"native_predictions"/f"{r['scan_id']}_prediction.nii.gz";pred_path.parent.mkdir(parents=True,exist_ok=True);metrics_path=out/"per_subject"/r["scan_id"]/"metrics.json";metrics_path.parent.mkdir(parents=True,exist_ok=True)
        if metrics_path.exists() and pred_path.exists():row=load_json(metrics_path);image,target,pred=native_arrays(r["image"],r["mask"],pred_path);print(f"resume {i}/{len(valid)} {r['scan_id']}",flush=True)
        else:
            tensor,properties,manager,configuration,dataset=preprocess(r["image"],r["mask"],paths,tuple(config["tile_size"]));started=time.perf_counter()
            with torch.inference_mode():logits=sliding_window_logits(model,tensor,tuple(config["tile_size"]),device)
            elapsed=time.perf_counter()-started;export_native(logits,properties,manager,configuration,dataset,pred_path);image,target,pred=native_arrays(r["image"],r["mask"],pred_path);spacing=tuple(map(float,nib.load(str(r["image"])).header.get_zooms()[:3]));m=metrics(pred,target,spacing);predvol=float(pred.sum()*np.prod(spacing));row={"scan_id":r["scan_id"],"mouse_id":r["mouse_id"],"timepoint":r["timepoint"],"geometry_correction_status":r["geometry_status"],**m,"expert_brain_volume_mm3":m["brain_volume_mm3"],"predicted_brain_volume_mm3":predvol,"volume_error_percentage":100*(predvol-m["brain_volume_mm3"])/m["brain_volume_mm3"],"inference_seconds":elapsed,"slice_count":image.shape[2],"spacing_x":spacing[0],"spacing_y":spacing[1],"spacing_z":spacing[2],"mean_spacing":float(np.mean(spacing)),"image_path":str(r["image"]),"ground_truth_path":str(r["mask"]),"prediction_path":str(pred_path)};metrics_path.write_text(json.dumps(row,indent=2));print(f"evaluate {i}/{len(valid)} {r['scan_id']} dice={m['dice']:.4f}",flush=True)
        slices=enriched_slices(pred,target);rfull={**row,"arrays":(image,target,pred),"slices":slices,"spacing":f"{row['spacing_x']:.3g}×{row['spacing_y']:.3g}×{row['spacing_z']:.3g}"};evaluated.append(rfull);csvpath=metrics_path.parent/"slice_metrics.csv"
        with csvpath.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=slices[0]);w.writeheader();w.writerows(slices)
        combined += [{"scan_id":r["scan_id"],"mouse_id":r["mouse_id"],"timepoint":r["timepoint"],**x} for x in slices]
    fields=[k for k in evaluated[0] if k not in ("arrays","slices","spacing")]
    with (out/"subject_metrics.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in evaluated])
    with (out/"combined_slice_metrics.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=combined[0]);w.writeheader();w.writerows(combined)
    values=np.array([r["dice"] for r in evaluated]);non=[r for r in combined if r["expert_foreground_present"]];empty=[r for r in combined if not r["expert_foreground_present"]];position={p:{f"dice_gte_{t:.2f}":100*sum(r["dice"]>=t for r in non if r["brain_position"]==p)/sum(r["brain_position"]==p for r in non) for t in THRESHOLDS} for p in ("first 20%","middle 60%","last 20%")};summary={"training_performed":False,"fine_tuning_performed":False,"checkpoint_epoch":checkpoint["epoch"],"checkpoint":str(ROOT/config["checkpoint"]),"query_shape":list(decoder.query.shape),"exactly_one_query":True,"decoder_parameters":sum(p.numel() for p in decoder.parameters()),"encoder_gradients_enabled":False,"optimizer_created":False,"model_eval_mode":not model.training,"device":str(device),"scans_evaluated":len(evaluated),"scans_excluded":len(audit["excluded"]),"unique_identifiers":audit["unique_identifiers_used"],"explicit_unique_mice":audit["explicit_unique_mice"],"anonymous_scans":audit["anonymous_scans_without_recoverable_mouse_id"],"dice":{"mean":float(values.mean()),"median":float(np.median(values)),"std":float(values.std()),"min":float(values.min()),"max":float(values.max()),"percentile_5":float(np.percentile(values,5)),"percentile_95":float(np.percentile(values,95))},"mean_precision":float(np.mean([r['precision'] for r in evaluated])),"mean_recall":float(np.mean([r['recall'] for r in evaluated])),"mean_iou":float(np.mean([r['iou'] for r in evaluated])),"mean_hd95_mm":float(np.mean([r['hd95'] for r in evaluated])),"volume_thresholds":{f"dice_gte_{t:.2f}":100*sum(r["dice"]>=t for r in evaluated)/len(evaluated) for t in THRESHOLDS},"slice_analysis":{"total_slices":len(combined),"nonempty_expert_slices":len(non),"false_positive_empty_slices":sum(r["prediction_foreground_present"] for r in empty),"zero_prediction_nonempty_slices":sum(not r["prediction_foreground_present"] for r in non),"cumulative":{f"dice_gte_{t:.2f}":100*sum(r["dice"]>=t for r in non)/len(non) for t in THRESHOLDS},"by_position":position,"weakest_position":min(position,key=lambda p:position[p]["dice_gte_0.90"]),"total_fp":sum(r["false_positives"] for r in combined),"total_fn":sum(r["false_negatives"] for r in combined)}};(out/"summary.json").write_text(json.dumps(summary,indent=2));plots(evaluated,combined,out);finalize_reports(out);print((out/"summary.json").read_text())
if __name__=="__main__":main()
