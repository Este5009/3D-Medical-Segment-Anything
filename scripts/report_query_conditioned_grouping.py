#!/usr/bin/env python3
"""Create the fixed post-hoc scientific report from already saved predictions."""
from __future__ import annotations
import csv,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt,label

OUT=ROOT/"outputs/query_conditioned_3d_grouping"
CONN=np.ones((3,3,3),np.uint8)

def rows(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def component_stats(pred,gt):
    lab,n=label(pred,CONN);sizes=np.bincount(lab.ravel());order=np.argsort(sizes[1:])[::-1]+1 if n else []
    detached=[i for i in order[1:] if not (gt&(lab==i)).any()]
    overlapping_removed=sum(int((gt&(lab==i)).sum()) for i in order[1:])
    return {"components":int(n),"detached_fp_count":len(detached),"detached_fp_voxels":sum(int((lab==i).sum()) for i in detached),"largest_detached_fp":max([int((lab==i).sum()) for i in detached]or[0]),"expert_voxels_outside_primary_component":overlapping_removed}
def boundary_error(pred,gt,spacing):
    surface=gt^binary_erosion(gt);dist=distance_transform_edt(~surface,sampling=spacing);error=pred^gt
    return int((error&(dist<=.5)).sum()),int((error&(dist>.5)).sum())
def regional_errors(pred,gt,spacing):
    surface=gt^binary_erosion(gt);dist=distance_transform_edt(~surface,sampling=spacing)
    lab,n=label(pred,CONN)
    primary=(lab==(np.bincount(lab.ravel())[1:].argmax()+1)) if n else np.zeros_like(pred)
    leakage=int((primary&~gt&(dist>.5)).sum())
    axis=int(np.argmin(gt.shape));projection=gt.any(tuple(i for i in range(3) if i!=axis));indices=np.where(projection)[0]
    terminal=np.zeros_like(gt)
    if len(indices):
        nterm=max(1,int(np.ceil(.2*len(indices))));chosen=np.r_[indices[:nterm],indices[-nterm:]]
        selector=[slice(None)]*3
        for index in chosen:selector[axis]=index;terminal[tuple(selector)]=True
    return leakage,int(((pred^gt)&terminal).sum())
def normalize(x):
    a,b=np.percentile(x,(1,99));return np.clip((x-a)/max(b-a,1e-8),0,1)
def exact_figure(r,destination):
    im=np.asarray(nib.load(r["image_path"]).dataobj,float);gt=np.asarray(nib.load(r["ground_truth_path"]).dataobj)>0
    base=np.asarray(nib.load(r["baseline_prediction_path"]).dataobj)>0;filt=np.asarray(nib.load(r["filter_prediction_path"]).dataobj)>0;learn=np.asarray(nib.load(r["learned_prediction_path"]).dataobj)>0
    brain=np.where(gt.any((0,1)))[0];zs=[int(brain[len(brain)//4]),int(brain[len(brain)//2]),int(brain[3*len(brain)//4])]
    fig,axes=plt.subplots(3,8,figsize=(20,7.2),constrained_layout=True)
    for i,z in enumerate(zs):
        m=normalize(im[:,:,z]); truth=gt[:,:,z]
        def overlay(pred):
            x=np.stack([m]*3,-1);x[pred&truth]=(0,.8,0);x[pred&~truth]=(1,0,0);x[~pred&truth]=(0,.35,1);return x
        def zoom(pred):
            x=overlay(pred);q=np.argwhere(gt[:,:,z]);lo=np.maximum(q.min(0)-5,0);hi=np.minimum(q.max(0)+6,gt.shape[:2]);return x[lo[0]:hi[0],lo[1]:hi[1]]
        panels=(m,truth,base[:,:,z],filt[:,:,z],learn[:,:,z],overlay(base[:,:,z]),overlay(learn[:,:,z]),zoom(learn[:,:,z]))
        for ax,x,title in zip(axes[i],panels,("MRI","Expert","Baseline","3D filter","Learned","Baseline FP/FN","Learned FP/FN","Boundary zoom")):
            ax.imshow(np.transpose(x,(1,0,2)) if x.ndim==3 else x.T,cmap=None if x.ndim==3 else "gray",origin="lower");ax.axis("off")
            if i==0:ax.set_title(title,fontsize=9)
        axes[i,0].set_ylabel(f"slice {z}",rotation=0,labelpad=28)
    fig.suptitle(f"{r['domain']} {r['subject']} | Dice {float(r['baseline_dice']):.4f} → {float(r['learned_dice']):.4f}")
    destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=170,bbox_inches="tight");plt.close(fig)

def main():
    comp=rows(OUT/"test_subject_comparison.csv");detail=[]
    for condition in ("baseline","filter","learned"):
        exported=[]
        for r in comp:
            exported.append({"domain":r["domain"],"subject":r["subject"],**{
                key:r[f"{condition}_{key}"] for key in
                ("dice","precision","recall","false_positives","false_negatives","volume_ratio","connected_components")
            }})
        filename={"baseline":"baseline_test_metrics.csv","filter":"deterministic_filter_test_metrics.csv","learned":"learned_grouping_test_metrics.csv"}[condition]
        write(OUT/filename,exported)
    for r in comp:
        gtobj=nib.load(r["ground_truth_path"]);gt=np.asarray(gtobj.dataobj)>0;spacing=gtobj.header.get_zooms()[:3]
        item={"domain":r["domain"],"subject":r["subject"]}
        for condition in ("baseline","filter","learned"):
            pred=np.asarray(nib.load(r[f"{condition}_prediction_path"]).dataobj)>0
            s=component_stats(pred,gt);boundary,nonboundary=boundary_error(pred,gt,spacing)
            leakage,terminal=regional_errors(pred,gt,spacing)
            item.update({f"{condition}_{k}":v for k,v in s.items()});item[f"{condition}_boundary_error_voxels"]=boundary;item[f"{condition}_nonboundary_error_voxels"]=nonboundary;item[f"{condition}_connected_leakage_voxels"]=leakage;item[f"{condition}_terminal_error_voxels"]=terminal
        detail.append(item)
    write(OUT/"component_comparison.csv",detail)
    # Exactly six cases by learned Dice rank: best, closest to median, absolute worst.
    manifest=[]
    for domain in ("CAMRI","Mouse"):
        q=sorted([r for r in comp if r["domain"]==domain],key=lambda r:float(r["learned_dice"]))
        median=float(np.median([float(r["learned_dice"]) for r in q]))
        chosen=(("worst",q[0]),("median",min(q,key=lambda r:abs(float(r["learned_dice"])-median))),("good",q[-1]))
        for category,r in chosen:
            path=OUT/"figures"/f"{domain}_{category}_{r['subject']}.png";exact_figure(r,path)
            manifest.append({"domain":domain,"category":category,"subject":r["subject"],"learned_dice":r["learned_dice"],"path":str(path)})
    write(OUT/"figures"/"manifest.csv",manifest)
    # A compact atlas of the three largest baseline detached-FP burdens.
    atlas=sorted(detail,key=lambda r:r["baseline_detached_fp_voxels"],reverse=True)[:3]
    amap=[]
    for item in atlas:
        r=next(x for x in comp if x["domain"]==item["domain"] and x["subject"]==item["subject"])
        path=OUT/"Grouping_Atlas"/f"{r['domain']}_{r['subject']}.png";exact_figure(r,path)
        amap.append({**item,"figure_path":str(path)})
    write(OUT/"Grouping_Atlas"/"manifest.csv",amap)
    # Validation curves and a machine-readable ablation table.
    ab=json.loads((OUT/"validation_ablation_summary.json").read_text());atable=[]
    fig,ax=plt.subplots(figsize=(8,5))
    for a in ab["ablations"]:
        h=rows(OUT/"ablations"/f"{a['name']}_history.csv");ax.plot([int(x["epoch"]) for x in h],[float(x["balanced_validation_dice"]) for x in h],marker="o",label=a["name"])
        atable.append({"variant":a["name"],"coordinates":a["options"]["use_coordinates"],"query_conditioning":a["options"]["use_query"],"parameters":a["parameter_count"],"best_epoch":a["best_epoch"],"balanced_validation_dice":a["balanced_validation_dice"]})
    ax.axhline(ab["ablations"][0]["baseline_balanced_validation_dice"],ls="--",color="black",label="baseline");ax.set(xlabel="Epoch",ylabel="Balanced validation Dice",title="Grouping validation ablations");ax.grid(alpha=.25);ax.legend(fontsize=7);fig.savefig(OUT/"figures"/"validation_learning_curves.png",dpi=180,bbox_inches="tight");plt.close(fig);write(OUT/"ablation_summary.csv",atable)
    shutil.copy2(ROOT/"outputs/mixed_domain_anatomical_training/split.json",OUT/"split.json")
    selected=json.loads((OUT/"validation_ablation_summary.json").read_text())["selected_ablation"]
    shutil.copy2(OUT/"ablations"/f"{selected}_history.csv",OUT/"training_history.csv")
    (OUT/"checkpoints").mkdir(exist_ok=True)
    shutil.copy2(OUT/"best_grouping_checkpoint.pt",OUT/"checkpoints"/"best_grouping_module.pt")
    summary=json.loads((OUT/"test_summary.json").read_text())
    changes={}
    for domain in ("CAMRI","Mouse"):
        q=[r for r in comp if r["domain"]==domain];delta=np.array([float(r["learned_dice"])-float(r["baseline_dice"]) for r in q])
        changes[domain]={"mean_dice_change":float(delta.mean()),"improved":int((delta>1e-4).sum()),"unchanged_within_0.0001":int((abs(delta)<=1e-4).sum()),"worsened":int((delta<-1e-4).sum())}
    det={}
    for condition in ("baseline","filter","learned"):
        det[condition]={"detached_fp_count":sum(int(r[f"{condition}_detached_fp_count"]) for r in detail),"detached_fp_voxels":sum(int(r[f"{condition}_detached_fp_voxels"]) for r in detail),"affected_subjects":sum(int(r[f"{condition}_detached_fp_count"])>0 for r in detail),"boundary_error_voxels":sum(int(r[f"{condition}_boundary_error_voxels"]) for r in detail),"connected_leakage_voxels":sum(int(r[f"{condition}_connected_leakage_voxels"]) for r in detail),"terminal_error_voxels":sum(int(r[f"{condition}_terminal_error_voxels"]) for r in detail),"expert_voxels_outside_primary_component":sum(int(r[f"{condition}_expert_voxels_outside_primary_component"]) for r in detail)}
    report=f"""# Query-conditioned 3D anatomical grouping experiment

## Controlled result

The frozen epoch-17 mixed-domain model was not changed. A separate {summary['parameter_count']:,}-parameter grouping module used frozen level-2 features, the initial mask logit/probability/uncertainty, normalized coordinates, and optional FiLM from the existing single query. Two depthwise-separable 3D blocks predicted a bounded residual logit correction. Encoder and baseline-decoder gradients remained disabled.

Validation selected `learned_coordinates_no_query` at epoch 1: balanced Dice 0.976714 versus baseline 0.976576 (+0.000138). The coordinate+query model reached 0.976706; query conditioning therefore contributed no measurable benefit. The no-coordinate result was 0.976683, so coordinates also added only 0.000031.

## Architecture

```text
frozen RS2 level2 [B,96,32,32,40] ── 1×1 projection ─┐
frozen decoder logits ── logit/probability/uncertainty ├─ concat + coordinates
existing single query ── optional FiLM ────────────────┘
               → two depthwise-separable 3D residual blocks
               → global context gate → bounded correction
               → frozen initial logits + correction → final mask
```

## Untouched tests

| Domain | Variant | Dice | Precision | Recall | HD95 mm | FP | FN | Volume ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for domain in ("CAMRI","Mouse"):
        for condition in ("baseline","deterministic_filter","learned"):
            m=summary["metrics"][domain][condition];report+=f"| {domain} | {condition} | {m['dice']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['hd95_mm']:.4f} | {m['false_positives']:.1f} | {m['false_negatives']:.1f} | {m['volume_ratio']:.6f} |\n"
    report+=f"""

Learned Dice changed by {changes['CAMRI']['mean_dice_change']:+.6f} on CAMRI and {changes['Mouse']['mean_dice_change']:+.6f} on Mouse. At tolerance 0.0001, CAMRI improved/unchanged/worsened = {changes['CAMRI']['improved']}/{changes['CAMRI']['unchanged_within_0.0001']}/{changes['CAMRI']['worsened']}; Mouse = {changes['Mouse']['improved']}/{changes['Mouse']['unchanged_within_0.0001']}/{changes['Mouse']['worsened']}. The absolute worst learned case was `{summary['absolute_worst_learned']}`.

## Grouping and boundary evidence

Across both test sets, baseline detached FP count/voxels/affected subjects were {det['baseline']['detached_fp_count']}/{det['baseline']['detached_fp_voxels']}/{det['baseline']['affected_subjects']}. The deterministic filter reduced these to {det['filter']['detached_fp_count']}/{det['filter']['detached_fp_voxels']}/{det['filter']['affected_subjects']}; learned grouping produced {det['learned']['detached_fp_count']}/{det['learned']['detached_fp_voxels']}/{det['learned']['affected_subjects']}. Boundary-error voxels were baseline {det['baseline']['boundary_error_voxels']:,}, filter {det['filter']['boundary_error_voxels']:,}, learned {det['learned']['boundary_error_voxels']:,}.

Connected leakage voxels were baseline/filter/learned = {det['baseline']['connected_leakage_voxels']:,}/{det['filter']['connected_leakage_voxels']:,}/{det['learned']['connected_leakage_voxels']:,}; terminal-region errors were {det['baseline']['terminal_error_voxels']:,}/{det['filter']['terminal_error_voxels']:,}/{det['learned']['terminal_error_voxels']:,}.

The deterministic filter improved mean Dice in both domains and reduced Mouse HD95 from 0.3435 to 0.2163 mm without changing recall. The learned residual instead reduced recall and mask volume, degraded Dice in both domains, did not reduce connected components, and slightly worsened Mouse HD95. Its apparent validation advantage did not reproduce on test. The fixed single query is constant for every case, so FiLM supplies no sample-varying object identity; the ablation correctly found no measurable query effect. Global context likewise failed to improve boundaries.

Safety checks show no slice-wise filtering: connectivity is fully 3D and diagonal connections are retained. The deterministic filter's `expert_voxels_outside_primary_component` column explicitly audits any expert-overlapping component removed. The learned model did not introduce a gross localization failure, but it systematically shrank correct anatomy (recall and volume ratio fell), including terminal/thin boundary regions visible in the six fixed figures. Test labels were used only after predictions were saved.

## Conclusion

**B. Deterministic 3D filtering captures most of the available benefit, so the learned grouping decoder is not currently justified.**

The learned module adds complexity without robust generalization; the simpler inference-only 26-connected-component rule gives the stronger Dice, topology, FP, and HD95 result. No further grouping-module revision is warranted on these benchmarks.
"""
    (OUT/"experiment_summary.md").write_text(report)
    (OUT/"scientific_summary.json").write_text(json.dumps({"changes":changes,"grouping":det,"recommendation":"B"},indent=2))
if __name__=="__main__":main()
