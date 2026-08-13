#!/usr/bin/env python3
"""Assemble paired old/new boundary, contour, visual, and summary evidence."""
from __future__ import annotations
import csv,json,math,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from analyze_boundary_error_diagnostics import expert_surface,distance_to_surface,local_complexity,VOXEL_BINS
from diagnose_model_spatial_resolution import digital_contour_statistics,quantization_rows,effective_native_footprint

OUT=ROOT/"outputs/corrected_label_retraining";OLD_BOUND=ROOT/"outputs/boundary_error_diagnostics";OLD_GEOM=ROOT/"outputs/model_spatial_resolution_diagnostics"
def rows(p):return list(csv.DictReader(open(p)))
def write(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def geometry(condition,path_key,records):
 out=[];quant=[]
 for r in records:
  gt=np.asarray(nib.load(r["ground_truth_path"]).dataobj)>0;pred=np.asarray(nib.load(r[path_key]).dataobj)>0;spacing=tuple(map(float,nib.load(r["ground_truth_path"]).header.get_zooms()[:3]));per=defaultdict(list)
  for z in np.where(gt.any((0,1)))[0]:
   for kind,m in (("expert",gt[:,:,z]),("prediction",pred[:,:,z])):
    q=digital_contour_statistics(m)
    if q:
     for k,v in q.items():per[(kind,k)].append(v)
  item={"condition":condition,"domain":r["domain"],"subject":r["subject"]}
  for kind in ("expert","prediction"):
   for k in sorted({x[1] for x in per}):item[f"{kind}_{k}"]=float(np.mean(per[(kind,k)]))
  for k in sorted({x[1] for x in per}):item[f"prediction_expert_{k}_ratio"]=item[f"prediction_{k}"]/max(item[f"expert_{k}"],1e-12)
  out.append(item);foot,_=effective_native_footprint(spacing);quant+=quantization_rows(r["domain"],r["subject"],gt,pred,foot)
 for q in quant:q["condition"]=condition
 return out,quant
def boundary_tail(records):
 values=defaultdict(list);curv=defaultdict(int);opps=defaultdict(int)
 for r in records:
  gt=np.asarray(nib.load(r["ground_truth_path"]).dataobj)>0;spacing=tuple(map(float,nib.load(r["ground_truth_path"]).header.get_zooms()[:3]));surf=expert_surface(gt);dv,dm,nearest=distance_to_surface(surf,spacing);coords=np.argwhere(surf);lookup=np.full(gt.size,-1,np.int32);lookup[np.ravel_multi_index(coords.T,gt.shape)]=np.arange(len(coords));ids=lookup[np.ravel_multi_index(nearest.reshape(3,-1),gt.shape)].reshape(gt.shape);complexity=local_complexity(gt,spacing,surf);q1,q2=np.quantile(complexity,(1/3,2/3));groups=np.where(complexity<=q1,0,np.where(complexity<=q2,1,2))
  for name,path in (("old",r["old_filtered_prediction_path"]),("new",r["new_filtered_prediction_path"])):
   pred=np.asarray(nib.load(path).dataobj)>0;err=pred^gt;values[(name,r["domain"])].append(dv[err]);eid=ids[err]
   for gid,label in ((0,"flat"),(2,"high complexity")):curv[(name,r["domain"],label)]+=int((groups[eid]==gid).sum());opps[(r["domain"],label)]+=int((groups==gid).sum()) if name=="old" else 0
 result=[]
 for domain in ("CAMRI","Mouse","Combined"):
  domains=("CAMRI","Mouse") if domain=="Combined" else (domain,)
  for name in ("old","new"):
   v=np.concatenate(sum((values[(name,d)] for d in domains),[]));row={"domain":domain,"condition":name,"error_voxels":len(v),"distance_voxels_mean":v.mean(),"distance_voxels_p95":np.percentile(v,95)}
   for lo,hi,label in VOXEL_BINS:
    s=v>lo if math.isinf(hi) else v<=hi if lo==0 else (v>lo)&(v<=hi);row[label+"_percent"]=100*s.mean()
   for label in ("flat","high complexity"):
    c=sum(curv[(name,d,label)] for d in domains);o=sum(opps[(d,label)] for d in domains);row[label+"_errors"]=c;row[label+"_errors_per_100_surface"]=100*c/o
   result.append(row)
 return result
def figure(r,role,path):
 im=np.asarray(nib.load(r["image_path"]).dataobj,float);gt=np.asarray(nib.load(r["ground_truth_path"]).dataobj)>0;old=np.asarray(nib.load(r["old_filtered_prediction_path"]).dataobj)>0;new=np.asarray(nib.load(r["new_filtered_prediction_path"]).dataobj)>0;score=(old^gt).sum((0,1))+(new^gt).sum((0,1));z=int(score.argmax());q=np.argwhere(gt[:,:,z]);lo=np.maximum(q.min(0)-8,0);hi=np.minimum(q.max(0)+9,gt.shape[:2]);a,b=np.percentile(im[:,:,z],[1,99]);m=np.clip((im[:,:,z]-a)/max(b-a,1e-8),0,1)
 def error(p):
  p=p[:,:,z];x=np.stack([m]*3,-1);x[p&~gt[:,:,z]]=(1,0,0);x[~p&gt[:,:,z]]=(1,1,0);return x
 def contour(p):
  x=np.stack([m]*3,-1);return x
 fig,axes=plt.subplots(2,4,figsize=(15,8),constrained_layout=True);panels=[m,gt[:,:,z],old[:,:,z],new[:,:,z],error(old),error(new),m[lo[0]:hi[0],lo[1]:hi[1]],m[lo[0]:hi[0],lo[1]:hi[1]]];titles=("MRI","Untouched expert","Old filtered","Corrected-label filtered","Old FP/FN","New FP/FN","Old boundary zoom","New boundary zoom")
 for ax,x,t in zip(axes.flat,panels,titles):ax.imshow(x.transpose(1,0,2) if x.ndim==3 else x.T,cmap=None if x.ndim==3 else "gray",origin="lower",interpolation="nearest");ax.axis("off");ax.set_title(t)
 axes[1,2].contour(gt[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="#24c96b",linewidths=1);axes[1,2].contour(old[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="#00d5e7",linewidths=1);axes[1,3].contour(gt[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="#24c96b",linewidths=1);axes[1,3].contour(new[lo[0]:hi[0],lo[1]:hi[1],z].T,colors="#00d5e7",linewidths=1)
 fig.legend(handles=[Patch(color="#24c96b",label="Expert contour"),Patch(color="#00d5e7",label="Prediction contour"),Patch(color="red",label="FP"),Patch(color="yellow",label="FN")],loc="lower center",ncol=4);fig.suptitle(f"{r['domain']} {r['subject']} | {role} | slice {z} | filtered Dice {float(r['old_filtered_dice']):.4f}→{float(r['new_filtered_dice']):.4f}");path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=170);plt.close(fig)
def main():
 rec=rows(OUT/"per_subject_test_metrics.csv");geom=[];quant=[]
 for name,key in (("old","old_filtered_prediction_path"),("new","new_filtered_prediction_path")):
  g,q=geometry(name,key,rec);geom+=g;quant+=q
 write(OUT/"contour_geometry_comparison.csv",geom);write(OUT/"factor_quantization_comparison.csv",quant);tail=boundary_tail(rec);write(OUT/"boundary_comparison.csv",tail)
 oldcur=rows(OLD_BOUND/"curvature_analysis.csv");newcur=rows(OUT/"boundary_diagnostics/curvature_analysis.csv");cur=[]
 for domain in ("CAMRI","Mouse","Combined"):
  for cond,src in (("old",oldcur),("new",newcur)):
   for group in ("flat","high complexity"):
    r=next(x for x in src if x["domain"]==domain and x["error_type"]=="All" and x["complexity_group"]==group);cur.append({"domain":domain,"condition":cond,"complexity_group":group,"error_voxels":r["error_voxels"],"errors_per_100_surface_voxels":r["errors_per_100_surface_voxels"]})
 write(OUT/"curvature_error_comparison.csv",cur)
 ag=rows(OUT/"aggregate_test_metrics.csv");wide=[]
 for domain in ("CAMRI","Mouse"):
  for pred in ("raw","filtered"):
   o=next(x for x in ag if x["domain"]==domain and x["condition"]=="old_"+pred);n=next(x for x in ag if x["domain"]==domain and x["condition"]=="new_"+pred)
   for metric in ("dice","iou","precision","recall","hd95_mm","assd_mm","fp_voxels","fn_voxels","fp_fn_ratio","total_residual_error"):
    wide.append({"domain":domain,"prediction_type":pred,"metric":metric,"old":o[metric],"new":n[metric],"change":float(n[metric])-float(o[metric])})
 write(OUT/"old_vs_new_metrics.csv",wide)
 figs=OUT/"figures";manifest=[]
 for domain in ("CAMRI","Mouse"):
  q=[r for r in rec if r["domain"]==domain];newg=[x for x in geom if x["domain"]==domain and x["condition"]=="new"];pixel=max(newg,key=lambda x:x["prediction_expert_axis_run_max_ratio"])["subject"];curved="064" if domain=="CAMRI" else "POLYIC_20190510_mouse43__E9_P1";roles={"largest_improvement":max(q,key=lambda x:float(x["new_filtered_dice"])-float(x["old_filtered_dice"])),"little_no_improvement":min(q,key=lambda x:abs(float(x["new_filtered_dice"])-float(x["old_filtered_dice"]))),"previous_worst_boundary":min(q,key=lambda x:float(x["old_filtered_dice"])),"representative":sorted(q,key=lambda x:float(x["new_filtered_dice"])-float(x["old_filtered_dice"]))[len(q)//2],"high_curvature":next(x for x in q if x["subject"]==curved),"visibly_pixelated":next(x for x in q if x["subject"]==pixel)}
  for role,r in roles.items():p=figs/domain.lower()/f"{role}_{r['subject']}.png";figure(r,role,p);manifest.append({"domain":domain,"role":role,"subject":r["subject"],"path":str(p)})
 write(figs/"manifest.csv",manifest)
 # Aggregate contour and factor-5 evidence.
 summary={"selected_epoch":17,"checkpoint":str((OUT/"checkpoints/best_corrected_labels.pt").resolve()),"subjects":{"CAMRI":6,"Mouse":80},"architecture_unchanged":True,"decoder_parameters":170401,"boundary":tail,"visual_manifest":manifest}
 for domain in ("CAMRI","Mouse"):
  summary[domain]={}
  for cond in ("old","new"):
   q=[r for r in geom if r["domain"]==domain and r["condition"]==cond];f=[r for r in quant if r["domain"]==domain and r["condition"]==cond and int(r["candidate_factor_pixels"])==5 and r["native_coordinate_axis"]=="y"]
   summary[domain][cond]={"direction_changes_ratio":float(np.mean([r["prediction_expert_direction_changes_per_100_steps_ratio"] for r in q])),"p95_axis_run_ratio":float(np.mean([r["prediction_expert_axis_run_p95_ratio"] for r in q])),"max_axis_run_ratio":float(np.mean([r["prediction_expert_axis_run_max_ratio"] for r in q])),"factor5_alignment_excess":float(np.mean([float(r["prediction_minus_expert_alignment"]) for r in f]))}
 (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
 hist=rows(OUT/"training/history.csv");fig,ax=plt.subplots(figsize=(8,5));ax.plot([int(x["epoch"]) for x in hist],[float(x["camri_validation_dice"]) for x in hist],label="CAMRI validation");ax.plot([int(x["epoch"]) for x in hist],[float(x["mouse_validation_dice"]) for x in hist],label="Mouse validation");ax.axvline(17,color="black",ls="--",label="selected epoch 17");ax.set(title="Corrected-label validation history",xlabel="Epoch",ylabel="Volumetric Dice",ylim=(0.9,1));ax.grid(alpha=.25);ax.legend();fig.savefig(OUT/"training/learning_curves.png",dpi=200,bbox_inches="tight");plt.close(fig)
if __name__=="__main__":main()
