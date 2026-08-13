#!/usr/bin/env python3
"""Evaluate the level0 model on the locked native cohort and filter identically."""
from __future__ import annotations
import csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from analyze_boundary_error_diagnostics import binary_metrics
from evaluate_external_holdout import preprocess,export_native
from evaluate_mouse_boundary_adaptation import export_probability
from models.query_mask_decoder import FrozenEncoderQueryModel,FullResolutionLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter,RS2NetPaths
from train_query_decoder_overfit import choose_device,load_json
OUT=ROOT/"outputs/full_resolution_level0_decoder";STRUCT=np.ones((3,3,3),bool)
LEVEL0_FEATURE_NAMES=("level0","level1","level2","level3","level4")

def sliding_window_level0_logits(model,image,tile_size,device):
 """Run the existing tiled inference scheme while retaining level0 features.

 The shared historical helper intentionally passes levels 1--4 because all
 earlier decoders ended at level1.  This local copy changes only that feature
 selection; tiling, overlap averaging, and padding remain identical.
 """
 spatial=image.shape[-3:];starts=[]
 for size,tile in zip(spatial,tile_size):
  if size<=tile:starts.append([0])
  else:
   stride=max(tile//2,1);axis=list(range(0,size-tile+1,stride))
   if axis[-1]!=size-tile:axis.append(size-tile)
   starts.append(axis)
 total=torch.zeros((1,1,*spatial),device=device);weight=torch.zeros_like(total)
 for x in starts[0]:
  for y in starts[1]:
   for z in starts[2]:
    tile=image[...,x:x+tile_size[0],y:y+tile_size[1],z:z+tile_size[2]].to(device)
    padding=[max(tile_size[i]-tile.shape[-3+i],0) for i in range(3)]
    if any(padding):tile=torch.nn.functional.pad(tile,(0,padding[2],0,padding[1],0,padding[0]))
    features=model.encode(tile)
    logits=model.decode({name:features[name] for name in LEVEL0_FEATURE_NAMES},tile_size)
    logits=logits[...,:tile_size[0]-padding[0] if padding[0] else None,:tile_size[1]-padding[1] if padding[1] else None,:tile_size[2]-padding[2] if padding[2] else None]
    total[...,x:x+logits.shape[-3],y:y+logits.shape[-2],z:z+logits.shape[-1]]+=logits
    weight[...,x:x+logits.shape[-3],y:y+logits.shape[-2],z:z+logits.shape[-1]]+=1
 return total/weight.clamp_min(1)
def rows(p):return list(csv.DictReader(open(p)))
def write(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def largest(x):
 lab,n=ndimage.label(x,STRUCT);return lab==(np.bincount(lab.ravel())[1:].argmax()+1) if n else np.zeros_like(x)
def metrics(pred,gt,spacing):
 m=binary_metrics(pred,gt,spacing);tp=int((pred & gt).sum());fp=int((pred & ~gt).sum());fn=int((~pred & gt).sum());return {**m,"iou":tp/max(tp+fp+fn,1),"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"fp_fn_ratio":fp/max(fn,1),"total_residual_error":fp+fn}
def main():
 config=load_json(ROOT/"configs/full_resolution_level0_decoder.yaml");split=load_json(ROOT/"outputs/mixed_domain_anatomical_training/split.json");cam={r["subject"]:r for r in rows(ROOT/config["camri_metrics"])};mouse={r["scan_id"]:r for r in rows(ROOT/config["mouse_metrics"])};records=[]
 for sid in split["camri"]["test"]:r=cam[sid];records.append({"domain":"CAMRI","subject":sid,"image_path":r["image_path"],"ground_truth_path":r["mask_path"]})
 for sid in split["mouse"]["test"]["scans"]:r=mouse[sid];records.append({"domain":"Mouse","subject":sid,"image_path":r["image_path"],"ground_truth_path":r["ground_truth_path"]})
 paths=RS2NetPaths.from_config(load_json(ROOT/config["encoder_config"]));device=choose_device();ck=torch.load(OUT/"checkpoints/best_level0_decoder.pt",map_location="cpu",weights_only=False);decoder=FullResolutionLevel0OneQueryMaskDecoder(32,4);decoder.load_state_dict(ck["decoder_state_dict"],strict=True);model=FrozenEncoderQueryModel(RS2NetEncoderAdapter(paths,image_size=tuple(config["tile_size"]),in_channels=1,out_channels=1,feature_size=48),decoder).to(device).eval();baseline={(r["domain"],r["subject"]):r for r in rows(ROOT/"outputs/corrected_label_retraining/per_subject_test_metrics.csv")};result=[];source=[]
 for i,r in enumerate(records,1):
  sid=r["subject"];domain=r["domain"].lower();rawp=OUT/"native_predictions"/domain/f"{sid}_prediction.nii.gz";probp=OUT/"probability_maps"/domain/f"{sid}_probability.nii.gz";rawp.parent.mkdir(parents=True,exist_ok=True);probp.parent.mkdir(parents=True,exist_ok=True);image,props,manager,configuration,dataset=preprocess(Path(r["image_path"]),Path(r["ground_truth_path"]),paths,tuple(config["tile_size"]));start=time.perf_counter()
  with torch.inference_mode():logits=sliding_window_level0_logits(model,image,tuple(config["tile_size"]),device)
  elapsed=time.perf_counter()-start;export_native(logits,props,manager,configuration,dataset,rawp);export_probability(logits,props,manager,configuration,dataset,probp,Path(r["image_path"]));obj=nib.load(r["ground_truth_path"]);gt=np.asarray(obj.dataobj)>0;raw=np.asarray(nib.load(rawp).dataobj)>0;filt=largest(raw);filtp=OUT/"filtered_predictions"/domain/f"{sid}_prediction.nii.gz";filtp.parent.mkdir(parents=True,exist_ok=True);nib.save(nib.Nifti1Image(filt.astype(np.uint8),obj.affine,obj.header),filtp);spacing=tuple(map(float,obj.header.get_zooms()[:3]));base=baseline[(r["domain"],sid)];item={"domain":r["domain"],"subject":sid,"image_path":r["image_path"],"ground_truth_path":r["ground_truth_path"],"baseline_raw_prediction_path":base["new_raw_prediction_path"],"baseline_filtered_prediction_path":base["new_filtered_prediction_path"],"baseline_probability_path":base["new_probability_path"],"level0_raw_prediction_path":str(rawp),"level0_filtered_prediction_path":str(filtp),"level0_probability_path":str(probp),"level0_inference_seconds":elapsed}
  for name,p in (("baseline_raw",base["new_raw_prediction_path"]),("baseline_filtered",base["new_filtered_prediction_path"]),("level0_raw",rawp),("level0_filtered",filtp)):item.update({f"{name}_{k}":v for k,v in metrics(np.asarray(nib.load(p).dataobj)>0,gt,spacing).items()})
  result.append(item);source.append({"domain":r["domain"],"subject":sid,"baseline_dice":item["level0_raw_dice"],"filtered_dice":item["level0_filtered_dice"],"dice_change":item["level0_filtered_dice"]-item["level0_raw_dice"],"image_path":r["image_path"],"ground_truth_path":r["ground_truth_path"],"baseline_prediction_path":str(rawp),"filtered_prediction_path":str(filtp)});print(f"{i}/86 {r['domain']} {sid} Dice={item['level0_filtered_dice']:.5f} seconds={elapsed:.2f}",flush=True)
 write(OUT/"per_subject_comparison.csv",result);write(OUT/"diagnostic_source.csv",source);aggregate=[]
 for d in ("CAMRI","Mouse"):
  q=[x for x in result if x["domain"]==d]
  for c in ("baseline_raw","baseline_filtered","level0_raw","level0_filtered"):
   row={"domain":d,"condition":c,"subjects":len(q)}
   for k in ("dice","iou","precision","recall","hd95_mm","assd_mm","surface_dice_0.1mm","surface_dice_0.2mm","surface_dice_0.5mm","surface_dice_1mm","fp_voxels","fn_voxels","fp_fn_ratio","total_residual_error"):
    row[k]=sum(int(x[f"{c}_{k}"]) for x in q) if k in ("fp_voxels","fn_voxels","total_residual_error") else float(np.mean([float(x[f"{c}_{k}"]) for x in q]))
   aggregate.append(row)
 write(OUT/"native_metrics.csv",aggregate);print(json.dumps(aggregate,indent=2))
if __name__=="__main__":main()
