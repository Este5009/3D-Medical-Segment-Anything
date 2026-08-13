#!/usr/bin/env python3
"""Native paired evaluation of the corrected-label checkpoint."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from analyze_boundary_error_diagnostics import binary_metrics
from evaluate_mouse_boundary_adaptation import run_records
from models.query_mask_decoder import FrozenEncoderQueryModel,MultiScaleOneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter,RS2NetPaths
from train_query_decoder_overfit import choose_device,load_json

OUT=ROOT/"outputs/corrected_label_retraining";STRUCT=np.ones((3,3,3),bool)
def rows(p):return list(csv.DictReader(open(p)))
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def largest(mask):
    lab,n=ndimage.label(mask,STRUCT)
    return lab==(np.bincount(lab.ravel())[1:].argmax()+1) if n else np.zeros_like(mask)
def full_metrics(pred,gt,spacing):
    m=binary_metrics(pred,gt,spacing);tp=int((pred & gt).sum());fp=int((pred & ~gt).sum());fn=int((~pred & gt).sum())
    return {**m,"iou":tp/max(tp+fp+fn,1),"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"fp_fn_ratio":fp/max(fn,1),"total_residual_error":fp+fn}
def main():
    config=load_json(ROOT/"configs/corrected_label_retraining.yaml");split=load_json(ROOT/"outputs/mixed_domain_anatomical_training/split.json");cam={r["subject"]:r for r in rows(ROOT/config["camri_metrics"])};mouse={r["scan_id"]:r for r in rows(ROOT/config["mouse_metrics"])}
    records=[]
    for sid in split["camri"]["test"]:
        r=cam[sid];records.append({"domain":"CAMRI","subject":sid,"image_path":r["image_path"],"ground_truth_path":r["mask_path"]})
    for sid in split["mouse"]["test"]["scans"]:
        r=mouse[sid];records.append({"domain":"Mouse","subject":sid,"image_path":r["image_path"],"ground_truth_path":r["ground_truth_path"]})
    paths=RS2NetPaths.from_config(load_json(ROOT/config["encoder_config"]));device=choose_device();ck=torch.load(OUT/"checkpoints/best_corrected_labels.pt",map_location="cpu",weights_only=False);decoder=MultiScaleOneQueryMaskDecoder(32,4);decoder.load_state_dict(ck["decoder_state_dict"],strict=True);model=FrozenEncoderQueryModel(RS2NetEncoderAdapter(paths,image_size=tuple(config["tile_size"]),in_channels=1,out_channels=1,feature_size=48),decoder).to(device).eval()
    native=[]
    for domain in ("CAMRI","Mouse"):
        selected=[r for r in records if r["domain"]==domain]
        rs,_=run_records(model,selected,paths,config,OUT,device,domain.lower())
        native.extend(rs)
    old={ (r["domain"],r["subject"]):r for r in rows(ROOT/"outputs/filtered_residual_failure_analysis/per_subject_metrics.csv")}
    comparison=[];source=[]
    for r in records:
        sid=r["subject"];domain=r["domain"];raw_path=OUT/"native_predictions"/domain.lower()/f"{sid}_prediction.nii.gz";prob_path=OUT/"probability_maps"/domain.lower()/f"{sid}_probability.nii.gz";obj=nib.load(r["ground_truth_path"]);gt=np.asarray(obj.dataobj)>0;raw=np.asarray(nib.load(raw_path).dataobj)>0;filt=largest(raw);filtered_path=OUT/"filtered_predictions"/domain.lower()/f"{sid}_prediction.nii.gz";filtered_path.parent.mkdir(parents=True,exist_ok=True);nib.save(nib.Nifti1Image(filt.astype(np.uint8),obj.affine,obj.header),filtered_path);spacing=tuple(map(float,obj.header.get_zooms()[:3]));om=old[(domain,sid)]
        item={"domain":domain,"subject":sid,"image_path":r["image_path"],"ground_truth_path":r["ground_truth_path"],"old_raw_prediction_path":om["baseline_prediction_path"],"old_filtered_prediction_path":om["filtered_prediction_path"],"new_raw_prediction_path":str(raw_path),"new_filtered_prediction_path":str(filtered_path),"new_probability_path":str(prob_path)}
        for cond,pred in (("new_raw",raw),("new_filtered",filt)):
            item.update({f"{cond}_{k}":v for k,v in full_metrics(pred,gt,spacing).items()})
        for cond,path in (("old_raw",om["baseline_prediction_path"]),("old_filtered",om["filtered_prediction_path"])):
            item.update({f"{cond}_{k}":v for k,v in full_metrics(np.asarray(nib.load(path).dataobj)>0,gt,spacing).items()})
        comparison.append(item);source.append({"domain":domain,"subject":sid,"baseline_dice":item["new_raw_dice"],"filtered_dice":item["new_filtered_dice"],"dice_change":item["new_filtered_dice"]-item["new_raw_dice"],"baseline_iou":item["new_raw_iou"],"filtered_iou":item["new_filtered_iou"],"baseline_precision":item["new_raw_precision"],"filtered_precision":item["new_filtered_precision"],"baseline_recall":item["new_raw_recall"],"filtered_recall":item["new_filtered_recall"],"baseline_hd95_mm":item["new_raw_hd95_mm"],"filtered_hd95_mm":item["new_filtered_hd95_mm"],"baseline_total_error_voxels":item["new_raw_total_residual_error"],"filtered_total_error_voxels":item["new_filtered_total_residual_error"],"residual_voxels_removed":item["new_raw_total_residual_error"]-item["new_filtered_total_residual_error"],"image_path":r["image_path"],"ground_truth_path":r["ground_truth_path"],"baseline_prediction_path":str(raw_path),"filtered_prediction_path":str(filtered_path)})
    write(OUT/"per_subject_test_metrics.csv",comparison);write(OUT/"diagnostic_source.csv",source)
    aggregates=[]
    for domain in ("CAMRI","Mouse"):
      q=[x for x in comparison if x["domain"]==domain]
      for condition in ("old_raw","old_filtered","new_raw","new_filtered"):
       row={"domain":domain,"condition":condition,"subjects":len(q)}
       for k in ("dice","iou","precision","recall","hd95_mm","assd_mm","surface_dice_0.1mm","surface_dice_0.2mm","surface_dice_0.5mm","surface_dice_1mm","fp_voxels","fn_voxels","fp_fn_ratio","total_residual_error"):
        row[k]=float(np.mean([float(x[f"{condition}_{k}"]) for x in q])) if k not in ("fp_voxels","fn_voxels","total_residual_error") else sum(int(x[f"{condition}_{k}"]) for x in q)
       aggregates.append(row)
    write(OUT/"aggregate_test_metrics.csv",aggregates);print(json.dumps(aggregates,indent=2))
if __name__=="__main__":main()
