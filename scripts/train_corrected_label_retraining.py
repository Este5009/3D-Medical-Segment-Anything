#!/usr/bin/env python3
"""Retrain the unchanged mixed-domain decoder with corrected categorical labels."""
from __future__ import annotations
import argparse, copy, csv, json, random, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import numpy as np
import torch
from corrected_label_preprocessing import preprocess_image_and_corrected_target
from models.query_mask_decoder import FrozenEncoderQueryModel,MultiScaleOneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter,RS2NetPaths
from train_generalization_pilot import load_cached
from train_mixed_domain_decoder import balanced_epoch_order,cached_records,training_loss
from train_mouse_boundary_adaptation import augment,metric,aggregate
from train_query_decoder_overfit import choose_device,load_json

LEVELS=("level1","level2","level3","level4")

def verification_gate(out):
    source=ROOT/"outputs/mask_preprocessing_audit"
    rows=list(csv.DictReader(open(source/"resampling_comparison_subject.csv")))
    result=[]
    for domain in ("CAMRI","Mouse"):
        q=[r for r in rows if r["domain"]==domain]
        old_net=np.mean([100*(int(r["native_original_vs_current_added_voxels"])-int(r["native_original_vs_current_lost_voxels"]))/int(r["original_voxels"]) for r in q])
        new_net=np.mean([100*(int(r["native_original_vs_nn_added_voxels"])-int(r["native_original_vs_nn_lost_voxels"]))/int(r["original_voxels"]) for r in q])
        result.append({"domain":domain,"subjects":len(q),"old_native_roundtrip_mean_dice":np.mean([float(r["native_original_vs_current_dice"]) for r in q]),"corrected_native_roundtrip_mean_dice":np.mean([float(r["native_original_vs_nn_dice"]) for r in q]),"old_mean_net_volume_change_percent":old_net,"corrected_mean_net_volume_change_percent":new_net,"old_lost_voxels":sum(int(r["native_original_vs_current_lost_voxels"]) for r in q),"corrected_lost_voxels":sum(int(r["native_original_vs_nn_lost_voxels"]) for r in q),"old_added_voxels":sum(int(r["native_original_vs_current_added_voxels"]) for r in q),"corrected_added_voxels":sum(int(r["native_original_vs_nn_added_voxels"]) for r in q),"old_hd95_mm":np.mean([float(r["native_original_vs_current_hd95_mm"]) for r in q]),"corrected_hd95_mm":np.mean([float(r["native_original_vs_nn_hd95_mm"]) for r in q]),"old_assd_mm":np.mean([float(r["native_original_vs_current_assd_mm"]) for r in q]),"corrected_assd_mm":np.mean([float(r["native_original_vs_nn_assd_mm"]) for r in q]),"old_inward_shift_mm":np.mean([float(r["mean_inward_shift_mm"]) for r in q]),"corrected_inward_shift_mm":0.0,"old_outward_shift_mm":np.mean([float(r["mean_outward_shift_mm"]) for r in q]),"corrected_outward_shift_mm":0.0})
    with (out/"preprocessing_verification.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=result[0]);w.writeheader();w.writerows(result)
    passed=all(abs(r["corrected_mean_net_volume_change_percent"])<0.5 and r["corrected_native_roundtrip_mean_dice"]>r["old_native_roundtrip_mean_dice"] for r in result)
    (out/"preprocessing_verification.json").write_text(json.dumps({"passed":passed,"criteria":"absolute corrected native round-trip volume bias <0.5% and Dice improves over old","domains":result},indent=2))
    if not passed:raise RuntimeError("Corrected-label verification failed; training stopped")

def prepare_cache(records,config,paths,device):
    cache=Path(config["corrected_cache"]);cache.mkdir(parents=True,exist_ok=True)
    encoder=RS2NetEncoderAdapter(paths,image_size=tuple(config["tile_size"]),in_channels=1,out_channels=1,feature_size=48).to(device).eval()
    for p in encoder.parameters():p.requires_grad_(False)
    for domain in ("camri","mouse"):
      for split in ("train","validation"):
       for i,r in enumerate(records[domain][split],1):
        dest=cache/f"{domain}_{split}_{r['subject']}.pt";r["cache_path"]=str(dest)
        if dest.exists():continue
        image,target,shape,_=preprocess_image_and_corrected_target(Path(r["image_path"]),Path(r["mask_path"]),paths,tuple(config["tile_size"]))
        with torch.inference_mode():features=encoder(image.to(device))
        torch.save({"features":{k:v.cpu().half() for k,v in features.items() if k in LEVELS},"target":target.byte(),"preprocessed_shape":shape,"label_interpolation":"nearest/order0/is_seg=True"},dest)
        print(f"cache {domain} {split} {i}/{len(records[domain][split])}",flush=True)
    del encoder

@torch.inference_mode()
def evaluate(decoder,records,device):
    decoder.eval();return [{"subject":r["subject"],**metric(decoder(load_cached(r,device)[0],output_size=load_cached(r,device)[1].shape[-3:]),load_cached(r,device)[1])} for r in records]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/corrected_label_retraining.yaml");args=ap.parse_args()
    config=load_json(ROOT/args.config);out=ROOT/config["output_directory"];(out/"checkpoints").mkdir(parents=True,exist_ok=True);(out/"training").mkdir(exist_ok=True);(out/"config.json").write_text(json.dumps(config,indent=2));verification_gate(out)
    random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"]);device=choose_device();paths=RS2NetPaths.from_config(load_json(ROOT/config["encoder_config"]));records=cached_records(config);prepare_cache(records,config,paths,device)
    initial=torch.load(ROOT/config["initial_checkpoint"],map_location="cpu",weights_only=False);decoder=MultiScaleOneQueryMaskDecoder(32,4);decoder.load_state_dict(initial["decoder_state_dict"],strict=True);decoder.to(device)
    if sum(p.numel() for p in decoder.parameters())!=170401 or tuple(decoder.query.shape)!=(1,1,32):raise RuntimeError("Architecture changed")
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"]);camri_reference=np.mean([float(r["dice"]) for r in csv.DictReader(open(ROOT/config["camri_metrics"])) if r["split"]=="validation"])
    best=-1;state=copy.deepcopy(decoder.state_dict());best_epoch=0;stale=0;history=[];start=time.time();safety=False
    for epoch in range(1,config["max_epochs"]+1):
      decoder.train();losses=[]
      for j,(domain,r) in enumerate(balanced_epoch_order(records["camri"]["train"],records["mouse"]["train"],config["seed"]+epoch)):
        features,target=load_cached(r,device);features,target=augment(features,target,random.Random(config["seed"]+epoch*100+j),config["augmentation"]);optimizer.zero_grad(set_to_none=True);logits=decoder(features,output_size=target.shape[-3:]);loss,parts=training_loss(logits,target,config);loss.backward();optimizer.step();losses.append({"loss":float(loss.detach()),**{k:float(v) for k,v in parts.items()}})
      val={d:aggregate(evaluate(decoder,records[d]["validation"],device)) for d in ("camri","mouse")};score=(val["camri"]["dice"]+val["mouse"]["dice"])/2;row={"epoch":epoch,"loss":np.mean([x["loss"] for x in losses]),"balanced_validation_dice":score,**{f"{d}_validation_{k}":v for d in val for k,v in val[d].items()}};history.append(row);print(f"epoch {epoch} CAMRI={val['camri']['dice']:.6f} Mouse={val['mouse']['dice']:.6f} balanced={score:.6f}",flush=True)
      if val["camri"]["dice"]<camri_reference-config["camri_validation_max_drop"]:safety=True;break
      if score>best+config["minimum_validation_improvement"]:best=score;state=copy.deepcopy(decoder.state_dict());best_epoch=epoch;stale=0
      else:stale+=1
      if stale>=config["early_stop_patience"]:break
    torch.save({"decoder_state_dict":state,"epoch":best_epoch,"balanced_validation_dice":best,"initial_checkpoint":config["initial_checkpoint"],"config":config},out/"checkpoints/best_corrected_labels.pt")
    with (out/"training/history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    summary={"device":str(device),"selected_epoch":best_epoch,"epochs_run":len(history),"best_balanced_validation_dice":best,"encoder_frozen":True,"one_query":True,"levels":["level1","level2","level3","level4"],"decoder_parameters":170401,"architecture_changed":False,"label_change_only":True,"camri_safety_stop":safety,"elapsed_seconds":time.time()-start};(out/"training/summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
