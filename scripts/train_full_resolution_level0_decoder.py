#!/usr/bin/env python3
"""Controlled corrected-label training with the sole addition of level0 decoding."""
from __future__ import annotations
import argparse,copy,csv,json,random,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import numpy as np
import torch
from corrected_label_preprocessing import preprocess_image_and_corrected_target
from models.query_mask_decoder import FullResolutionLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter,RS2NetPaths
from train_generalization_pilot import load_cached
from train_mixed_domain_decoder import balanced_epoch_order,cached_records,training_loss
from train_mouse_boundary_adaptation import augment,metric,aggregate
from train_query_decoder_overfit import choose_device,load_json
LEVELS=("level0","level1","level2","level3","level4")
def peak_mib():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=="darwin" else 1024)

def prepare_cache(records,config,paths,device):
 cache=Path(config["corrected_cache"]);cache.mkdir(parents=True,exist_ok=True);encoder=RS2NetEncoderAdapter(paths,image_size=tuple(config["tile_size"]),in_channels=1,out_channels=1,feature_size=48).to(device).eval()
 for p in encoder.parameters():p.requires_grad_(False)
 for domain in ("camri","mouse"):
  for split in ("train","validation"):
   for i,r in enumerate(records[domain][split],1):
    dest=cache/f"{domain}_{split}_{r['subject']}.pt";r["cache_path"]=str(dest)
    if dest.exists():continue
    image,target,shape,_=preprocess_image_and_corrected_target(Path(r["image_path"]),Path(r["mask_path"]),paths,tuple(config["tile_size"]));
    with torch.inference_mode():features=encoder(image.to(device))
    torch.save({"features":{k:features[k].cpu().half() for k in LEVELS},"target":target.byte(),"preprocessed_shape":shape,"label_interpolation":"nearest/order0/is_seg=True"},dest);print(f"cache {domain} {split} {i}/{len(records[domain][split])}",flush=True)
 del encoder

@torch.inference_mode()
def evaluate(decoder,records,device):
 decoder.eval();out=[]
 for r in records:
  f,t=load_cached(r,device);logits=decoder(f,output_size=t.shape[-3:]);out.append({"subject":r["subject"],**metric(logits,t)})
 return out

def initialize(decoder,path):
 payload=torch.load(path,map_location="cpu",weights_only=False);old=payload["decoder_state_dict"];current=decoder.state_dict();shared={k:v for k,v in old.items() if k in current and current[k].shape==v.shape};missing,unexpected=decoder.load_state_dict(shared,strict=False)
 # Every old baseline tensor must transfer; only new level0 tensors are missing.
 if set(shared)!=set(old):raise RuntimeError("Not all baseline decoder tensors transferred")
 return payload,missing,unexpected

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/full_resolution_level0_decoder.yaml");args=ap.parse_args();config=load_json(ROOT/args.config);out=ROOT/config["output_directory"];(out/"checkpoints").mkdir(parents=True,exist_ok=True);(out/"training").mkdir(exist_ok=True);(out/"config.json").write_text(json.dumps(config,indent=2))
 random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"]);device=choose_device();paths=RS2NetPaths.from_config(load_json(ROOT/config["encoder_config"]));records=cached_records(config);prepare_cache(records,config,paths,device);decoder=FullResolutionLevel0OneQueryMaskDecoder(32,4);initial,missing,unexpected=initialize(decoder,ROOT/config["initial_checkpoint"]);decoder.to(device);parameters=sum(p.numel() for p in decoder.parameters())
 # Mandatory pre-training geometry gate.
 f,t=load_cached(records["camri"]["validation"][0],device);with_shape=decoder(f,output_size=t.shape[-3:]);native=decoder(f)
 if native.shape[-3:]!=(128,128,160) or with_shape.shape!=t.shape:raise RuntimeError("Level0 logits are not full-grid")
 optimizer=torch.optim.AdamW(decoder.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"]);camri_ref=np.mean([float(r["dice"]) for r in csv.DictReader(open(ROOT/config["camri_metrics"])) if r["split"]=="validation"]);best=-1;state=copy.deepcopy(decoder.state_dict());best_epoch=0;stale=0;history=[];safety=False;start=time.time();peak=0.
 for epoch in range(1,config["max_epochs"]+1):
  decoder.train();losses=[];epoch_start=time.time()
  for j,(domain,r) in enumerate(balanced_epoch_order(records["camri"]["train"],records["mouse"]["train"],config["seed"]+epoch)):
   features,target=load_cached(r,device);features,target=augment(features,target,random.Random(config["seed"]+epoch*100+j),config["augmentation"]);optimizer.zero_grad(set_to_none=True);logits=decoder(features,output_size=target.shape[-3:]);loss,parts=training_loss(logits,target,config);loss.backward();optimizer.step();losses.append(float(loss.detach()))
  val={d:aggregate(evaluate(decoder,records[d]["validation"],device)) for d in ("camri","mouse")};score=(val["camri"]["dice"]+val["mouse"]["dice"])/2;elapsed=time.time()-epoch_start;peak=max(peak,peak_mib());row={"epoch":epoch,"loss":np.mean(losses),"epoch_seconds":elapsed,"balanced_validation_dice":score,**{f"{d}_validation_{k}":v for d in val for k,v in val[d].items()}};history.append(row);print(f"epoch {epoch} CAMRI={val['camri']['dice']:.6f} Mouse={val['mouse']['dice']:.6f} balanced={score:.6f} seconds={elapsed:.1f}",flush=True)
  if val["camri"]["dice"]<camri_ref-config["camri_validation_max_drop"]:safety=True;break
  if score>best+config["minimum_validation_improvement"]:best=score;state=copy.deepcopy(decoder.state_dict());best_epoch=epoch;stale=0
  else:stale+=1
  if stale>=config["early_stop_patience"]:break
 torch.save({"decoder_state_dict":state,"epoch":best_epoch,"balanced_validation_dice":best,"initial_checkpoint":config["initial_checkpoint"],"config":config},out/"checkpoints/best_level0_decoder.pt")
 with (out/"training/history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
 summary={"device":str(device),"selected_epoch":best_epoch,"epochs_run":len(history),"best_balanced_validation_dice":best,"encoder_frozen":True,"one_query":True,"levels":list(LEVELS),"old_decoder_parameters":170401,"level0_decoder_parameters":parameters,"architecture_change":"level0 full-grid branch only","native_logits_shape":[1,1,128,128,160],"final_logit_interpolation":False,"query_conditioning_at_level0":True,"level0_query_tokens_pooled_2x":True,"camri_safety_stop":safety,"mean_epoch_seconds":np.mean([r["epoch_seconds"] for r in history]),"peak_process_memory_mib":peak,"elapsed_seconds":time.time()-start,"transferred_baseline_tensors":len(initial["decoder_state_dict"]),"new_parameter_keys":list(missing),"unexpected_keys":list(unexpected)};(out/"training/summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
