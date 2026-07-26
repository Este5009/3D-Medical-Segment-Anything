#!/usr/bin/env python3
"""Train only the final boundary-producing layers of the locked one-query decoder."""
from __future__ import annotations
import copy,csv,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import numpy as np
import torch
import torch.nn.functional as F
from models.query_mask_decoder import FrozenEncoderQueryModel,MultiScaleOneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter,RS2NetPaths
from train_generalization_pilot import load_cached
from train_query_decoder_overfit import choose_device,load_json,preprocess_pair

LEVELS=("level1","level2","level3","level4")

def freeze_boundary_head_only(decoder):
    """Freeze query, attention, projections, and FPN; expose only final mask layers."""
    allowed=("mask_embedding.","mask_refinement.","mask_bias")
    for name,p in decoder.named_parameters():p.requires_grad_(name.startswith(allowed))
    names=[n for n,p in decoder.named_parameters() if p.requires_grad];return names,sum(p.numel() for p in decoder.parameters() if p.requires_grad)

def adaptation_loss(logits,target,weights,alpha,beta):
    probability=logits.sigmoid();dims=tuple(range(1,logits.ndim));tp=(probability*target).sum(dims);fp=(probability*(1-target)).sum(dims);fn=((1-probability)*target).sum(dims);smooth=1e-5
    dice=1-(2*tp+smooth)/(2*tp+fp+fn+smooth);tversky=1-(tp+smooth)/(tp+alpha*fp+beta*fn+smooth);bce=F.binary_cross_entropy_with_logits(logits,target)
    total=weights["dice"]*dice.mean()+weights["bce"]*bce+weights["tversky"]*tversky.mean();return total,{"dice_loss":float(dice.mean().detach()),"bce_loss":float(bce.detach()),"tversky_loss":float(tversky.mean().detach())}

def metric(logits,target):
    p=logits.sigmoid()>=.5;t=target>=.5;tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum());return {"dice":2*tp/max(2*tp+fp+fn,1),"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"false_positives":fp,"false_negatives":fn,"volume_ratio":int(p.sum())/max(int(t.sum()),1)}

def make_split(config):
    source=list(csv.DictReader(open(ROOT/config["mouse_metrics"])));train=set(config["train_mouse_ids"]);val=set(config["validation_mouse_ids"]);split={"train":[],"validation":[],"test":[]}
    for r in source:split["train" if r["mouse_id"] in train else "validation" if r["mouse_id"] in val else "test"].append(r)
    ids={k:{r["mouse_id"] for r in v if not r["mouse_id"].startswith("anonymous-")} for k,v in split.items()};assert not(ids["train"]&ids["validation"] or ids["train"]&ids["test"] or ids["validation"]&ids["test"]);assert sum(map(len,split.values()))==101
    return split

def save_split(split,out):
    payload={}
    flat=[]
    for name,rs in split.items():
        payload[name]={"scan_count":len(rs),"identified_mice":sorted({r["mouse_id"] for r in rs if not r["mouse_id"].startswith("anonymous-")}),"anonymous_scans":sum(r["mouse_id"].startswith("anonymous-") for r in rs),"scans":[r["scan_id"] for r in rs],"baseline_dice_mean":float(np.mean([float(r["dice"]) for r in rs])),"baseline_dice_std":float(np.std([float(r["dice"]) for r in rs]))}
        flat += [{"split":name,"scan_id":r["scan_id"],"mouse_id":r["mouse_id"],"timepoint":r["timepoint"],"baseline_dice":r["dice"],"shape_and_spacing_group":f"{r['slice_count']} slices; {r['spacing_x']}x{r['spacing_y']}x{r['spacing_z']} mm"} for r in rs]
    (out/"split.json").write_text(json.dumps(payload,indent=2));
    with (out/"split.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=flat[0]);w.writeheader();w.writerows(flat)

def cache_features(model,split,paths,tile,root,device):
    records={k:[] for k in split};root.mkdir(parents=True,exist_ok=True)
    for name in ("train","validation"):
        for i,r in enumerate(split[name],1):
            path=root/f"{name}_{r['scan_id']}.pt"
            if not path.exists():
                image,target,shape,_=preprocess_pair(Path(r["image_path"]),Path(r["ground_truth_path"]),paths,tile);features=model.encode(image.to(device));torch.save({"features":{k:v.cpu().half() for k,v in features.items() if k in LEVELS},"target":target.byte(),"preprocessed_shape":shape},path)
            records[name].append({"subject":r["scan_id"],"cache_path":str(path)});print(f"cache {name} {i}/{len(split[name])}",flush=True)
    return records

@torch.inference_mode()
def evaluate(model,records,device):
    model.eval();result=[]
    for r in records:
        features,target=load_cached(r,device);logits=model.decode(features,target.shape[-3:]);result.append({"scan_id":r["subject"],**metric(logits,target)})
    return result

def aggregate(rs):return {k:float(np.mean([r[k] for r in rs])) for k in ("dice","precision","recall","false_positives","false_negatives","volume_ratio")}

def augment(features,target,rng,config):
    features={k:v.clone() for k,v in features.items()};target=target.clone()
    for axis in (2,3,4):
        if rng.random()<config["flip_probability"]:
            target=torch.flip(target,(axis,));features={k:torch.flip(v,(axis,)) for k,v in features.items()}
    scale=1+rng.uniform(-config["feature_scale"],config["feature_scale"]);features={k:v*scale+torch.randn_like(v)*config["feature_noise_std"] for k,v in features.items()};return features,target

def main():
    config=load_json(ROOT/"configs/mouse_boundary_adaptation.yaml");out=ROOT/config["output_directory"];out.mkdir(parents=True,exist_ok=True);(out/"checkpoints").mkdir(exist_ok=True);(out/"learning_curves").mkdir(exist_ok=True);(out/"configuration.json").write_text(json.dumps(config,indent=2));random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"])
    split=make_split(config);save_split(split,out);enc_cfg=load_json(ROOT/config["encoder_config"]);paths=RS2NetPaths.from_config(enc_cfg);device=choose_device();decoder=MultiScaleOneQueryMaskDecoder(32,4);initial=torch.load(ROOT/config["initial_checkpoint"],map_location="cpu",weights_only=False);decoder.load_state_dict(initial["decoder_state_dict"],strict=True);names,count=freeze_boundary_head_only(decoder);encoder=RS2NetEncoderAdapter(paths,image_size=tuple(config["tile_size"]),in_channels=1,out_channels=1,feature_size=48);model=FrozenEncoderQueryModel(encoder,decoder).to(device);assert tuple(decoder.query.shape)==(1,1,32) and not decoder.query.requires_grad and not any(p.requires_grad for p in model.encoder.parameters())
    records=cache_features(model,split,paths,tuple(config["tile_size"]),Path(config["temporary_feature_cache"]),device);baseline={k:aggregate(evaluate(model,records[k],device)) for k in records};optimizer=torch.optim.AdamW([p for p in decoder.parameters() if p.requires_grad],lr=config["learning_rate"],weight_decay=config["weight_decay"]);best=-1;best_state=None;best_epoch=0;stale=0;history=[];started=time.time()
    for epoch in range(1,config["max_epochs"]+1):
        model.train();batch=[]
        order=list(records["train"]);random.Random(config["seed"]+epoch).shuffle(order)
        for j,r in enumerate(order):
            features,target=load_cached(r,device);features,target=augment(features,target,random.Random(config["seed"]+epoch*100+j),config["augmentation"]);optimizer.zero_grad(set_to_none=True);logits=model.decode(features,target.shape[-3:]);loss,parts=adaptation_loss(logits,target,config["loss_weights"],config["tversky_alpha_fp"],config["tversky_beta_fn"]);loss.backward();optimizer.step();batch.append({"loss":float(loss.detach()),**parts})
        tr=aggregate(evaluate(model,records["train"],device));va=aggregate(evaluate(model,records["validation"],device));row={"epoch":epoch,**{k:float(np.mean([x[k] for x in batch])) for k in batch[0]},**{f"train_{k}":v for k,v in tr.items()},**{f"validation_{k}":v for k,v in va.items()}};history.append(row);print(f"epoch {epoch} train={tr['dice']:.4f} val={va['dice']:.4f} P={va['precision']:.4f} R={va['recall']:.4f} ratio={va['volume_ratio']:.4f}",flush=True)
        if va["dice"]>best+config["minimum_validation_improvement"]:best=va["dice"];best_state=copy.deepcopy(decoder.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=config["early_stop_patience"]:break
    decoder.load_state_dict(best_state,strict=True);final={k:aggregate(evaluate(model,records[k],device)) for k in records};torch.save({"decoder_state_dict":best_state,"epoch":best_epoch,"validation":final["validation"],"trainable_parameter_names":names,"trainable_parameters":count,"initial_checkpoint":config["initial_checkpoint"],"config":config},out/"checkpoints"/"boundary_head_best.pt")
    with (out/"learning_curves"/"boundary_head_history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    trigger=config["experiment_c_trigger"];reasons=[]
    if final["validation"]["dice"]-baseline["validation"]["dice"]<trigger["validation_dice_improvement_below"]:reasons.append("validation Dice improvement below 0.02")
    if final["validation"]["precision"]<trigger["validation_precision_below"]:reasons.append("validation precision below 0.90")
    if final["validation"]["volume_ratio"]>trigger["validation_volume_ratio_above"]:reasons.append("validation volume ratio above 1.10")
    summary={"training_performed":True,"experiment":"boundary-head-only","device":str(device),"initial_epoch":initial["epoch"],"exactly_one_frozen_query":True,"trained_parameters":names,"trainable_parameters":count,"encoder_frozen":True,"query_and_attention_frozen":True,"split":json.loads((out/"split.json").read_text()),"baseline_model_space":baseline,"adapted_model_space":final,"best_epoch":best_epoch,"epochs_run":len(history),"elapsed_seconds":time.time()-started,"experiment_c_triggered":bool(reasons),"experiment_c_trigger_reasons":reasons};(out/"training_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
