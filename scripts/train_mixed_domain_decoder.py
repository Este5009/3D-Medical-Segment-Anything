#!/usr/bin/env python3
"""Balanced CAMRI+mouse training of the unchanged one-query decoder."""
from __future__ import annotations
import argparse,copy,csv,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"scripts"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import numpy as np
import torch
from models.query_mask_decoder import MultiScaleOneQueryMaskDecoder,dice_bce_boundary_loss,dice_bce_loss
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter,RS2NetPaths
from train_generalization_pilot import load_cached
from train_mouse_boundary_adaptation import augment,metric,aggregate
from train_query_decoder_overfit import choose_device,load_json,preprocess_pair

def balanced_epoch_order(camri,mouse,seed):
    rng=random.Random(seed);a=list(camri);b=list(mouse);rng.shuffle(a);rng.shuffle(b);n=max(len(a),len(b));a=(a*((n+len(a)-1)//len(a)))[:n];b=(b*((n+len(b)-1)//len(b)))[:n];return [x for pair in zip([("camri",v) for v in a],[('mouse',v) for v in b]) for x in pair]

def cached_records(config):
    cs=load_json(ROOT/config["camri_split"]);ms=load_json(ROOT/config["mouse_split"]);result={d:{s:[] for s in ("train","validation","test")} for d in ("camri","mouse")}
    camri_source={r["subject"]:r for r in csv.DictReader(open(ROOT/config["camri_metrics"]))}
    mouse_source={r["scan_id"]:r for r in csv.DictReader(open(ROOT/config["mouse_metrics"]))}
    for split,ids in cs.items():
        for sid in ids:
            source=camri_source[sid]
            result["camri"][split].append({"subject":sid,"cache_path":str(Path(config["camri_cache"])/f"{split}_{sid}.pt"),"image_path":source["image_path"],"mask_path":source["mask_path"]})
    for split in ("train","validation"):
        for sid in ms[split]["scans"]:
            source=mouse_source[sid]
            result["mouse"][split].append({"subject":sid,"cache_path":str(Path(config["mouse_cache"])/f"{split}_{sid}.pt"),"image_path":source["image_path"],"mask_path":source["ground_truth_path"]})
    return result

def ensure_feature_cache(records,config,device):
    """Rebuild expired `/tmp` features from the locked split when necessary."""
    required=[r for domain in records.values() for split in ("train","validation") for r in domain[split]]
    missing=[r for r in required if not Path(r["cache_path"]).exists()]
    if not missing:
        return
    encoder_config=load_json(ROOT/config["encoder_config"])
    paths=RS2NetPaths.from_config(encoder_config)
    encoder=RS2NetEncoderAdapter(paths,image_size=tuple(config["tile_size"]),in_channels=1,out_channels=1,feature_size=48).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for index,record in enumerate(missing,1):
        image,target,preprocessed_shape,_=preprocess_pair(Path(record["image_path"]),Path(record["mask_path"]),paths,tuple(config["tile_size"]))
        with torch.inference_mode():
            features=encoder(image.to(device))
        destination=Path(record["cache_path"]);destination.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"features":{name:value.detach().cpu().half() for name,value in features.items() if name!="level0"},"target":target.byte(),"preprocessed_shape":preprocessed_shape},destination)
        print(f"cached frozen features {index}/{len(missing)}: {record['subject']}",flush=True)
    del encoder

@torch.inference_mode()
def evaluate(decoder,records,device):
    decoder.eval();rs=[]
    for r in records:
        features,target=load_cached(r,device);logits=decoder(features,output_size=target.shape[-3:]);rs.append({"subject":r["subject"],**metric(logits,target)})
    return rs

def training_loss(logits,target,config):
    """Select the one declared supervision recipe without changing the model."""
    loss_config=config.get("loss",{"name":"dice_bce"})
    if loss_config["name"]=="dice_bce":
        return dice_bce_loss(logits,target)
    if loss_config["name"]=="dice_bce_boundary":
        return dice_bce_boundary_loss(
            logits,
            target,
            boundary_weight=float(loss_config["boundary_weight"]),
            boundary_width=int(loss_config["boundary_width"]),
        )
    raise ValueError(f"Unsupported loss recipe: {loss_config['name']}")

def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/mixed_domain_decoder.yaml",
        help="JSON-formatted experiment config relative to the repository root.",
    )
    return parser.parse_args()

def main():
    args=parse_args();config=load_json(ROOT/args.config);out=ROOT/config["output_directory"];out.mkdir(parents=True,exist_ok=True);(out/"checkpoints").mkdir(exist_ok=True);(out/"learning_curves").mkdir(exist_ok=True);(out/"configuration.json").write_text(json.dumps(config,indent=2));random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"]);device=choose_device();initial=torch.load(ROOT/config["initial_checkpoint"],map_location="cpu",weights_only=False);decoder=MultiScaleOneQueryMaskDecoder(32,4);decoder.load_state_dict(initial["decoder_state_dict"],strict=True);decoder=decoder.to(device);assert tuple(decoder.query.shape)==(1,1,32) and sum(p.numel() for p in decoder.parameters())==170401;records=cached_records(config);ensure_feature_cache(records,config,device)
    camri_reference=np.mean([float(r["dice"]) for r in csv.DictReader(open(ROOT/config["camri_metrics"])) if r["split"]=="validation"]);baseline={d:aggregate(evaluate(decoder,records[d]["validation"],device)) for d in ("camri","mouse")};optimizer=torch.optim.AdamW(decoder.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"]);best=-1;best_state=copy.deepcopy(decoder.state_dict());best_epoch=0;stale=0;history=[];stopped_for_camri=False
    for epoch in range(1,config["max_epochs"]+1):
        decoder.train();losses=[];order=balanced_epoch_order(records["camri"]["train"],records["mouse"]["train"],config["seed"]+epoch)
        for j,(domain,r) in enumerate(order):
            features,target=load_cached(r,device);features,target=augment(features,target,random.Random(config["seed"]+epoch*100+j),config["augmentation"]);optimizer.zero_grad(set_to_none=True);logits=decoder(features,output_size=target.shape[-3:]);loss,parts=training_loss(logits,target,config);loss.backward();optimizer.step();losses.append({"loss":float(loss.detach()),**{k:float(v) for k,v in parts.items()}})
        va={d:aggregate(evaluate(decoder,records[d]["validation"],device)) for d in ("camri","mouse")};score=(va["camri"]["dice"]+va["mouse"]["dice"])/2;loss_means={k:float(np.mean([x[k] for x in losses])) for k in losses[0]};row={"epoch":epoch,**loss_means,"balanced_validation_dice":score,**{f"{d}_validation_{k}":v for d in va for k,v in va[d].items()}};history.append(row);print(f"epoch {epoch} CAMRI={va['camri']['dice']:.4f} Mouse={va['mouse']['dice']:.4f} balanced={score:.4f}",flush=True)
        if va["camri"]["dice"]<camri_reference-config["camri_validation_max_drop"]:stopped_for_camri=True;print("CAMRI safety stop",flush=True);break
        if score>best+config["minimum_validation_improvement"]:best=score;best_state=copy.deepcopy(decoder.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=config["early_stop_patience"]:break
    decoder.load_state_dict(best_state);torch.save({"decoder_state_dict":best_state,"epoch":best_epoch,"balanced_validation_dice":best,"initial_checkpoint":config["initial_checkpoint"],"config":config},out/"checkpoints"/"best_mixed_domain.pt")
    with (out/"learning_curves"/"history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    split={"camri":load_json(ROOT/config["camri_split"]),"mouse":load_json(ROOT/config["mouse_split"])};(out/"split.json").write_text(json.dumps(split,indent=2));summary={"device":str(device),"initial_epoch":initial["epoch"],"encoder_frozen":True,"exactly_one_query":True,"decoder_parameters_trained":170401,"architecture_changed":False,"loss":config.get("loss",{"name":"dice_bce"}),"baseline_validation":baseline,"camri_validation_reference":camri_reference,"best_epoch":best_epoch,"epochs_run":len(history),"best_balanced_validation_dice":best,"camri_safety_stop":stopped_for_camri};(out/"training_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
