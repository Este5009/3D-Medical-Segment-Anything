#!/usr/bin/env python3
"""First controlled subject-level generalization pilot for the frozen RS2 encoder."""

from __future__ import annotations

import copy
import csv
import json
import random
import re
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from models.query_mask_decoder import FrozenEncoderQueryModel, MultiScaleOneQueryMaskDecoder, dice_bce_loss
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import choose_device, load_json, preprocess_pair, verify_geometry

FEATURE_NAMES = ("level1", "level2", "level3", "level4")


def discover_pairs(config: dict, dataset_root: Path):
    """Pair by subject identifier and exclude shape/affine-invalid pairs."""
    image_root = dataset_root / "Image_Database/CAMRI Rat Brain MRI Data"
    masks = list((dataset_root / config["mask_directory"]).glob("*.nii.gz"))
    pairs = []
    for image in sorted(image_root.glob("sub-*/ses-1/anat/*RARE*T2w.nii*")):
        match = re.search(r"sub-(\d+)", str(image))
        if match is None:
            continue
        subject = match.group(1)
        subject_masks = [mask for mask in masks if re.search(rf"sub-{subject}(?:_|$)", mask.name)]
        if len(subject_masks) == 1:
            try:
                verify_geometry(image, subject_masks[0])
            except ValueError as error:
                print(f"excluded sub-{subject}: {error}", flush=True)
                continue
            pairs.append((subject, image, subject_masks[0]))
    return pairs


def make_subject_split(pairs, config: dict):
    """Select 40 subjects, then split them 28/6/6 without slice leakage."""
    shuffled = list(pairs)
    random.Random(int(config["seed"])).shuffle(shuffled)
    subset = shuffled[: int(config["subset_subject_count"])]
    train_end = int(config["train_count"])
    validation_end = train_end + int(config["validation_count"])
    split = {
        "train": subset[:train_end],
        "validation": subset[train_end:validation_end],
        "test": subset[validation_end:],
    }
    ids = {name: {item[0] for item in values} for name, values in split.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise RuntimeError("Subject leakage detected between splits")
    if sum(map(len, split.values())) != int(config["subset_subject_count"]):
        raise RuntimeError("Configured split sizes do not equal subset size")
    return split


def binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    prediction = logits.sigmoid() >= 0.5
    truth = target >= 0.5
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
    precision = (tp + 1e-5) / (tp + fp + 1e-5)
    recall = (tp + 1e-5) / (tp + fn + 1e-5)
    return {"dice": dice, "precision": precision, "recall": recall, "false_positives": fp, "false_negatives": fn}


def prepare_cache(model, split, paths, tile_size, cache_root: Path):
    """Compute immutable features once; save float16 maps and uint8 targets."""
    cache_root.mkdir(parents=True, exist_ok=True)
    records = {name: [] for name in split}
    for split_name, pairs in split.items():
        for index, (subject, image_path, mask_path) in enumerate(pairs, 1):
            cache_path = cache_root / f"{split_name}_{subject}.pt"
            geometry = verify_geometry(image_path, mask_path)
            if not cache_path.exists():
                image, mask, preprocessed_shape, _ = preprocess_pair(image_path, mask_path, paths, tile_size)
                encoded = model.encode(image.to(next(model.parameters()).device))
                payload = {
                    "features": {name: encoded[name].detach().cpu().half() for name in FEATURE_NAMES},
                    "target": (mask >= 0.5).to(torch.uint8),
                    "preprocessed_shape": preprocessed_shape,
                }
                torch.save(payload, cache_path)
                del encoded, payload
            records[split_name].append({
                "subject": subject, "image_path": str(image_path), "mask_path": str(mask_path),
                "cache_path": str(cache_path), "geometry": geometry,
            })
            print(f"cached {split_name} {index}/{len(pairs)}: sub-{subject}", flush=True)
    return records


def load_cached(record, device):
    payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
    features = {name: value.float().to(device) for name, value in payload["features"].items()}
    target = payload["target"].float().to(device)
    return features, target


@torch.inference_mode()
def evaluate_split(model, records, device, split_name):
    model.eval()
    rows = []
    for record in records:
        features, target = load_cached(record, device)
        started = time.perf_counter()
        logits = model.decode(features, output_size=target.shape[-3:])
        inference_seconds = time.perf_counter() - started
        metrics = binary_metrics(logits, target)
        payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        row = {
            "split": split_name, "subject": record["subject"], **metrics,
            "brain_volume_voxels": int(target.sum().cpu()),
            "slice_count": int(record["geometry"]["original_shape"][2]),
            "inference_seconds": inference_seconds,
            "image_path": record["image_path"], "mask_path": record["mask_path"],
            "logits": logits.cpu(), "target": target.cpu(),
            "preprocessed_shape": payload["preprocessed_shape"],
        }
        rows.append(row)
    return rows


def aggregate(rows):
    dices = np.asarray([row["dice"] for row in rows], dtype=float)
    best = max(rows, key=lambda row: row["dice"])
    worst = min(rows, key=lambda row: row["dice"])
    return {
        "count": len(rows), "mean_dice": float(dices.mean()), "median_dice": float(np.median(dices)),
        "std_dice": float(dices.std()), "mean_precision": float(np.mean([r["precision"] for r in rows])),
        "mean_recall": float(np.mean([r["recall"] for r in rows])),
        "false_positives": int(sum(r["false_positives"] for r in rows)),
        "false_negatives": int(sum(r["false_negatives"] for r in rows)),
        "best_subject": best["subject"], "best_dice": best["dice"],
        "worst_subject": worst["subject"], "worst_dice": worst["dice"],
    }


def _canvas(width=800, height=480):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _line(canvas, x0, y0, x1, y1, color, thickness=2):
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).astype(int); ys = np.linspace(y0, y1, steps + 1).astype(int)
    for offset in range(-thickness // 2, thickness // 2 + 1):
        canvas[np.clip(ys + offset, 0, canvas.shape[0]-1), np.clip(xs, 0, canvas.shape[1]-1)] = color


def save_learning_curve(history, destination):
    import imageio.v3 as iio
    canvas = _canvas(); margin = 55; max_epoch = max(row["epoch"] for row in history)
    _line(canvas, margin, 420, 760, 420, (0,0,0)); _line(canvas, margin, 30, margin, 420, (0,0,0))
    for key, color in (("train_dice", (0,100,180)), ("validation_dice", (210,70,40))):
        points = [(int(margin + row["epoch"]/max_epoch*705), int(420-row[key]*390)) for row in history]
        for first, second in zip(points, points[1:]): _line(canvas, *first, *second, color, 3)
    iio.imwrite(destination, canvas)


def save_histogram(rows, destination):
    import imageio.v3 as iio
    canvas = _canvas(); margin = 55; values = np.asarray([r["dice"] for r in rows]); bins = np.linspace(0,1,21)
    counts, _ = np.histogram(values, bins); maximum = max(int(counts.max()), 1)
    for i, count in enumerate(counts):
        x0 = margin + int(i*705/20); x1 = margin + int((i+1)*705/20)-2; y = 420-int(count/maximum*370)
        canvas[y:420, x0:x1] = (0,105,175)
    _line(canvas, margin,420,760,420,(0,0,0)); _line(canvas,margin,30,margin,420,(0,0,0)); iio.imwrite(destination,canvas)


def save_scatter(rows, x_key, destination):
    import imageio.v3 as iio
    canvas = _canvas(); margin=55; xs=np.asarray([r[x_key] for r in rows],float); ys=np.asarray([r["dice"] for r in rows],float)
    xmin,xmax=float(xs.min()),float(xs.max()); span=max(xmax-xmin,1.0)
    for x,y in zip(xs,ys):
        px=int(margin+(x-xmin)/span*705); py=int(420-y*390); canvas[max(0,py-4):py+5,max(0,px-4):px+5]=(190,40,40)
    _line(canvas,margin,420,760,420,(0,0,0)); _line(canvas,margin,30,margin,420,(0,0,0)); iio.imwrite(destination,canvas)


def save_case_figure(row, output_dir, paths, tile_size):
    import imageio.v3 as iio
    import nibabel as nib
    image, _, _, _ = preprocess_pair(Path(row["image_path"]), Path(row["mask_path"]), paths, tile_size)
    image = image[0,0].numpy(); target=row["target"][0,0].numpy()>0.5; probability=row["logits"].sigmoid()[0,0].numpy(); prediction=probability>=0.5
    depth=int(target.sum(axis=(1,2)).argmax()); mri=image[depth]; mri=(mri-mri.min())/max(float(np.ptp(mri)),1e-8)
    gt=target[depth].astype(float); pred=prediction[depth].astype(float)
    overlay=np.stack([mri,mri,mri],axis=-1); fp=prediction[depth]&~target[depth]; fn=~prediction[depth]&target[depth]
    overlay[fp]=(1,0,0); overlay[fn]=(0,0.4,1)
    panels=[np.stack([mri]*3,-1),np.stack([pred]*3,-1),np.stack([gt]*3,-1),overlay]
    sep=np.ones((panels[0].shape[0],4,3)); montage=np.concatenate((panels[0],sep,panels[1],sep,panels[2],sep,panels[3]),axis=1)
    iio.imwrite(output_dir/f"sub-{row['subject']}_mri_prediction_gt_fpfn.png",(np.clip(montage,0,1)*255).astype(np.uint8))
    affine=np.diag([0.25,0.20000000298,0.15999999642,1.0]); nib.save(nib.Nifti1Image(prediction.astype(np.uint8),affine),output_dir/f"sub-{row['subject']}_prediction_model_space.nii.gz")


def main():
    config=load_json(REPO_ROOT/"configs/generalization_pilot.yaml"); encoder_config=load_json(REPO_ROOT/config["encoder_config"])
    output_dir=REPO_ROOT/config["output_directory"]; output_dir.mkdir(parents=True,exist_ok=True)
    cache_root=Path(config["temporary_feature_cache"]); paths=RS2NetPaths.from_config(encoder_config); tile_size=tuple(encoder_config["model"]["image_size"])
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"]); device=choose_device()
    split=make_subject_split(discover_pairs(config,paths.dataset_root),config)
    (output_dir/"subject_split.json").write_text(json.dumps({k:[x[0] for x in v] for k,v in split.items()},indent=2))
    encoder=RS2NetEncoderAdapter(paths,image_size=tile_size,in_channels=1,out_channels=1,feature_size=48)
    model=FrozenEncoderQueryModel(encoder,MultiScaleOneQueryMaskDecoder(config["embedding_dim"],config["num_heads"])).to(device)
    records=prepare_cache(model,split,paths,tile_size,cache_root)
    optimizer=torch.optim.AdamW(model.decoder.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    best_dice=-1.0; best_epoch=0; best_state=None; stale=0; history=[]; started=time.perf_counter()
    for epoch in range(1,int(config["max_epochs"])+1):
        model.train(); losses=[]
        shuffled=list(records["train"]); random.Random(config["seed"]+epoch).shuffle(shuffled)
        for record in shuffled:
            features,target=load_cached(record,device); optimizer.zero_grad(set_to_none=True); logits=model.decode(features,target.shape[-3:]); loss,_=dice_bce_loss(logits,target); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        train_rows=evaluate_split(model,records["train"],device,"train"); validation_rows=evaluate_split(model,records["validation"],device,"validation")
        train_dice=aggregate(train_rows)["mean_dice"]; validation_dice=aggregate(validation_rows)["mean_dice"]
        history.append({"epoch":epoch,"loss":float(np.mean(losses)),"train_dice":train_dice,"validation_dice":validation_dice})
        print(f"epoch={epoch:03d} loss={np.mean(losses):.6f} train_dice={train_dice:.6f} validation_dice={validation_dice:.6f}",flush=True)
        if validation_dice>best_dice+config["minimum_validation_improvement"]:
            best_dice=validation_dice;best_epoch=epoch;best_state=copy.deepcopy(model.decoder.state_dict());stale=0
        else: stale+=1
        if stale>=config["early_stop_patience"]: break
    torch.save({"decoder_state_dict":best_state,"epoch":best_epoch,"validation_dice":best_dice,"config":config},output_dir/"best_checkpoint.pt");model.decoder.load_state_dict(best_state)
    all_rows=[]
    for name in ("train","validation","test"): all_rows.extend(evaluate_split(model,records[name],device,name))
    csv_fields=["split","subject","dice","precision","recall","false_positives","false_negatives","brain_volume_voxels","slice_count","inference_seconds","image_path","mask_path"]
    with (output_dir/"metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=csv_fields);writer.writeheader();writer.writerows([{k:r[k] for k in csv_fields} for r in all_rows])
    with (output_dir/"training_history.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["epoch","loss","train_dice","validation_dice"]);writer.writeheader();writer.writerows(history)
    save_learning_curve(history,output_dir/"learning_curves.png");save_histogram(all_rows,output_dir/"dice_histogram.png");save_scatter(all_rows,"brain_volume_voxels",output_dir/"dice_vs_brain_volume.png");save_scatter(all_rows,"slice_count",output_dir/"dice_vs_slice_count.png")
    ranked=sorted(all_rows,key=lambda r:r["dice"]); worst=ranked[:10];best=list(reversed(ranked[-10:]));failure_dir=output_dir/"failure_cases";qual_dir=output_dir/"qualitative_figures";failure_dir.mkdir(exist_ok=True);qual_dir.mkdir(exist_ok=True)
    for row in worst: save_case_figure(row,failure_dir,paths,tile_size)
    for row in best: save_case_figure(row,qual_dir,paths,tile_size)
    aggregates={name:aggregate([r for r in all_rows if r["split"]==name]) for name in ("train","validation","test")}
    train_test_gap=aggregates["train"]["mean_dice"]-aggregates["test"]["mean_dice"]; train_val_gap=aggregates["train"]["mean_dice"]-aggregates["validation"]["mean_dice"]
    summary={"architecture":"frozen RS2 encoder; one query; multi-scale attention; top-down FPN; one mask head","trainable_parameters":model.trainable_parameter_count(),"device":str(device),"encoder_gradients_disabled":all(not p.requires_grad and p.grad is None for p in model.encoder.parameters()),"available_paired_subjects":len(discover_pairs(config,paths.dataset_root)),"subset_subjects":sum(len(v) for v in records.values()),"split_subjects":{k:[r["subject"] for r in v] for k,v in records.items()},"best_epoch":best_epoch,"epochs_run":len(history),"best_validation_dice":best_dice,"metrics":aggregates,"train_validation_gap":train_val_gap,"train_test_gap":train_test_gap,"overfitting_detected":train_test_gap>0.05 or train_val_gap>0.05,"worst_10":[{"split":r["split"],"subject":r["subject"],"dice":r["dice"]} for r in worst],"best_10":[{"split":r["split"],"subject":r["subject"],"dice":r["dice"]} for r in best],"mean_decoder_inference_seconds":float(np.mean([r["inference_seconds"] for r in all_rows])),"peak_memory_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 if sys.platform=="darwin" else None,"elapsed_training_seconds":time.perf_counter()-started}
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
