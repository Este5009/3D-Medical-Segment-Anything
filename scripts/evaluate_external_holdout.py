#!/usr/bin/env python3
"""Pure external-holdout evaluation of the locked epoch-14 pilot checkpoint.

This file intentionally contains no optimizer, no training loop, no backward
call, and no parameter updates. Every prediction is produced under
``torch.inference_mode`` from the strictly loaded checkpoint.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt

from models.query_mask_decoder import FrozenEncoderQueryModel, MultiScaleOneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import choose_device, load_json, verify_geometry

FEATURE_NAMES = ("level1", "level2", "level3", "level4")


def discover_subjects(config, dataset_root):
    image_root = dataset_root / config["image_root"]
    masks = list((dataset_root / config["mask_directory"]).glob("*.nii.gz"))
    records = []
    for image in sorted(image_root.glob("sub-*/ses-1/anat/*RARE*T2w.nii*")):
        match = re.search(r"sub-(\d+)", str(image))
        if not match:
            continue
        subject = match.group(1)
        matched = [mask for mask in masks if re.search(rf"sub-{subject}(?:_|$)", mask.name)]
        if len(matched) == 1:
            records.append({"subject": subject, "image": image, "mask": matched[0]})
    return records


def restore_mask_geometry(image_path: Path, mask_path: Path, destination: Path):
    """Resample a label into image physical coordinates with nearest neighbor."""
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    restored = sitk.Resample(mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    array = sitk.GetArrayFromImage(restored)
    if int((array > 0).sum()) == 0:
        raise ValueError("physical-coordinate resampling produced an empty mask")
    geometry_matches = (
        restored.GetSize() == image.GetSize()
        and np.allclose(restored.GetSpacing(), image.GetSpacing())
        and np.allclose(restored.GetOrigin(), image.GetOrigin())
        and np.allclose(restored.GetDirection(), image.GetDirection())
    )
    if not geometry_matches:
        raise ValueError("restored SimpleITK geometry does not equal the image reference grid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(restored, str(destination), True)
    return {
        "status": "restored",
        "method": "SimpleITK identity physical transform with nearest-neighbor interpolation",
        "original_mask_size": list(mask.GetSize()),
        "image_size": list(image.GetSize()),
        "restored_foreground_voxels": int((array > 0).sum()),
        "simpleitk_reference_geometry_matches": True,
        "restored_mask": str(destination),
    }


def center_tile(array: torch.Tensor, size):
    original = tuple(array.shape[-3:])
    padding = []
    for current, target in reversed(list(zip(original, size))):
        total = max(target - current, 0)
        padding.extend((total // 2, total - total // 2))
    padded = F.pad(array, padding)
    starts = [(current - target) // 2 for current, target in zip(padded.shape[-3:], size)]
    tile = padded[..., starts[0]:starts[0]+size[0], starts[1]:starts[1]+size[1], starts[2]:starts[2]+size[2]]
    return tile, {"original": original, "padded": tuple(padded.shape[-3:]), "starts": starts}


def sliding_window_logits(model, image, tile_size, device):
    """Average overlapping tile logits over the complete preprocessed volume."""
    original = tuple(image.shape[-3:])
    padding = []
    for current, target in reversed(list(zip(original, tile_size))):
        total = max(target-current, 0); padding.extend((total//2, total-total//2))
    padded = F.pad(image, padding); spatial = tuple(padded.shape[-3:])
    starts_per_axis = []
    for current, tile in zip(spatial, tile_size):
        starts = list(range(0, max(current-tile, 0)+1, max(tile//2, 1)))
        if starts[-1] != current-tile: starts.append(current-tile)
        starts_per_axis.append(starts)
    logits_sum = torch.zeros((1,1,*spatial), device=device); counts = torch.zeros_like(logits_sum)
    for d in starts_per_axis[0]:
        for h in starts_per_axis[1]:
            for w in starts_per_axis[2]:
                tile = padded[...,d:d+tile_size[0],h:h+tile_size[1],w:w+tile_size[2]].to(device)
                features=model.encode(tile); logits=model.decode({name:features[name] for name in FEATURE_NAMES},tile_size)
                logits_sum[...,d:d+tile_size[0],h:h+tile_size[1],w:w+tile_size[2]] += logits
                counts[...,d:d+tile_size[0],h:h+tile_size[1],w:w+tile_size[2]] += 1
    averaged = logits_sum / counts.clamp_min(1)
    crop_starts=[(current-original_size)//2 for current,original_size in zip(spatial,original)]
    return averaged[...,crop_starts[0]:crop_starts[0]+original[0],crop_starts[1]:crop_starts[1]+original[1],crop_starts[2]:crop_starts[2]+original[2]]


def preprocess(image_path, mask_path, paths, tile_size):
    from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from RS2.utilities.plans_handling.plans_handler import PlansManager

    json_root = paths.baseline_project / "RS2" / "jsons"
    plans = load_json(json_root / "plans.json"); dataset = load_json(json_root / "dataset.json")
    manager = PlansManager(plans); configuration = manager.get_configuration("3d_fullres")
    data, segmentation, properties = DefaultPreprocessor(verbose=False).run_case(
        [str(image_path)], str(mask_path), manager, configuration, dataset
    )
    image = torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0)
    return image, properties, manager, configuration, dataset


def export_native(logits, properties, manager, configuration, dataset, destination):
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

    unpadded = logits.cpu()[0].numpy().astype(np.float32)
    current_spacing = configuration.spacing if len(configuration.spacing) == len(
        properties["shape_after_cropping_and_before_resampling"]
    ) else [properties["spacing"][0], *configuration.spacing]
    native_crop_logits = configuration.resampling_fn(
        unpadded, properties["shape_after_cropping_and_before_resampling"],
        current_spacing, properties["spacing"],
    )
    # The released binary network has one output channel although dataset.json
    # declares background/foreground labels. A logit threshold of zero is the
    # exact binary equivalent of sigmoid probability >= 0.5.
    segmentation = (native_crop_logits[0] > 0).astype(np.uint8)
    reverted = np.zeros(properties["shape_before_cropping"], dtype=np.uint8)
    reverted[bounding_box_to_slice(properties["bbox_used_for_cropping"])] = segmentation
    reverted = reverted.transpose(manager.transpose_backward)
    manager.image_reader_writer_class().write_seg(reverted, str(destination), properties)


def surface_hd95(prediction, target, spacing):
    prediction = prediction.astype(bool); target = target.astype(bool)
    if not prediction.any() or not target.any():
        return float("inf")
    pred_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate((distance_to_target[pred_surface], distance_to_pred[target_surface]))
    return float(np.percentile(distances, 95))


def metrics(prediction, target, spacing):
    prediction = prediction.astype(bool); target = target.astype(bool)
    tp = int((prediction & target).sum()); fp = int((prediction & ~target).sum()); fn = int((~prediction & target).sum())
    dice = (2*tp+1e-5)/(2*tp+fp+fn+1e-5); iou=(tp+1e-5)/(tp+fp+fn+1e-5)
    precision=(tp+1e-5)/(tp+fp+1e-5); recall=(tp+1e-5)/(tp+fn+1e-5)
    return {"dice":dice,"iou":iou,"precision":precision,"recall":recall,"false_positives":fp,"false_negatives":fn,
            "hd95":surface_hd95(prediction,target,spacing),"brain_volume_voxels":int(target.sum()),
            "brain_volume_mm3":float(target.sum()*np.prod(spacing))}


def slice_metrics(prediction, target):
    rows=[]
    for index in range(target.shape[2]):
        pred=prediction[:,:,index].astype(bool); truth=target[:,:,index].astype(bool);tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
        denom=2*tp+fp+fn; union=tp+fp+fn
        rows.append({"slice_index":index,"dice":(2*tp/denom if denom else 1.0),"iou":(tp/union if union else 1.0),"false_positives":fp,"false_negatives":fn,"error":fp+fn,"brain_voxels":int(truth.sum())})
    return rows


def normalize_slice(array):
    array=array.astype(np.float32); return (array-array.min())/max(float(np.ptp(array)),1e-8)


def render_selected_slices(image, target, prediction, rows, destination):
    import imageio.v3 as iio
    from PIL import Image, ImageDraw
    brain=[r["slice_index"] for r in rows if r["brain_voxels"]>0]
    selected=[brain[0],brain[len(brain)//2],brain[-1],max(rows,key=lambda r:r["error"])["slice_index"]]
    bands=[]
    for index in selected:
        mri=normalize_slice(image[:,:,index]); gt=target[:,:,index].astype(bool); pred=prediction[:,:,index].astype(bool)
        base=np.stack([mri]*3,-1); gt_overlay=base.copy();gt_overlay[gt]=(0,1,0);pred_overlay=base.copy();pred_overlay[pred]=(1,0,0)
        fpfn=base.copy();fpfn[gt&pred]=(1,1,0);fpfn[gt&~pred]=(0,1,0);fpfn[pred&~gt]=(1,0,0)
        panels=[base,gt_overlay,pred_overlay,fpfn];sep=np.ones((base.shape[0],3,3));band=np.concatenate([x for p in panels for x in (p,sep)][:-1],axis=1)
        label=f"slice {index}  Dice {rows[index]['dice']:.4f}"
        pil=Image.fromarray((np.clip(band,0,1)*255).astype(np.uint8));ImageDraw.Draw(pil).text((4,4),label,fill=(255,255,255),stroke_width=1,stroke_fill=(0,0,0));bands.append(np.asarray(pil)/255.0)
    iio.imwrite(destination,(np.clip(np.concatenate(bands,axis=0),0,1)*255).astype(np.uint8))


def save_curve(rows,key,destination):
    import imageio.v3 as iio
    canvas=np.full((420,760,3),255,np.uint8);values=np.asarray([r[key] for r in rows],float);worst=int(np.argmin(values) if key in ("dice","iou") else np.argmax(values));maximum=max(float(values.max()),1.0)
    points=[]
    for i,value in enumerate(values): points.append((50+int(i/max(len(values)-1,1)*680),390-int(value/maximum*350)))
    for a,b in zip(points,points[1:]):
        n=max(abs(b[0]-a[0]),abs(b[1]-a[1]),1);xs=np.linspace(a[0],b[0],n+1).astype(int);ys=np.linspace(a[1],b[1],n+1).astype(int);canvas[ys,xs]=(0,90,180)
    x,y=points[worst];canvas[max(0,y-5):y+6,max(0,x-5):x+6]=(220,20,20);iio.imwrite(destination,canvas)


def save_native_overlays(image_path,prediction_path,target_path,subject_dir):
    import nibabel as nib
    image_obj=nib.load(str(image_path)); image=np.asarray(image_obj.dataobj,dtype=np.float32); target=np.asarray(nib.load(str(target_path)).dataobj)>0;prediction=np.asarray(nib.load(str(prediction_path)).dataobj)>0
    normalized=(image-image.min())/max(float(np.ptp(image)),1e-8);rgb=np.stack([normalized]*3,-1);rgb[target]=(0,1,0);rgb[prediction]=(1,0,0);rgb[target&prediction]=(1,1,0)
    labels=np.zeros(target.shape,np.uint8);labels[target&~prediction]=1;labels[prediction&~target]=2;labels[target&prediction]=3
    nib.save(nib.Nifti1Image(rgb.astype(np.float32),image_obj.affine,image_obj.header),subject_dir/"overlay_mri_gt_prediction.nii.gz")
    nib.save(nib.Nifti1Image(labels,image_obj.affine,image_obj.header),subject_dir/"fpfn_overlay_labels.nii.gz")
    return image,target,prediction


def plot_dataset(rows,key,destination,histogram=False):
    import imageio.v3 as iio
    canvas=np.full((480,800,3),255,np.uint8)
    if histogram:
        values=np.asarray([r[key] for r in rows],float);counts,_=np.histogram(values,bins=np.linspace(values.min(),values.max()+1e-12,21));maximum=max(int(counts.max()),1)
        for i,count in enumerate(counts):canvas[430-int(count/maximum*380):430,55+int(i*700/20):55+int((i+1)*700/20)-2]=(0,100,175)
    else:
        xs=np.asarray([r[key] for r in rows],float);ys=np.asarray([r["dice"] for r in rows]);span=max(float(xs.max()-xs.min()),1e-8)
        for x,y in zip(xs,ys):
            px=55+int((x-xs.min())/span*700);py=430-int(y*390);canvas[max(0,py-4):py+5,max(0,px-4):px+5]=(190,40,40)
    iio.imwrite(destination,canvas)


def main():
    config=load_json(REPO_ROOT/"configs/external_holdout.yaml");encoder_config=load_json(REPO_ROOT/config["encoder_config"]);paths=RS2NetPaths.from_config(encoder_config);out=REPO_ROOT/config["output_directory"]
    for name in ("best_subjects","worst_subjects","per_subject","per_slice","native_predictions"):(out/name).mkdir(parents=True,exist_ok=True)
    split=load_json(REPO_ROOT/config["pilot_split"]);used=set(sum(split.values(),[]));discovered=discover_subjects(config,paths.dataset_root);holdout=[r for r in discovered if r["subject"] not in used]
    checkpoint=torch.load(REPO_ROOT/config["checkpoint"],map_location="cpu",weights_only=False)
    if checkpoint["epoch"]!=config["expected_checkpoint_epoch"]:raise RuntimeError("checkpoint epoch mismatch")
    decoder=MultiScaleOneQueryMaskDecoder(config["embedding_dim"],config["num_heads"]);decoder.load_state_dict(checkpoint["decoder_state_dict"],strict=True)
    if tuple(decoder.query.shape)!=(1,1,config["embedding_dim"]):raise RuntimeError("model does not contain exactly one learned query")
    if sum(p.numel() for p in decoder.parameters())!=config["expected_decoder_parameters"]:raise RuntimeError("decoder parameter count mismatch")
    device=choose_device();encoder=RS2NetEncoderAdapter(paths,image_size=(128,128,160),in_channels=1,out_channels=1,feature_size=48);model=FrozenEncoderQueryModel(encoder,decoder).to(device).eval()
    audit={"subjects_discovered":len(discovered),"pilot_subjects_excluded":sorted(used),"holdout_candidates":len(holdout),"geometry":{},"excluded":[]};evaluated=[]
    for number,record in enumerate(holdout,1):
        subject=record["subject"];subject_json=out/"per_subject"/f"sub-{subject}_metrics.json";mask_path=record["mask"]
        try:verify_geometry(record["image"],mask_path);audit["geometry"][subject]={"status":"original_geometry_valid"}
        except ValueError as error:
            restored=out/"per_subject"/f"sub-{subject}_restored_mask.nii.gz"
            try:audit["geometry"][subject]=restore_mask_geometry(record["image"],mask_path,restored);mask_path=restored
            except Exception as restore_error:audit["excluded"].append({"subject":subject,"reason":str(restore_error),"original_error":str(error)});continue
        if subject_json.exists():
            row=load_json(subject_json)
            subject_dir=out/"per_subject"/f"sub-{subject}";subject_dir.mkdir(exist_ok=True)
            import nibabel as nib
            image=np.asarray(nib.load(row["image_path"]).dataobj,dtype=np.float32);target=np.asarray(nib.load(row["ground_truth_path"]).dataobj)>0;prediction=np.asarray(nib.load(row["prediction_path"]).dataobj)>0
            slices=[{k:(int(v) if k in ("slice_index","false_positives","false_negatives","error","brain_voxels") else float(v)) for k,v in item.items()} for item in csv.DictReader((out/"per_slice"/f"sub-{subject}_slices.csv").open())]
            render_selected_slices(image,target,prediction,slices,subject_dir/"selected_slices.png")
            evaluated.append(row);print(f"resume {number}/{len(holdout)} sub-{subject}",flush=True);continue
        image_preprocessed,properties,manager,configuration,dataset=preprocess(record["image"],mask_path,paths,(128,128,160))
        started=time.perf_counter()
        with torch.inference_mode():logits=sliding_window_logits(model,image_preprocessed,(128,128,160),device)
        inference=time.perf_counter()-started;prediction_path=out/"native_predictions"/f"sub-{subject}_prediction_0000.nii.gz";export_native(logits,properties,manager,configuration,dataset,prediction_path)
        subject_dir=out/"per_subject"/f"sub-{subject}";subject_dir.mkdir(exist_ok=True)
        image,target,prediction=save_native_overlays(record["image"],prediction_path,mask_path,subject_dir)
        import nibabel as nib
        image_obj=nib.load(str(record["image"]));spacing=tuple(map(float,image_obj.header.get_zooms()[:3]));m=metrics(prediction,target,spacing);slices=slice_metrics(prediction,target)
        slice_csv=out/"per_slice"/f"sub-{subject}_slices.csv"
        with slice_csv.open("w",newline="") as stream:w=csv.DictWriter(stream,fieldnames=list(slices[0]));w.writeheader();w.writerows(slices)
        for key in ("dice","iou","false_positives","false_negatives"):save_curve(slices,key,out/"per_slice"/f"sub-{subject}_{key}.png")
        render_selected_slices(image,target,prediction,slices,subject_dir/"selected_slices.png")
        row={"subject":subject,**m,"slice_count":int(image.shape[2]),"spacing_x":spacing[0],"spacing_y":spacing[1],"spacing_z":spacing[2],"mean_spacing":float(np.mean(spacing)),"inference_seconds":inference,"image_path":str(record["image"]),"ground_truth_path":str(mask_path),"prediction_path":str(prediction_path)}
        subject_json.write_text(json.dumps(row,indent=2));evaluated.append(row);print(f"evaluated {number}/{len(holdout)} sub-{subject} dice={m['dice']:.5f}",flush=True)
    audit["subjects_evaluated"]=len(evaluated);audit["subjects_excluded"]=len(audit["excluded"]);(out/"cohort_audit.json").write_text(json.dumps(audit,indent=2))
    fields=list(evaluated[0]);
    with (out/"subject_metrics.csv").open("w",newline="") as stream:w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(evaluated)
    slice_rows=[]
    for row in evaluated:
        for item in csv.DictReader((out/"per_slice"/f"sub-{row['subject']}_slices.csv").open()):slice_rows.append({"subject":row["subject"],**item})
    with (out/"metrics.csv").open("w",newline="") as stream:w=csv.DictWriter(stream,fieldnames=list(slice_rows[0]));w.writeheader();w.writerows(slice_rows)
    ranked=sorted(evaluated,key=lambda r:r["dice"]);import imageio.v3 as iio
    for group,rows in (("worst_subjects",ranked[:10]),("best_subjects",list(reversed(ranked[-10:])))):
        for row in rows:iio.imwrite(out/group/f"sub-{row['subject']}_comparison.png",iio.imread(out/"per_subject"/f"sub-{row['subject']}"/"selected_slices.png"))
    plot_dataset(evaluated,"dice",out/"dice_histogram.png",True);plot_dataset(evaluated,"brain_volume_mm3",out/"dice_vs_volume.png");plot_dataset(evaluated,"slice_count",out/"dice_vs_slice_count.png");plot_dataset(evaluated,"mean_spacing",out/"dice_vs_spacing.png");plot_dataset(evaluated,"inference_seconds",out/"inference_times.png",True)
    dices=np.asarray([r["dice"] for r in evaluated]);pilot=load_json(REPO_ROOT/"outputs/generalization_pilot/summary.json")["metrics"]
    summary={"training_performed":False,"checkpoint_epoch":checkpoint["epoch"],"exactly_one_query":tuple(decoder.query.shape)==(1,1,32),"query_shape":list(decoder.query.shape),"decoder_parameters":sum(p.numel() for p in decoder.parameters()),"subjects_discovered":len(discovered),"pilot_subjects_excluded":len(used),"holdout_candidates":len(holdout),"subjects_evaluated":len(evaluated),"subjects_excluded":len(audit["excluded"]),"mean_dice":float(dices.mean()),"median_dice":float(np.median(dices)),"std_dice":float(dices.std()),"min_dice":float(dices.min()),"max_dice":float(dices.max()),"percentile_5":float(np.percentile(dices,5)),"percentile_95":float(np.percentile(dices,95)),"mean_precision":float(np.mean([r['precision'] for r in evaluated])),"mean_recall":float(np.mean([r['recall'] for r in evaluated])),"mean_iou":float(np.mean([r['iou'] for r in evaluated])),"mean_hd95":float(np.mean([r['hd95'] for r in evaluated])),"mean_inference_seconds":float(np.mean([r['inference_seconds'] for r in evaluated])),"pilot_dice":{"train":pilot['train']['mean_dice'],"validation":pilot['validation']['mean_dice'],"test":pilot['test']['mean_dice']},"gaps":{"pilot_train_minus_holdout":pilot['train']['mean_dice']-float(dices.mean()),"pilot_validation_minus_holdout":pilot['validation']['mean_dice']-float(dices.mean()),"pilot_test_minus_holdout":pilot['test']['mean_dice']-float(dices.mean())},"best_10":[{"subject":r['subject'],"dice":r['dice']} for r in reversed(ranked[-10:])],"worst_10":[{"subject":r['subject'],"dice":r['dice']} for r in ranked[:10]]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
