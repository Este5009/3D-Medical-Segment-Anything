#!/usr/bin/env python3
"""Audit expert-mask preprocessing without changing any experiment behavior.

The script reproduces the *current* RS2 ``3d_fullres`` segmentation path and
compares it with a label-aware nearest-neighbour reference.  The reference is
diagnostic only: it is never written into a cache and never used for training
or inference.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from models.rs2net_encoder_adapter import RS2NetPaths
from train_query_decoder_overfit import _pad_and_center_crop, load_json

OUT = ROOT / "outputs/mask_preprocessing_audit"
TILE = (128, 128, 160)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), bool))


def metrics(pred: np.ndarray, ref: np.ndarray, spacing) -> dict:
    pred = pred.astype(bool); ref = ref.astype(bool)
    tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
    ps, rs = surface(pred), surface(ref)
    if ps.any() and rs.any():
        dref = ndimage.distance_transform_edt(~rs, sampling=spacing)[ps]
        dpred = ndimage.distance_transform_edt(~ps, sampling=spacing)[rs]
        sym = np.concatenate((dref, dpred))
        hd95, assd = float(np.percentile(sym, 95)), float(sym.mean())
    else:
        hd95 = assd = float("inf")
    return {"dice": 2 * tp / max(2 * tp + fp + fn, 1), "hd95_mm": hd95,
            "assd_mm": assd, "added_voxels": fp, "lost_voxels": fn}


def contour_stats(mask: np.ndarray) -> tuple[int, float]:
    """Count 2-D contour direction changes and normalized perimeter."""
    changes = 0; perimeter = 0.0; area = int(mask.sum())
    for axis in range(3):
        for index in range(mask.shape[axis]):
            sl = np.take(mask, index, axis=axis)
            boundary = sl & ~ndimage.binary_erosion(sl, structure=np.ones((3, 3), bool))
            perimeter += float(boundary.sum())
            coords = np.argwhere(boundary)
            if len(coords) > 2:
                center = coords.mean(0); angle = np.arctan2(coords[:, 0]-center[0], coords[:, 1]-center[1])
                ordered = coords[np.argsort(angle)]
                steps = np.sign(np.diff(ordered, axis=0))
                changes += int(np.any(steps[1:] != steps[:-1], axis=1).sum())
    return changes, perimeter / max(area ** (2 / 3), 1.0)


def records() -> list[dict]:
    split = load_json(ROOT / "outputs/mixed_domain_anatomical_training/split.json")
    cam = {r["subject"]: r for r in csv.DictReader(open(ROOT / "outputs/generalization_pilot/metrics.csv"))}
    mouse = {r["scan_id"]: r for r in csv.DictReader(open(ROOT / "outputs/mouse_external_evaluation/subject_metrics.csv"))}
    result = []
    for split_name, ids in split["camri"].items():
        for sid in ids:
            r = cam[sid]; result.append({"domain":"CAMRI", "split":split_name, "subject":sid,
                "image_path":r["image_path"], "mask_path":r["mask_path"]})
    for split_name in ("train", "validation", "test"):
        for sid in split["mouse"][split_name]["scans"]:
            r = mouse[sid]; result.append({"domain":"Mouse", "split":split_name, "subject":sid,
                "image_path":r["image_path"], "mask_path":r["ground_truth_path"]})
    return result


def setup():
    from RS2.utilities.plans_handling.plans_handler import PlansManager
    paths = RS2NetPaths.from_config(load_json(ROOT / "configs/rs2net_encoder.yaml"))
    base = paths.baseline_project / "RS2/jsons"
    plans = load_json(base / "plans.json"); dataset = load_json(base / "dataset.json")
    manager = PlansManager(plans); config = manager.get_configuration("3d_fullres")
    return manager, config, dataset


def process_case(record, manager, config, dataset):
    """Return current/reference masks at processed and restored-native grids."""
    from RS2.preprocessing.cropping.cropping import crop_to_nonzero
    from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from RS2.preprocessing.resampling.default_resampling import compute_new_shape, resample_data_or_seg_to_shape
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
    import torch

    rw = manager.image_reader_writer_class()
    data, props = rw.read_images([record["image_path"]]); seg, _ = rw.read_seg(record["mask_path"])
    original = seg[0] > 0
    data_t = data.transpose([0, *[i+1 for i in manager.transpose_forward]])
    seg_t = seg.transpose([0, *[i+1 for i in manager.transpose_forward]])
    spacing_t = [props["spacing"][i] for i in manager.transpose_forward]
    data_c, seg_c, bbox = crop_to_nonzero(data_t.copy(), seg_t.copy())
    target_spacing = list(config.spacing)
    new_shape = compute_new_shape(data_c.shape[1:], spacing_t, target_spacing)

    # Capture the actual float interpolation before DefaultPreprocessor's final
    # int8 cast, then independently call run_case to assert faithful reproduction.
    current_float = config.resampling_fn(seg_c, new_shape, spacing_t, target_spacing)
    current = current_float.astype(np.int8) > 0
    _, official, official_props = DefaultPreprocessor(False).run_case(
        [record["image_path"]], record["mask_path"], manager, config, dataset)
    if not np.array_equal(current, official > 0):
        raise AssertionError(f"Current-method reproduction mismatch: {record['subject']}")
    nearest = resample_data_or_seg_to_shape(seg_c, new_shape, spacing_t, target_spacing,
        is_seg=True, order=0, order_z=0, force_separate_z=None) > 0

    def restore(processed):
        cropped = resample_data_or_seg_to_shape(processed.astype(np.uint8), data_c.shape[1:],
            target_spacing, spacing_t, is_seg=True, order=0, order_z=0, force_separate_z=None)[0] > 0
        full = np.zeros(data_t.shape[1:], bool); full[bounding_box_to_slice(bbox)] = cropped
        return full.transpose(manager.transpose_backward)

    # This is the exact target subsequently cached by all current train/val
    # loaders (threshold, batch dim, centre pad/crop, uint8 cache).
    cur_tile = _pad_and_center_crop(torch.from_numpy(current.astype(np.float32)).unsqueeze(0), TILE).numpy() > 0
    nn_tile = _pad_and_center_crop(torch.from_numpy(nearest.astype(np.float32)).unsqueeze(0), TILE).numpy() > 0
    return {"original":original, "current":current[0], "nearest":nearest[0],
            "current_restored":restore(current), "nearest_restored":restore(nearest),
            "current_float":current_float, "current_tile":cur_tile[0,0], "nearest_tile":nn_tile[0,0],
            "native_spacing":tuple(props["spacing"]), "processed_spacing":tuple(target_spacing),
            "bbox":bbox, "official_props":official_props}


def make_figure(record, arrays, row, destination):
    original = arrays["original"]; cur = arrays["current_restored"]; nn = arrays["nearest_restored"]
    diff = np.zeros((*original.shape, 3), np.float32)
    diff[nn & ~cur] = (1, 1, 0); diff[cur & ~nn] = (1, 0, 0)
    errors = (nn ^ cur).sum(axis=(1,2)); z = int(errors.argmax())
    coords = np.argwhere((nn | cur)[z]); lo = np.maximum(coords.min(0)-8, 0); hi = np.minimum(coords.max(0)+9, nn.shape[1:])
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    panels = [original[z], cur[z], nn[z], diff[z], diff[z,lo[0]:hi[0],lo[1]:hi[1]]]
    titles = ["Original native expert", "Current: linear + int cast\nrestored with NN",
              "NN reference\nrestored with NN", "Difference\nyellow=lost, red=added", "Boundary zoom"]
    for ax, panel, title in zip(axes, panels, titles):
        shown = panel.transpose(1, 0, 2) if panel.ndim == 3 else panel.T
        ax.imshow(shown, cmap=None if panel.ndim==3 else "gray", origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=9); ax.axis("off")
    fig.suptitle(f"{record['domain']} | {record['subject']}\n"
                 f"native spacing {arrays['native_spacing']} mm | processed {arrays['current'].shape} | "
                 f"volume change {row['processed_volume_change_percent']:+.2f}% | "
                 f"current-vs-NN Dice {row['processed_current_vs_nn_dice']:.5f}", fontsize=10)
    destination.parent.mkdir(parents=True, exist_ok=True); fig.savefig(destination, dpi=180); plt.close(fig)


def summarize(subject_rows):
    summaries = []
    for domain in ("CAMRI", "Mouse", "Combined"):
        rows = subject_rows if domain == "Combined" else [r for r in subject_rows if r["domain"] == domain]
        for scope in ("processed_full", "training_tile", "restored_native"):
            prefix = {"processed_full":"processed_current_vs_nn", "training_tile":"tile_current_vs_nn",
                      "restored_native":"native_original_vs_current"}[scope]
            out = {"domain":domain, "scope":scope, "subject_count":len(rows)}
            for key in ("dice", "hd95_mm", "assd_mm"):
                values = np.asarray([float(r[f"{prefix}_{key}"]) for r in rows])
                out[f"mean_{key}"] = float(values.mean()); out[f"median_{key}"] = float(np.median(values))
                out[f"p95_{key}"] = float(np.percentile(values,95)); out[f"max_{key}"] = float(values.max())
            out.update({"total_lost_voxels":sum(int(r[f"{prefix}_lost_voxels"]) for r in rows),
                        "total_added_voxels":sum(int(r[f"{prefix}_added_voxels"]) for r in rows),
                        "mean_volume_change_percent":float(np.mean([float(r["processed_volume_change_percent"]) for r in rows]))})
            summaries.append(out)
    return summaries


def documentation(subject_rows, summary_rows):
    train_example = next(r for r in subject_rows if r["domain"]=="CAMRI" and r["split"]=="train")
    val_example = next(r for r in subject_rows if r["domain"]=="Mouse" and r["split"]=="validation")
    table = [
      {"stage":"load", "train":"yes", "validation":"yes", "test":"preprocess only", "operation":"SimpleITK GetArrayFromImage; [C,Z,Y,X]", "input_shape":"native", "output_shape":"[1,Z,Y,X]", "interpolation":"none", "dtype":"float32"},
      {"stage":"transpose", "train":"yes", "validation":"yes", "test":"image/unused seg", "operation":"transpose_forward [1,0,2]", "input_shape":"[1,Z,Y,X]", "output_shape":"[1,Y,Z,X]", "interpolation":"none", "dtype":"float32"},
      {"stage":"crop", "train":"yes", "validation":"yes", "test":"image/unused seg", "operation":"image nonzero bbox; outside label set -1", "input_shape":"transposed", "output_shape":"bbox dependent", "interpolation":"none", "dtype":"float32"},
      {"stage":"spacing resample", "train":"yes", "validation":"yes", "test":"seg result discarded", "operation":"same configuration.resampling_fn as image", "input_shape":"cropped", "output_shape":"computed target shape", "interpolation":"is_seg=False, order=1; order_z=0 if separate-z", "dtype":"float32"},
      {"stage":"integer cast", "train":"yes", "validation":"yes", "test":"seg result discarded", "operation":"DefaultPreprocessor final cast", "input_shape":"resampled", "output_shape":"unchanged", "interpolation":"none; truncation toward zero", "dtype":"int8"},
      {"stage":"target conversion", "train":"yes", "validation":"yes", "test":"no", "operation":"segmentation > 0; float tensor; center pad/crop", "input_shape":"processed", "output_shape":"[1,1,128,128,160]", "interpolation":"padding/cropping only", "dtype":"float32 then uint8 cache"},
      {"stage":"augmentation", "train":"yes", "validation":"no", "test":"no", "operation":"axis flips shared with frozen features", "input_shape":"tile", "output_shape":"tile", "interpolation":"none", "dtype":"float32"},
      {"stage":"validation metric", "train":"training reports too", "validation":"yes", "test":"no", "operation":"threshold logits at 0.5 vs cached processed tile", "input_shape":"model tile", "output_shape":"scalar", "interpolation":"none", "dtype":"bool"},
      {"stage":"canonical native test", "train":"no", "validation":"no", "test":"yes", "operation":"export logits to native; reload untouched expert NIfTI", "input_shape":"native prediction + native expert", "output_shape":"scalar metrics", "interpolation":"order=1 applies to logits only", "dtype":"expert thresholded bool"},
    ]
    write_csv(OUT/"mask_pipeline_table.csv", table)
    usage = [{"file":"../RS2-Net-Reproduction/Rodent-Skull-Stripping/RS2/jsons/plans.json", "function":"3d_fullres resampling_fn_kwargs", "caller":"DefaultPreprocessor.run_case", "setting":"is_seg=False, order=1, order_z=0", "training_labels":"yes", "validation_labels":"yes", "test_labels":"computed then discarded", "metric_calculation":"train/validation model-space only", "visualization_only":"no"},
      {"file":"scripts/evaluate_external_holdout.py", "function":"export_native", "caller":"canonical native evaluators", "setting":"configured is_seg=False, order=1 on logits", "training_labels":"no", "validation_labels":"no", "test_labels":"no", "metric_calculation":"affects prediction geometry, never expert GT", "visualization_only":"no"},
      {"file":"scripts/evaluate_mouse_boundary_adaptation.py", "function":"export_probability", "caller":"native evaluation", "setting":"configured is_seg=False, order=1 on logits", "training_labels":"no", "validation_labels":"no", "test_labels":"no", "metric_calculation":"probability map only", "visualization_only":"diagnostics/probability"}]
    write_csv(OUT/"interpolation_usage.csv", usage)
    write_csv(OUT/"training_mask_examples.csv", [train_example])
    write_csv(OUT/"validation_mask_examples.csv", [val_example])
    sources = [
      {"evaluation":"mixed training validation Dice/precision/recall", "script":"scripts/train_mixed_domain_decoder.py -> evaluate", "mask_source":"cached preprocess_pair target", "representation":"linearly resampled, int8-cast, >0, center tile", "native_untouched":"no", "metrics":"Dice, precision, recall, FP, FN"},
      {"evaluation":"canonical CAMRI/Mouse native test", "script":"scripts/evaluate_mouse_boundary_adaptation.py -> run_records/native_eval", "mask_source":"ground_truth_path loaded with nibabel", "representation":"original native expert mask >0", "native_untouched":"yes", "metrics":"Dice, IoU, precision, recall, HD95, FP/FN, volume ratio"},
      {"evaluation":"fast mixed-domain Dice", "script":"scripts/evaluate_mixed_domain_dice_fast.py", "mask_source":"source mask_path/ground_truth_path loaded with nibabel", "representation":"original native expert mask >0", "native_untouched":"yes", "metrics":"Dice and overlap metrics"},
      {"evaluation":"mixed results analysis", "script":"scripts/analyze_mixed_domain_results.py", "mask_source":"ground_truth_path loaded with nibabel", "representation":"original native expert mask >0", "native_untouched":"yes", "metrics":"native overlap/error analysis"},
      {"evaluation":"filtered residual failures", "script":"scripts/analyze_filtered_residual_failures.py", "mask_source":"ground_truth_path loaded with nibabel", "representation":"original native expert mask >0", "native_untouched":"yes", "metrics":"Dice/HD95/error composition"},
      {"evaluation":"boundary diagnostics", "script":"scripts/analyze_boundary_error_diagnostics.py", "mask_source":"ground_truth_path loaded with nibabel", "representation":"original native expert mask >0", "native_untouched":"yes", "metrics":"Dice, HD95, ASSD, surface/boundary diagnostics"},
    ]
    write_csv(OUT/"test_metric_mask_sources.csv", sources)
    trace = f"""# Expert-mask lifecycle trace\n\n## Definitive paths\n\n- **Training:** original expert NIfTI -> SimpleITK float32 `[C,Z,Y,X]` -> transpose `[1,0,2]` -> image-nonzero crop -> the shared `3d_fullres` `is_seg=False, order=1` resampler -> int8 truncation -> `>0` -> center pad/crop to `[1,1,128,128,160]` -> uint8 feature-cache target -> optional axis flips. This is a **linearly interpolated mask cast to integer**, option C in the audit question.\n- **Validation:** identical preprocessing and cache target, without training augmentation. Validation Dice is prediction thresholded at 0.5 versus that processed model-space tile, not native GT. Concrete example: `{val_example['mask_path']}`.\n- **Canonical test:** preprocessing receives the GT path, but `preprocess()` discards the returned segmentation. Logits are restored to native geometry, thresholded, and compared with `{chr(96)}ground_truth_path{chr(96)}` reloaded directly by nibabel and binarized `>0`. Thus canonical Dice/IoU/precision/recall/HD95 and later ASSD/boundary diagnostics use untouched native expert labels.\n\n## Concrete training example\n\n`{train_example['mask_path']}` starts at shape `{train_example['original_shape']}`, spacing `{train_example['native_spacing_mm']}` mm and `{train_example['original_voxels']}` foreground voxels. Its processed target is `{train_example['processed_shape']}` at `{train_example['processed_spacing_mm']}` mm; after the current method it contains `{train_example['current_processed_voxels']}` foreground voxels versus `{train_example['nearest_processed_voxels']}` with nearest-neighbour. The actual cached tile is `[1,1,128,128,160]`.\n\n## Important distinction\n\nThe faulty label interpolation directly affects train and validation supervision. It does **not** alter the expert mask used by the canonical native test metrics. The order-1 native export is applied to logits/probabilities, not expert labels.\n"""
    (OUT/"mask_pipeline_trace.md").write_text(trace, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT/"figures").mkdir(exist_ok=True)
    manager, config, dataset = setup(); rows=[]; figure_payload=[]
    for index, record in enumerate(records(), 1):
        a = process_case(record, manager, config, dataset)
        proc = metrics(a["current"], a["nearest"], a["processed_spacing"])
        tile = metrics(a["current_tile"], a["nearest_tile"], a["processed_spacing"])
        native_cur = metrics(a["current_restored"], a["original"], a["native_spacing"])
        native_nn = metrics(a["nearest_restored"], a["original"], a["native_spacing"])
        lost = a["nearest"] & ~a["current"]; added = a["current"] & ~a["nearest"]
        dcur = ndimage.distance_transform_edt(~surface(a["current"]), sampling=a["processed_spacing"])
        dnn = ndimage.distance_transform_edt(~surface(a["nearest"]), sampling=a["processed_spacing"])
        cc, cp = contour_stats(a["current"]); nc, np_ = contour_stats(a["nearest"])
        original_count=int(a["original"].sum()); current_count=int(a["current"].sum()); nn_count=int(a["nearest"].sum())
        row={**record, "original_shape":str(tuple(a["original"].shape)), "native_spacing_mm":str(a["native_spacing"]),
             "processed_shape":str(tuple(a["current"].shape)), "processed_spacing_mm":str(a["processed_spacing"]),
             "original_unique_values":"0,1", "pre_integer_resample_unique_count":int(np.unique(a["current_float"]).size),
             "pre_integer_resample_min":float(a["current_float"].min()), "pre_integer_resample_max":float(a["current_float"].max()),
             "post_integer_unique_values":str(np.unique(a["current_float"].astype(np.int8)).tolist()),
             "original_voxels":original_count, "current_processed_voxels":current_count, "nearest_processed_voxels":nn_count,
             "current_processed_volume_mm3":current_count*float(np.prod(a["processed_spacing"])),
             "nearest_processed_volume_mm3":nn_count*float(np.prod(a["processed_spacing"])),
             "processed_volume_change_percent":100*(current_count-nn_count)/max(nn_count,1),
             **{f"processed_current_vs_nn_{k}":v for k,v in proc.items()},
             **{f"tile_current_vs_nn_{k}":v for k,v in tile.items()},
             **{f"native_original_vs_current_{k}":v for k,v in native_cur.items()},
             **{f"native_original_vs_nn_{k}":v for k,v in native_nn.items()},
             "lost_added_ratio":int(lost.sum())/max(int(added.sum()),1),
             "net_processed_voxel_change":current_count-nn_count,
             "mean_inward_shift_mm":float(dcur[lost].mean()) if lost.any() else 0.0,
             "mean_outward_shift_mm":float(dnn[added].mean()) if added.any() else 0.0,
             "current_contour_direction_changes":cc, "nearest_contour_direction_changes":nc,
             "current_normalized_perimeter":cp, "nearest_normalized_perimeter":np_}
        rows.append(row); figure_payload.append((record,a,row))
        print(f"{index:03d}/141 {record['domain']} {record['subject']} Dice={proc['dice']:.5f}", flush=True)
    write_csv(OUT/"resampling_comparison_subject.csv",rows)
    summaries=summarize(rows);write_csv(OUT/"resampling_comparison_summary.csv",summaries)
    erosion=[]
    for domain in ("CAMRI","Mouse","Combined"):
        rr=rows if domain=="Combined" else [r for r in rows if r["domain"]==domain]
        erosion.append({"domain":domain,"subject_count":len(rr),"subjects_shrunk":sum(r["net_processed_voxel_change"]<0 for r in rr),
          "subjects_expanded":sum(r["net_processed_voxel_change"]>0 for r in rr),"total_lost":sum(r["processed_current_vs_nn_lost_voxels"] for r in rr),
          "total_added":sum(r["processed_current_vs_nn_added_voxels"] for r in rr),
          "pooled_lost_added_ratio":(sum(r["processed_current_vs_nn_lost_voxels"] for r in rr)/sum(r["processed_current_vs_nn_added_voxels"] for r in rr)
                                      if sum(r["processed_current_vs_nn_added_voxels"] for r in rr) else float("inf")),
          "mean_net_volume_change_percent":float(np.mean([r["processed_volume_change_percent"] for r in rr])),
          "median_net_volume_change_percent":float(np.median([r["processed_volume_change_percent"] for r in rr])),
          "mean_inward_shift_mm":float(np.mean([r["mean_inward_shift_mm"] for r in rr])),
          "mean_outward_shift_mm":float(np.mean([r["mean_outward_shift_mm"] for r in rr]))})
    write_csv(OUT/"erosion_bias_analysis.csv",erosion)
    for domain in ("CAMRI","Mouse"):
        candidates=[x for x in figure_payload if x[0]["domain"]==domain]
        candidates.sort(key=lambda x:x[2]["processed_current_vs_nn_dice"])
        chosen=[("worst",candidates[0]),("representative",candidates[len(candidates)//2])]
        for label,(record,a,row) in chosen:make_figure(record,a,row,OUT/"figures"/f"{domain.lower()}_{label}_{record['subject']}.png")
    documentation(rows,summaries)
    e={x["domain"]:x for x in erosion}; s={(x["domain"],x["scope"]):x for x in summaries}
    meaningful = abs(e["Combined"]["mean_net_volume_change_percent"]) >= .1 or s[("Combined","processed_full")]["mean_dice"] < .999
    report=f"""# Mask preprocessing audit\n\nThis diagnostic analyzed all 141 locked mixed-domain subjects (40 CAMRI, 101 Mouse); no model inference or training was run.\n\n## Findings\n\nThe current train/validation target is the original mask after transpose, image-nonzero crop, **image-mode order-1 resampling (`is_seg=False`)**, int8 truncation, `>0`, and center tiling. Validation uses the same cached representation. Canonical native test metrics reload the untouched expert NIfTI and therefore are not scored against this processed target.\n\n| Domain | Current-vs-NN mean Dice | Mean volume change | Shrunk subjects | Lost/added ratio | Native round-trip Dice (current) | Native round-trip Dice (NN) |\n|---|---:|---:|---:|---:|---:|---:|\n| CAMRI | {s[('CAMRI','processed_full')]['mean_dice']:.6f} | {e['CAMRI']['mean_net_volume_change_percent']:+.3f}% | {e['CAMRI']['subjects_shrunk']}/40 | {e['CAMRI']['pooled_lost_added_ratio']:.3f} | {s[('CAMRI','restored_native')]['mean_dice']:.6f} | {np.mean([r['native_original_vs_nn_dice'] for r in rows if r['domain']=='CAMRI']):.6f} |\n| Mouse | {s[('Mouse','processed_full')]['mean_dice']:.6f} | {e['Mouse']['mean_net_volume_change_percent']:+.3f}% | {e['Mouse']['subjects_shrunk']}/101 | {e['Mouse']['pooled_lost_added_ratio']:.3f} | {s[('Mouse','restored_native')]['mean_dice']:.6f} | {np.mean([r['native_original_vs_nn_dice'] for r in rows if r['domain']=='Mouse']):.6f} |\n\nThe distortion is classified as **{'meaningful' if meaningful else 'negligible'}** under the predeclared descriptive rule (absolute mean volume change at least 0.1% or mean processed Dice below 0.999). A pooled lost/added ratio above 1 and predominantly negative volume changes constitute evidence of inward/erosive bias. This makes preprocessing a plausible contributor to learned FN bias, but does not establish causation: the canonical test labels are untouched and model optimization/domain shift can independently produce FN errors.\n\n## Decision\n\nCorrect label-aware preprocessing in a new, controlled cache and retrain/evaluate the unchanged decoder **before** interpreting a full-resolution decoder experiment. Preserve the current checkpoint/results as the comparator; do not retroactively alter reported native test metrics. The next experiment should change only segmentation interpolation and compare validation/native-test FN balance.\n"""
    (OUT/"experiment_summary.md").write_text(report,encoding="utf-8")
    print(report)

if __name__ == "__main__": main()
