#!/usr/bin/env python3
"""Native paired evaluation of the TRUE level0 decoder checkpoint.

Per the audit at outputs/full_resolution_level0_decoder/implementation_audit.md
(section 7, confounder #3), the earlier level0 experiment's native evaluation
used a locally reimplemented sliding-window function with different (and
uncontrolled) edge-padding behavior versus the corrected-label baseline's
evaluation path. This script does not reimplement tiling. It imports and
calls the *exact same* `sliding_window_logits` function object that
`evaluate_corrected_label_retraining.py` uses (via `run_records` in
`evaluate_mouse_boundary_adaptation.py`, which itself imports it unchanged
from `evaluate_external_holdout.py`). The only difference at test time is
which encoder feature levels get sliced into the dict handed to
`model.decode` -- controlled by monkeypatching the shared module's
`FEATURE_NAMES` constant to include `level0` before calling. Tiling, stride,
padding (whole-volume, centered/symmetric), and overlap-averaging logic are
byte-identical to the baseline's own evaluation.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from analyze_boundary_error_diagnostics import binary_metrics
import evaluate_external_holdout as eeh  # noqa: monkeypatched below, not called directly
from models.query_mask_decoder import FrozenEncoderQueryModel, TrueFullResolutionLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import choose_device, load_json

OUT = ROOT / "outputs/true_full_resolution_level0_decoder"
STRUCT = np.ones((3, 3, 3), bool)

# The one and only sanctioned difference from the baseline's evaluation path:
# request level0 features in addition to levels1-4. `sliding_window_logits`
# and `run_records` are otherwise called completely unmodified.
eeh.FEATURE_NAMES = ("level0", "level1", "level2", "level3", "level4")
from evaluate_mouse_boundary_adaptation import run_records  # noqa: E402  (import after monkeypatch is intentional)


def rows(p):
    return list(csv.DictReader(open(p)))


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)


def largest(mask):
    lab, n = ndimage.label(mask, STRUCT)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n else np.zeros_like(mask)


def full_metrics(pred, gt, spacing):
    m = binary_metrics(pred, gt, spacing)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    return {**m, "iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "fp_fn_ratio": fp / max(fn, 1), "total_residual_error": fp + fn}


def main():
    assert eeh.FEATURE_NAMES == ("level0", "level1", "level2", "level3", "level4"), "monkeypatch did not take effect"

    config = load_json(ROOT / "configs/true_full_resolution_level0_decoder.yaml")
    split = load_json(ROOT / "outputs/mixed_domain_anatomical_training/split.json")
    cam = {r["subject"]: r for r in rows(ROOT / config["camri_metrics"])}
    mouse = {r["scan_id"]: r for r in rows(ROOT / config["mouse_metrics"])}
    records = []
    for sid in split["camri"]["test"]:
        r = cam[sid]; records.append({"domain": "CAMRI", "subject": sid, "image_path": r["image_path"], "ground_truth_path": r["mask_path"]})
    for sid in split["mouse"]["test"]["scans"]:
        r = mouse[sid]; records.append({"domain": "Mouse", "subject": sid, "image_path": r["image_path"], "ground_truth_path": r["ground_truth_path"]})

    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    device = torch.device("cpu")  # MPS lacks Conv3d support in the installed torch 2.0.0; forced CPU, matching baseline evaluation.
    ck = torch.load(OUT / "checkpoints/best_true_level0_decoder.pt", map_location="cpu", weights_only=False)
    decoder = TrueFullResolutionLevel0OneQueryMaskDecoder(32, 4, level0_width=config["level0_width"])
    decoder.load_state_dict(ck["decoder_state_dict"], strict=True)
    model = FrozenEncoderQueryModel(
        RS2NetEncoderAdapter(paths, image_size=tuple(config["tile_size"]), in_channels=1, out_channels=1, feature_size=48),
        decoder,
    ).to(device).eval()

    native = []
    for domain in ("CAMRI", "Mouse"):
        selected = [r for r in records if r["domain"] == domain]
        rs, _ = run_records(model, selected, paths, config, OUT, device, domain.lower())
        native.extend(rs)

    baseline = {(r["domain"], r["subject"]): r for r in rows(ROOT / "outputs/corrected_label_retraining/per_subject_test_metrics.csv")}
    comparison, source = [], []
    for r in records:
        sid = r["subject"]; domain = r["domain"]
        raw_path = OUT / "native_predictions" / domain.lower() / f"{sid}_prediction.nii.gz"
        prob_path = OUT / "probability_maps" / domain.lower() / f"{sid}_probability.nii.gz"
        obj = nib.load(r["ground_truth_path"]); gt = np.asarray(obj.dataobj) > 0
        raw = np.asarray(nib.load(raw_path).dataobj) > 0
        filt = largest(raw)
        filtered_path = OUT / "filtered_predictions" / domain.lower() / f"{sid}_prediction.nii.gz"
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(filt.astype(np.uint8), obj.affine, obj.header), filtered_path)
        spacing = tuple(map(float, obj.header.get_zooms()[:3]))
        base = baseline[(domain, sid)]
        item = {
            "domain": domain, "subject": sid, "image_path": r["image_path"], "ground_truth_path": r["ground_truth_path"],
            "baseline_raw_prediction_path": base["new_raw_prediction_path"],
            "baseline_filtered_prediction_path": base["new_filtered_prediction_path"],
            "baseline_probability_path": base["new_probability_path"],
            "level0_raw_prediction_path": str(raw_path),
            "level0_filtered_prediction_path": str(filtered_path),
            "level0_probability_path": str(prob_path),
        }
        for name, p in (("baseline_raw", base["new_raw_prediction_path"]), ("baseline_filtered", base["new_filtered_prediction_path"]),
                        ("level0_raw", raw_path), ("level0_filtered", filtered_path)):
            item.update({f"{name}_{k}": v for k, v in full_metrics(np.asarray(nib.load(p).dataobj) > 0, gt, spacing).items()})
        comparison.append(item)
        source.append({
            "domain": domain, "subject": sid,
            "baseline_dice": item["level0_raw_dice"], "filtered_dice": item["level0_filtered_dice"],
            "dice_change": item["level0_filtered_dice"] - item["level0_raw_dice"],
            "image_path": r["image_path"], "ground_truth_path": r["ground_truth_path"],
            "baseline_prediction_path": str(raw_path), "filtered_prediction_path": str(filtered_path),
        })
        print(f"{domain} {sid} baseline_dice={item['baseline_filtered_dice']:.5f} level0_dice={item['level0_filtered_dice']:.5f}", flush=True)

    write(OUT / "per_subject_comparison.csv", comparison)
    write(OUT / "diagnostic_source.csv", source)

    aggregate = []
    for d in ("CAMRI", "Mouse"):
        q = [x for x in comparison if x["domain"] == d]
        for c in ("baseline_raw", "baseline_filtered", "level0_raw", "level0_filtered"):
            row = {"domain": d, "condition": c, "subjects": len(q)}
            for k in ("dice", "iou", "precision", "recall", "hd95_mm", "assd_mm", "surface_dice_0.1mm",
                      "surface_dice_0.2mm", "surface_dice_0.5mm", "surface_dice_1mm", "fp_voxels", "fn_voxels",
                      "fp_fn_ratio", "total_residual_error"):
                row[k] = (sum(int(x[f"{c}_{k}"]) for x in q) if k in ("fp_voxels", "fn_voxels", "total_residual_error")
                          else float(np.mean([float(x[f"{c}_{k}"]) for x in q])))
            aggregate.append(row)
    write(OUT / "native_metrics.csv", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
