#!/usr/bin/env python3
"""Native-space test evaluation for TwoTaskLevel0OneQueryMaskDecoder --
BOTH tasks, using the SAME trained checkpoint (one encoder + one decoder,
two task-fixed wrapper views). This produces the numbers the whole
experiment exists to compare: this model's brain Dice against the
single-task brain decoder (outputs/paper_width_level0_decoder), and this
model's lesion Dice against the single-task lesion decoder
(outputs/stroke_lesion_only_decoder).
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

import evaluate_external_holdout as eeh
eeh.FEATURE_NAMES = ("level0", "level1", "level2", "level3", "level4")
from higher_resolution_preprocessing import preprocess_at_spacing_for_eval, HIGHER_RESOLUTION_SPACING
from models.query_mask_decoder import TaskFixedFrozenEncoderQueryModel, TwoTaskLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import load_json
import evaluate_mouse_boundary_adaptation as emba
emba.preprocess = lambda image_path, mask_path, paths, tile_size: preprocess_at_spacing_for_eval(
    image_path, mask_path, paths, tile_size, spacing=HIGHER_RESOLUTION_SPACING)
from evaluate_mouse_boundary_adaptation import run_records

STRUCT = np.ones((3, 3, 3), bool)


def rows(p):
    return list(csv.DictReader(open(p)))


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def largest(mask):
    lab, n = ndimage.label(mask, STRUCT)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n else np.zeros_like(mask)


def main():
    assert eeh.FEATURE_NAMES == ("level0", "level1", "level2", "level3", "level4")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_task_joint.yaml")
    args = ap.parse_args()
    config = load_json(ROOT / args.config)
    OUT = ROOT / config["output_directory"]
    emb = config["embedding_dim"]

    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT / "checkpoints/best_two_task_joint_decoder.pt", map_location="cpu", weights_only=False)
    decoder = TwoTaskLevel0OneQueryMaskDecoder(emb, 4)
    decoder.load_state_dict(ck["decoder_state_dict"], strict=True)
    encoder = RS2NetEncoderAdapter(paths, image_size=tuple(config["tile_size"]), in_channels=1, out_channels=1, feature_size=48)
    brain_model = TaskFixedFrozenEncoderQueryModel(encoder, decoder, "brain").to(device).eval()
    lesion_model = TaskFixedFrozenEncoderQueryModel(encoder, decoder, "lesion").to(device).eval()

    # --- brain side (CAMRI + Mouse test sets, filtered to largest CC like every other brain-family evaluation) ---
    split = load_json(ROOT / "outputs/mixed_domain_anatomical_training/split.json")
    cam = {r["subject"]: r for r in rows(ROOT / config["camri_metrics"])}
    mouse = {r["scan_id"]: r for r in rows(ROOT / config["mouse_metrics"])}
    brain_records = []
    for sid in split["camri"]["test"]:
        r = cam[sid]; brain_records.append({"domain": "CAMRI", "subject": sid, "scan_id": sid, "image_path": r["image_path"], "ground_truth_path": r["mask_path"]})
    for sid in split["mouse"]["test"]["scans"]:
        r = mouse[sid]; brain_records.append({"domain": "Mouse", "subject": sid, "scan_id": sid, "image_path": r["image_path"], "ground_truth_path": r["ground_truth_path"]})

    brain_native = []
    for domain in ("CAMRI", "Mouse"):
        selected = [r for r in brain_records if r["domain"] == domain]
        rs, _ = run_records(brain_model, selected, paths, config, OUT, device, f"brain_{domain.lower()}")
        brain_native.extend(rs)
    for r in brain_native:
        raw = np.asarray(nib.load(r["prediction_path"]).dataobj) > 0
        filt = largest(raw)
        gt = np.asarray(nib.load(r["ground_truth_path"]).dataobj) > 0
        tp = int((filt & gt).sum()); fp = int((filt & ~gt).sum()); fn = int((~filt & gt).sum())
        r["filtered_dice"] = 2 * tp / max(2 * tp + fp + fn, 1)
    write(OUT / "brain_native_metrics_per_subject.csv", brain_native)

    # --- lesion side (stroke test set, no CC filtering -- lesions can be multi-focal) ---
    stroke_split = load_json(ROOT / config["stroke_split"])
    raw_root = Path(config["stroke_raw_root"])
    lesion_records = [{"scan_id": sid, "image_path": str(raw_root / sid / "t2.nii"), "ground_truth_path": str(raw_root / sid / "masklesion_manual.nii")}
                       for sid in stroke_split["test"]]
    lesion_native, _ = run_records(lesion_model, lesion_records, paths, config, OUT, device, "lesion")
    write(OUT / "lesion_native_metrics_per_subject.csv", lesion_native)

    baseline_brain = {"CAMRI": [r for r in rows(ROOT / "outputs/paper_width_level0_decoder/native_metrics.csv")
                                 if r["domain"] == "CAMRI" and r["condition"] == "paper_width_filtered"][0],
                       "Mouse": [r for r in rows(ROOT / "outputs/paper_width_level0_decoder/native_metrics.csv")
                                 if r["domain"] == "Mouse" and r["condition"] == "paper_width_filtered"][0]}
    baseline_lesion = rows(ROOT / "outputs/stroke_lesion_only_decoder/native_metrics_aggregate.csv")[0]

    summary = []
    for domain in ("CAMRI", "Mouse"):
        q = [r for r in brain_native if r["domain"] == domain]
        joint_dice = float(np.mean([r["filtered_dice"] for r in q]))
        summary.append({"task": "brain", "domain": domain, "subjects": len(q),
                         "joint_model_dice": joint_dice, "single_task_model_dice": float(baseline_brain[domain]["dice"]),
                         "difference": joint_dice - float(baseline_brain[domain]["dice"])})
    joint_lesion_dice = float(np.mean([float(r["dice"]) for r in lesion_native]))
    summary.append({"task": "lesion", "domain": "stroke", "subjects": len(lesion_native),
                     "joint_model_dice": joint_lesion_dice, "single_task_model_dice": float(baseline_lesion["dice_mean"]),
                     "difference": joint_lesion_dice - float(baseline_lesion["dice_mean"])})
    write(OUT / "joint_vs_single_task_comparison.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
