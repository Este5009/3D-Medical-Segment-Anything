#!/usr/bin/env python3
"""Native paired evaluation for PaperWidthLevel0OneQueryMaskDecoder vs.
current best (outputs/higher_resolution_true_level0). Reuses the exact same
sliding-window/padding/export logic as every other experiment in this
family.
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
import torch
from scipy import ndimage

from analyze_boundary_error_diagnostics import binary_metrics
import evaluate_external_holdout as eeh  # noqa: monkeypatched below
from higher_resolution_preprocessing import preprocess_at_spacing_for_eval, HIGHER_RESOLUTION_SPACING
from models.query_mask_decoder import FrozenEncoderQueryModel, PaperWidthLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import load_json

BASELINE_OUT = ROOT / "outputs/higher_resolution_true_level0"
STRUCT = np.ones((3, 3, 3), bool)

eeh.FEATURE_NAMES = ("level0", "level1", "level2", "level3", "level4")
import evaluate_mouse_boundary_adaptation as emba  # noqa: E402
emba.preprocess = lambda image_path, mask_path, paths, tile_size: preprocess_at_spacing_for_eval(
    image_path, mask_path, paths, tile_size, spacing=HIGHER_RESOLUTION_SPACING)
from evaluate_mouse_boundary_adaptation import run_records  # noqa: E402


def rows(p):
    return list(csv.DictReader(open(p)))


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def largest(mask):
    lab, n = ndimage.label(mask, STRUCT)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n else np.zeros_like(mask)


def full_metrics(pred, gt, spacing):
    m = binary_metrics(pred, gt, spacing)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    return {**m, "iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "fp_fn_ratio": fp / max(fn, 1), "total_residual_error": fp + fn}


def main():
    assert eeh.FEATURE_NAMES == ("level0", "level1", "level2", "level3", "level4")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper_width_level0.yaml")
    args = ap.parse_args()
    config = load_json(ROOT / args.config)
    OUT = ROOT / config["output_directory"]
    emb = config["embedding_dim"]

    split = load_json(ROOT / "outputs/mixed_domain_anatomical_training/split.json")
    cam = {r["subject"]: r for r in rows(ROOT / config["camri_metrics"])}
    mouse = {r["scan_id"]: r for r in rows(ROOT / config["mouse_metrics"])}
    records = []
    for sid in split["camri"]["test"]:
        r = cam[sid]; records.append({"domain": "CAMRI", "subject": sid, "image_path": r["image_path"], "ground_truth_path": r["mask_path"]})
    for sid in split["mouse"]["test"]["scans"]:
        r = mouse[sid]; records.append({"domain": "Mouse", "subject": sid, "image_path": r["image_path"], "ground_truth_path": r["ground_truth_path"]})

    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT / "checkpoints/best_paper_width_level0_decoder.pt", map_location="cpu", weights_only=False)
    decoder = PaperWidthLevel0OneQueryMaskDecoder(emb, 4)
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

    baseline = {(r["domain"], r["subject"]): r for r in rows(BASELINE_OUT / "per_subject_comparison.csv")}
    comparison = []
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
            "baseline_raw_prediction_path": base["higher_res_raw_prediction_path"],
            "baseline_filtered_prediction_path": base["higher_res_filtered_prediction_path"],
            "baseline_probability_path": base["higher_res_probability_path"],
            "paper_width_raw_prediction_path": str(raw_path),
            "paper_width_filtered_prediction_path": str(filtered_path),
            "paper_width_probability_path": str(prob_path),
        }
        for name, p in (("baseline_raw", base["higher_res_raw_prediction_path"]), ("baseline_filtered", base["higher_res_filtered_prediction_path"]),
                        ("paper_width_raw", raw_path), ("paper_width_filtered", filtered_path)):
            item.update({f"{name}_{k}": v for k, v in full_metrics(np.asarray(nib.load(p).dataobj) > 0, gt, spacing).items()})
        comparison.append(item)
        print(f"{domain} {sid} baseline_dice={item['baseline_filtered_dice']:.5f} paper_width_dice={item['paper_width_filtered_dice']:.5f}", flush=True)

    write(OUT / "per_subject_comparison.csv", comparison)

    aggregate = []
    for d in ("CAMRI", "Mouse"):
        q = [x for x in comparison if x["domain"] == d]
        for c in ("baseline_filtered", "paper_width_filtered"):
            row = {"domain": d, "condition": c, "subjects": len(q), "embedding_dim": emb}
            for k in ("dice", "iou", "precision", "recall", "hd95_mm", "assd_mm", "surface_dice_0.1mm",
                      "surface_dice_0.2mm", "surface_dice_0.5mm", "surface_dice_1mm"):
                row[k] = float(np.mean([float(x[f"{c}_{k}"]) for x in q]))
            aggregate.append(row)
    write(OUT / "native_metrics.csv", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
