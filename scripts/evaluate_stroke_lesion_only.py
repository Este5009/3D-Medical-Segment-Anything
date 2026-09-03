#!/usr/bin/env python3
"""Native-space test evaluation for the lesion-only single-query control.
No largest-connected-component filtering: ischemic lesions can be
genuinely multi-focal, so raw thresholded predictions are evaluated as-is,
unlike the brain-segmentation family's single-component assumption.
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

import evaluate_external_holdout as eeh
eeh.FEATURE_NAMES = ("level0", "level1", "level2", "level3", "level4")
from higher_resolution_preprocessing import preprocess_at_spacing_for_eval, HIGHER_RESOLUTION_SPACING
from models.query_mask_decoder import FrozenEncoderQueryModel, PaperWidthLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import load_json
import evaluate_mouse_boundary_adaptation as emba
emba.preprocess = lambda image_path, mask_path, paths, tile_size: preprocess_at_spacing_for_eval(
    image_path, mask_path, paths, tile_size, spacing=HIGHER_RESOLUTION_SPACING)
from evaluate_mouse_boundary_adaptation import run_records


def rows(p):
    return list(csv.DictReader(open(p)))


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def main():
    assert eeh.FEATURE_NAMES == ("level0", "level1", "level2", "level3", "level4")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stroke_lesion_only.yaml")
    args = ap.parse_args()
    config = load_json(ROOT / args.config)
    OUT = ROOT / config["output_directory"]
    emb = config["embedding_dim"]

    split = load_json(ROOT / config["stroke_split"])
    raw = Path(config["stroke_raw_root"])
    records = [{"scan_id": sid, "image_path": str(raw / sid / "t2.nii"), "ground_truth_path": str(raw / sid / "masklesion_manual.nii")}
               for sid in split["test"]]

    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT / "checkpoints/best_stroke_lesion_only_decoder.pt", map_location="cpu", weights_only=False)
    decoder = PaperWidthLevel0OneQueryMaskDecoder(emb, 4)
    decoder.load_state_dict(ck["decoder_state_dict"], strict=True)
    model = FrozenEncoderQueryModel(
        RS2NetEncoderAdapter(paths, image_size=tuple(config["tile_size"]), in_channels=1, out_channels=1, feature_size=48),
        decoder,
    ).to(device).eval()

    native, _ = run_records(model, records, paths, config, OUT, device, "lesion")
    write(OUT / "native_metrics_per_subject.csv", native)

    dice = [float(r["dice"]) for r in native]
    aggregate = {"domain": "stroke_lesion", "condition": "raw_unfiltered", "subjects": len(native),
                 "dice_mean": float(np.mean(dice)), "dice_std": float(np.std(dice)), "dice_median": float(np.median(dice))}
    for k in ("hd95_mm", "assd_mm") if "hd95_mm" in native[0] else ():
        aggregate[k] = float(np.mean([float(r[k]) for r in native]))
    write(OUT / "native_metrics_aggregate.csv", [aggregate])
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
