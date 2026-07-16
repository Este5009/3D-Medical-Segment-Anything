#!/usr/bin/env python3
"""Evaluate the saved best decoder without performing any optimization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch

from models.query_mask_decoder import FrozenEncoderQueryModel, OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import (
    choose_device,
    evaluate,
    load_json,
    preprocess_pair,
    resolve_pairs,
    save_prediction_and_figure,
    verify_geometry,
)


def main() -> None:
    config = load_json(REPO_ROOT / "configs/query_decoder_overfit.yaml")
    encoder_config = load_json(REPO_ROOT / config["encoder_config"])
    paths = RS2NetPaths.from_config(encoder_config)
    output_dir = REPO_ROOT / config["output_directory"]
    device = choose_device()
    tile_size = tuple(encoder_config["model"]["image_size"])

    encoder = RS2NetEncoderAdapter(paths, **{
        "image_size": tile_size,
        "in_channels": encoder_config["model"]["in_channels"],
        "out_channels": encoder_config["model"]["out_channels"],
        "feature_size": encoder_config["model"]["feature_size"],
    })
    model = FrozenEncoderQueryModel(
        encoder, OneQueryMaskDecoder(config["embedding_dim"], config["num_heads"])
    ).to(device)
    checkpoint = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)

    samples = []
    for subject, image_path, mask_path in resolve_pairs(config, paths.dataset_root):
        verify_geometry(image_path, mask_path)
        image, mask, _, _ = preprocess_pair(image_path, mask_path, paths, tile_size)
        features = model.encode(image.to(device))
        samples.append({
            "subject": subject,
            "image": image,
            "mask": mask,
            "features": {name: features[name].cpu() for name in ("level1", "level4")},
        })

    results = evaluate(model, samples, device)
    subject_dice = {}
    for sample, logits, dice in results:
        subject_dice[sample["subject"]] = dice
        save_prediction_and_figure(
            output_dir, sample["subject"], sample["image"], sample["mask"], logits,
            spacing=(0.25, 0.20000000298, 0.15999999642),
        )
    result = {
        "checkpoint": str(output_dir / "best_checkpoint.pt"),
        "device": str(device),
        "subject_dice": subject_dice,
        "mean_dice": sum(subject_dice.values()) / len(subject_dice),
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
