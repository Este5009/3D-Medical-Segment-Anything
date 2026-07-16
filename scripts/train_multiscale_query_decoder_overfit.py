#!/usr/bin/env python3
"""Controlled two-subject overfit test for the multi-scale one-query decoder."""

from __future__ import annotations

import copy
import argparse
import csv
import json
import random
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

from models.query_mask_decoder import (
    FrozenEncoderQueryModel,
    MultiScaleAttentionOneQueryMaskDecoder,
    MultiScaleOneQueryMaskDecoder,
    dice_bce_loss,
    volumetric_dice,
)
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import (
    choose_device,
    load_json,
    preprocess_pair,
    resolve_pairs,
    save_prediction_and_figure,
    verify_geometry,
)


def evaluate(model, samples, device):
    model.eval()
    results = []
    with torch.inference_mode():
        for sample in samples:
            features = {name: value.to(device) for name, value in sample["features"].items()}
            target = sample["mask"].to(device)
            logits = model.decode(features, output_size=target.shape[-3:])
            results.append((sample, logits.cpu(), volumetric_dice(logits, target)))
    return results


def save_learning_curve_svg(history, previous_csv: Path, destination: Path) -> None:
    """Write a dependency-free SVG comparison of training Dice curves."""
    previous = []
    with previous_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            previous.append((int(row["epoch"]), float(row["mean_dice"])))
    current = [(int(row["epoch"]), float(row["mean_dice"])) for row in history]
    width, height, margin = 760, 420, 55
    max_epoch = max(previous[-1][0], current[-1][0])

    def points(series):
        return " ".join(
            f"{margin + (epoch / max_epoch) * (width - 2 * margin):.1f},"
            f"{height - margin - dice * (height - 2 * margin):.1f}"
            for epoch, dice in series
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>
<polyline fill="none" stroke="#777" stroke-width="3" points="{points(previous)}"/>
<polyline fill="none" stroke="#0068b5" stroke-width="3" points="{points(current)}"/>
<text x="{width/2-30}" y="{height-12}" font-family="sans-serif">Epoch</text>
<text x="8" y="{height/2}" font-family="sans-serif" transform="rotate(-90 15 {height/2})">Training Dice</text>
<text x="{width-245}" y="30" fill="#777" font-family="sans-serif">previous decoder</text>
<text x="{width-245}" y="50" fill="#0068b5" font-family="sans-serif">multi-scale decoder</text>
</svg>'''
    destination.write_text(svg, encoding="utf-8")


def save_comparison(previous_png: Path, current_png: Path, destination: Path) -> None:
    import imageio.v3 as iio

    previous = iio.imread(previous_png)
    current = iio.imread(current_png)
    width = max(previous.shape[1], current.shape[1])
    separator = np.full((6, width), 255, dtype=np.uint8)
    if previous.shape[1] < width:
        previous = np.pad(previous, ((0, 0), (0, width - previous.shape[1])), constant_values=255)
    if current.shape[1] < width:
        current = np.pad(current, ((0, 0), (0, width - current.shape[1])), constant_values=255)
    iio.imwrite(destination, np.concatenate((previous, separator, current), axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/query_decoder_multiscale_overfit.yaml")
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    encoder_config = load_json(REPO_ROOT / config["encoder_config"])
    previous_summary = load_json(REPO_ROOT / config["previous_output_directory"] / "experiment_summary.json")
    paths = RS2NetPaths.from_config(encoder_config)
    output_dir = REPO_ROOT / config["output_directory"]
    previous_dir = REPO_ROOT / config["previous_output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device()
    tile_size = tuple(encoder_config["model"]["image_size"])
    encoder = RS2NetEncoderAdapter(paths, image_size=tile_size, in_channels=1, out_channels=1, feature_size=48)
    decoder_class = (
        MultiScaleAttentionOneQueryMaskDecoder
        if config.get("architecture_variant") == "attention_only"
        else MultiScaleOneQueryMaskDecoder
    )
    decoder = decoder_class(config["embedding_dim"], config["num_heads"])
    model = FrozenEncoderQueryModel(encoder, decoder).to(device)

    samples = []
    geometry = {}
    encoder_times = []
    for subject, image_path, mask_path in resolve_pairs(config, paths.dataset_root):
        geometry[subject] = verify_geometry(image_path, mask_path)
        image, mask, preprocessed_shape, _ = preprocess_pair(image_path, mask_path, paths, tile_size)
        started = time.perf_counter()
        encoded = model.encode(image.to(device))
        encoder_times.append(time.perf_counter() - started)
        selected = {name: encoded[name].detach().cpu() for name in ("level1", "level2", "level3", "level4")}
        samples.append({"subject": subject, "image": image, "mask": mask, "features": selected})
        geometry[subject].update({
            "image": str(image_path), "mask": str(mask_path),
            "preprocessed_shape": preprocessed_shape, "tile_shape": list(image.shape),
        })

    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    initial_dice = float(np.mean([result[2] for result in evaluate(model, samples, device)]))
    best_dice, best_epoch = initial_dice, 0
    best_state = copy.deepcopy(model.decoder.state_dict())
    stale_epochs = 0
    history = []
    training_started = time.perf_counter()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        losses = []
        for sample in samples:
            features = {name: value.to(device) for name, value in sample["features"].items()}
            target = sample["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.decode(features, output_size=target.shape[-3:])
            loss, _ = dice_bce_loss(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_dice = float(np.mean([result[2] for result in evaluate(model, samples, device)]))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "mean_dice": mean_dice}
        history.append(row)
        print(f"epoch={epoch:03d} loss={row['loss']:.6f} mean_dice={mean_dice:.6f}")
        if mean_dice > best_dice + float(config["minimum_improvement"]):
            best_dice, best_epoch = mean_dice, epoch
            best_state = copy.deepcopy(model.decoder.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(config["early_stop_patience"]):
            break

    torch.save({
        "decoder_state_dict": best_state, "config": config, "epoch": best_epoch, "dice": best_dice,
    }, output_dir / "best_checkpoint.pt")
    model.decoder.load_state_dict(best_state, strict=True)
    final_results = evaluate(model, samples, device)
    final_dice = float(np.mean([result[2] for result in final_results]))

    decoder_times = []
    with torch.inference_mode():
        for _ in range(5):
            for sample in samples:
                features = {name: value.to(device) for name, value in sample["features"].items()}
                started = time.perf_counter()
                model.decode(features, output_size=sample["mask"].shape[-3:])
                decoder_times.append(time.perf_counter() - started)

    failure_cases = {}
    for sample, logits, dice in final_results:
        prediction = logits.sigmoid() >= 0.5
        target = sample["mask"] >= 0.5
        false_positive = int((prediction & ~target).sum())
        false_negative = int((~prediction & target).sum())
        failure_cases[sample["subject"]] = {
            "dice": dice, "false_positive_voxels": false_positive, "false_negative_voxels": false_negative,
        }
        geometry[sample["subject"]]["final_dice"] = dice
        save_prediction_and_figure(output_dir, sample["subject"], sample["image"], sample["mask"], logits,
                                   spacing=(0.25, 0.20000000298, 0.15999999642))
        save_comparison(
            previous_dir / f"sub-{sample['subject']}_qualitative.png",
            output_dir / f"sub-{sample['subject']}_qualitative.png",
            output_dir / f"sub-{sample['subject']}_previous_vs_multiscale.png",
        )

    with (output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "mean_dice"])
        writer.writeheader(); writer.writerows(history)
    save_learning_curve_svg(history, previous_dir / "training_metrics.csv", output_dir / "learning_curves.svg")

    peak_memory = None
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    elif device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        peak_memory = torch.mps.driver_allocated_memory() / 1024**2
    elif sys.platform == "darwin":
        peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    summary = {
        "architecture": (
            "one query; four-scale cross-attention; independent level1 mask dot product"
            if config.get("architecture_variant") == "attention_only"
            else "one query; level4-to-level1 FPN; coarse-to-fine cross-attention; level1 mask dot product"
        ),
        "previous_trainable_parameters": previous_summary["trainable_parameters"],
        "trainable_parameters": model.trainable_parameter_count(),
        "device": str(device),
        "encoder_gradients_disabled": all(not p.requires_grad and p.grad is None for p in model.encoder.parameters()),
        "subjects": config["subjects"],
        "starting_mean_dice": initial_dice,
        "previous_final_mean_dice": previous_summary["final_mean_dice"],
        "final_mean_dice": final_dice,
        "absolute_dice_improvement": final_dice - previous_summary["final_mean_dice"],
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_by_plateau": len(history) < int(config["max_epochs"]),
        "mean_encoder_seconds": float(np.mean(encoder_times)),
        "mean_decoder_seconds": float(np.mean(decoder_times)),
        "estimated_end_to_end_seconds": float(np.mean(encoder_times) + np.mean(decoder_times)),
        "peak_memory_mib": peak_memory,
        "training_seconds_excluding_preprocessing_and_encoder": time.perf_counter() - training_started,
        "geometry": geometry,
        "failure_cases": failure_cases,
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
