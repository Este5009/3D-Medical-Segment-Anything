#!/usr/bin/env python3
"""Lesion-only single-query control for the two-query generalization
experiment. See configs/stroke_lesion_only.yaml docstring for the full
rationale -- this exists purely as the single-task baseline that
TwoTaskLevel0OneQueryMaskDecoder's joint lesion performance is compared
against, not as a test of query-conditioning itself.

Resilience: writes checkpoint + history to disk after every epoch (not only
at early-stop/completion), matching the practice adopted after tonight's
pod-interruption scare -- a background sync loop pulling this output
directory never has more than one epoch's work to lose.
"""
from __future__ import annotations
import argparse, copy, csv, json, random, resource, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import numpy as np
import torch

from higher_resolution_preprocessing import preprocess_image_and_corrected_target_at_spacing
from models.query_mask_decoder import PaperWidthLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_mixed_domain_decoder import training_loss
from train_mouse_boundary_adaptation import metric, aggregate
from train_query_decoder_overfit import load_json

LEVELS = ("level0", "level1", "level2", "level3", "level4")


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def stroke_records(config):
    split = load_json(ROOT / config["stroke_split"])
    raw = Path(config["stroke_raw_root"])
    out = {}
    for name in ("train", "validation", "test"):
        out[name] = [{"subject": sid, "image_path": str(raw / sid / "t2.nii"), "mask_path": str(raw / sid / "masklesion_manual.nii")}
                     for sid in split[name]]
    return out


def prepare_cache(records, config, paths, device):
    cache = Path(config["stroke_cache"]); cache.mkdir(parents=True, exist_ok=True)
    tile_size = tuple(config["tile_size"])
    spacing = tuple(config["model_spacing_mm"])
    encoder = RS2NetEncoderAdapter(paths, image_size=tile_size, in_channels=1, out_channels=1, feature_size=48).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for split in ("train", "validation"):
        for i, r in enumerate(records[split], 1):
            dest = cache / f"stroke_{split}_{r['subject']}.pt"
            r["cache_path"] = str(dest)
            if dest.exists():
                continue
            image, target, shape, _ = preprocess_image_and_corrected_target_at_spacing(
                Path(r["image_path"]), Path(r["mask_path"]), paths, spacing, tile_size)
            with torch.inference_mode():
                features = encoder(image.to(device))
            torch.save({
                "features": {k: features[k].cpu().half() for k in LEVELS},
                "target": target.byte(),
            }, dest)
            print(f"cache stroke {split} {i}/{len(records[split])}", flush=True)
    del encoder


def load_cached(record, device):
    payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
    features = {k: payload["features"][k].to(device).float() for k in LEVELS}
    target = payload["target"].to(device).float()
    return features, target


def augment_lesion(features, target, rng, config):
    a = config["augmentation"]
    features = {k: v.clone() for k, v in features.items()}
    if rng.random() < a["flip_probability"]:
        features = {k: torch.flip(v, dims=[-1]) for k, v in features.items()}
        target = torch.flip(target, dims=[-1])
    scale = 1.0 + (rng.random() * 2 - 1) * a["feature_scale"]
    features = {k: v * scale for k, v in features.items()}
    for k in features:
        features[k] = features[k] + torch.randn_like(features[k]) * a["feature_noise_std"]
    return features, target


@torch.inference_mode()
def evaluate(decoder, records, device):
    decoder.eval(); out = []
    for r in records:
        f, t = load_cached(r, device)
        logits = decoder(f, output_size=t.shape[-3:])
        out.append({"subject": r["subject"], **metric(logits, t)})
    return out


def save_checkpoint(path, state, epoch, best, config, parameters):
    torch.save({
        "decoder_state_dict": state, "epoch": epoch, "validation_dice": best,
        "embedding_dim": config["embedding_dim"], "decoder_parameters": parameters, "config": config,
    }, path)


def write_history(path, history):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stroke_lesion_only.yaml")
    args = ap.parse_args()
    config = load_json(ROOT / args.config)
    tile_size = tuple(config["tile_size"])
    out = ROOT / config["output_directory"]
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "training").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    records = stroke_records(config)
    prepare_cache(records, config, paths, device)
    print(f"stroke records: train={len(records['train'])} validation={len(records['validation'])} test={len(records['test'])}", flush=True)

    decoder = PaperWidthLevel0OneQueryMaskDecoder(config["embedding_dim"], 4).to(device)
    parameters = sum(p.numel() for p in decoder.parameters())
    print(f"PaperWidthLevel0OneQueryMaskDecoder (lesion-only) parameters: {parameters}", flush=True)

    f, t = load_cached(records["validation"][0], device)
    with torch.no_grad():
        with_shape = decoder(f, output_size=t.shape[-3:])
        native = decoder(f)
    if native.shape[-3:] != tile_size or with_shape.shape != t.shape or not torch.equal(with_shape, native):
        raise RuntimeError("Level0 logits are not full-grid, or an interpolation branch fired unexpectedly")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best = -1; state = copy.deepcopy(decoder.state_dict()); best_epoch = 0; stale = 0; history = []
    start = time.time(); peak = 0.
    checkpoint_path = out / "checkpoints/best_stroke_lesion_only_decoder.pt"
    latest_path = out / "checkpoints/latest_stroke_lesion_only_decoder.pt"
    history_path = out / "training/history.csv"

    for epoch in range(1, config["max_epochs"] + 1):
        decoder.train(); losses = []; epoch_start = time.time()
        order = list(range(len(records["train"]))); random.Random(config["seed"] + epoch).shuffle(order)
        for j, idx in enumerate(order):
            r = records["train"][idx]
            features, target = load_cached(r, device)
            features, target = augment_lesion(features, target, random.Random(config["seed"] + epoch * 100 + j), config)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = decoder(features, output_size=target.shape[-3:])
            loss, parts = training_loss(logits.float(), target, config)
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        val = aggregate(evaluate(decoder, records["validation"], device))
        elapsed = time.time() - epoch_start; peak = max(peak, peak_mib())
        row = {"epoch": epoch, "loss": np.mean(losses), "epoch_seconds": elapsed,
               **{f"validation_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"epoch {epoch} lesion_dice={val['dice']:.6f} seconds={elapsed:.1f}", flush=True)

        write_history(history_path, history)
        save_checkpoint(latest_path, decoder.state_dict(), epoch, val["dice"], config, parameters)

        if val["dice"] > best + config["minimum_validation_improvement"]:
            best = val["dice"]; state = copy.deepcopy(decoder.state_dict()); best_epoch = epoch; stale = 0
            save_checkpoint(checkpoint_path, state, best_epoch, best, config, parameters)
        else:
            stale += 1
        if stale >= config["early_stop_patience"]:
            break

    summary = {
        "device": str(device), "selected_epoch": best_epoch, "epochs_run": len(history),
        "best_validation_dice": best, "encoder_frozen": True, "one_query": True, "task": "lesion_only",
        "levels": list(LEVELS), "decoder_parameters": parameters,
        "embedding_dim": config["embedding_dim"], "tile_size": list(tile_size),
        "model_spacing_mm": config["model_spacing_mm"], "architecture_change": config["architecture_change"],
        "records": {k: len(v) for k, v in records.items()},
        "mean_epoch_seconds": float(np.mean([r["epoch_seconds"] for r in history])),
        "peak_process_memory_mib": peak, "elapsed_seconds": time.time() - start,
    }
    (out / "training/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
