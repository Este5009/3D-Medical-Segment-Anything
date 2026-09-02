#!/usr/bin/env python3
"""UNETR-style full-width (level0_width=32) level0 decoder, retrained from
the SAME initialization as outputs/unetr_style_level0_full_width_decoder,
with richer, paper-style training augmentation (see
scripts/paper_style_augmentation.py) replacing this project's prior
feature-space flip+noise trick.

Motivation: outputs/unetr_style_level0_full_width_decoder/ improved Mouse
boundary metrics in aggregate (76/80 subjects, every metric), but visual
inspection found small, locally-confident false-positive "bulbs" escaping
the true contour on some subjects -- consistent with a higher-capacity
decoder (444,449 params) memorizing specific patterns from only 39 real
training images rather than learning boundary features that generalize.
Richer augmentation (rotation, zoom, Gaussian blur/noise, brightness/
contrast, gamma, simulated low-resolution -- the original RS2-Net paper's
own recipe) is the standard, literature-grounded remedy for exactly this.

Single isolated variable relative to outputs/unetr_style_level0_full_width_
decoder: training augmentation richness. Same initial_checkpoint (outputs/
higher_resolution_true_level0 -- NOT the just-trained, possibly-overfit
full-width checkpoint, so this cleanly tests "does richer augmentation
prevent the overfitting" rather than "can it correct an already-overfit
model"), same architecture (level0_width=32), same resolution, same loss,
same optimizer/learning rate/weight decay, same seed.

Mechanical consequence: spatial augmentation (rotation, zoom) must be
applied to the raw image before the frozen encoder, not to cached
post-encoder features -- so training can no longer use this project's
usual "cache encoder features once, train the decoder only" shortcut.
Training subjects are preprocessed once (deterministic, cached as raw
image/target pairs) but the frozen encoder now runs fresh every training
step, every epoch. Validation is unaffected (no augmentation, so its
existing cached-feature evaluation path is reused unchanged).
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
from models.query_mask_decoder import UnetrStyleLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from paper_style_augmentation import augment_pair, DEFAULT_SPEC
from train_generalization_pilot import load_cached
from train_mixed_domain_decoder import balanced_epoch_order, cached_records, training_loss
from train_mouse_boundary_adaptation import metric, aggregate
from train_query_decoder_overfit import load_json
from train_corrected_label_retraining import verification_gate
from verify_unetr_style_level0_full_width import verify as run_pretraining_gates, EXPECTED_NOT_TRANSFERRED, EXPECTED_TRANSFERRED_COUNT

LEVELS = ("level0", "level1", "level2", "level3", "level4")


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def prepare_validation_feature_cache(records, config, paths, device):
    """Unchanged from every prior experiment: validation is never augmented,
    so caching its post-encoder features once is exact and efficient."""
    cache = Path(config["corrected_cache"]); cache.mkdir(parents=True, exist_ok=True)
    tile_size = tuple(config["tile_size"])
    spacing = tuple(config["model_spacing_mm"])
    encoder = RS2NetEncoderAdapter(paths, image_size=tile_size, in_channels=1, out_channels=1, feature_size=48).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for domain in ("camri", "mouse"):
        for i, r in enumerate(records[domain]["validation"], 1):
            dest = cache / f"{domain}_validation_{r['subject']}.pt"
            r["cache_path"] = str(dest)
            if dest.exists():
                continue
            image, target, shape, _ = preprocess_image_and_corrected_target_at_spacing(
                Path(r["image_path"]), Path(r["mask_path"]), paths, spacing, tile_size)
            with torch.inference_mode():
                features = encoder(image.to(device))
            torch.save({
                "features": {k: features[k].cpu().half() for k in LEVELS},
                "target": target.byte(), "preprocessed_shape": shape,
                "label_interpolation": "nearest/order0/is_seg=True", "model_spacing_mm": list(spacing),
            }, dest)
            print(f"cache validation {domain} {i}/{len(records[domain]['validation'])}", flush=True)
    return encoder


def prepare_train_raw_cache(records, config, paths):
    """Training subjects: cache the RAW preprocessed (image, target) pair,
    not encoder features -- augmentation happens fresh each epoch, so the
    encoder must run fresh each epoch too."""
    cache = Path(config["raw_train_cache"]); cache.mkdir(parents=True, exist_ok=True)
    tile_size = tuple(config["tile_size"])
    spacing = tuple(config["model_spacing_mm"])
    for domain in ("camri", "mouse"):
        for i, r in enumerate(records[domain]["train"], 1):
            dest = cache / f"{domain}_train_{r['subject']}.pt"
            r["raw_cache_path"] = str(dest)
            if dest.exists():
                continue
            image, target, shape, _ = preprocess_image_and_corrected_target_at_spacing(
                Path(r["image_path"]), Path(r["mask_path"]), paths, spacing, tile_size)
            torch.save({"image": image.half(), "target": target.byte(), "preprocessed_shape": shape}, dest)
            print(f"cache raw train {domain} {i}/{len(records[domain]['train'])}", flush=True)


def load_raw_train(record):
    payload = torch.load(record["raw_cache_path"], map_location="cpu", weights_only=False)
    return payload["image"].float(), payload["target"].float()


@torch.inference_mode()
def evaluate(decoder, records, device):
    decoder.eval(); out = []
    for r in records:
        f, t = load_cached(r, device)
        logits = decoder(f, output_size=t.shape[-3:])
        out.append({"subject": r["subject"], **metric(logits, t)})
    return out


def initialize(decoder, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    old = payload["decoder_state_dict"]
    current = decoder.state_dict()
    shared = {k: v for k, v in old.items() if k in current and current[k].shape == v.shape}
    missing, unexpected = decoder.load_state_dict(shared, strict=False)
    not_transferred = set(old) - set(shared)
    if not_transferred != EXPECTED_NOT_TRANSFERRED:
        raise RuntimeError(f"Unexpected non-transferred tensor set: {not_transferred}")
    if len(shared) != EXPECTED_TRANSFERRED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_TRANSFERRED_COUNT} transferred tensors, got {len(shared)}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in new architecture: {unexpected}")
    return payload, missing, unexpected, shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unetr_style_level0_full_width_augmented.yaml")
    ap.add_argument("--skip-pretraining-gates", action="store_true")
    args = ap.parse_args()
    config = load_json(ROOT / args.config)
    tile_size = tuple(config["tile_size"])
    out = ROOT / config["output_directory"]
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "training").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    verification_gate(out)

    if not args.skip_pretraining_gates:
        gate_report = run_pretraining_gates(config.get("verify_config", "configs/unetr_style_level0_full_width.yaml"))
        if not gate_report.get("all_gates_passed"):
            raise RuntimeError("Pre-training verification gates failed; training aborted.")
        print("Pre-training verification gates passed; proceeding to training.\n", flush=True)

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    records = cached_records(config)

    encoder = prepare_validation_feature_cache(records, config, paths, device)
    prepare_train_raw_cache(records, config, paths)

    decoder = UnetrStyleLevel0OneQueryMaskDecoder(32, 4, level0_width=config["level0_width"])
    initial, missing, unexpected, shared = initialize(decoder, ROOT / config["initial_checkpoint"])
    decoder.to(device)
    parameters = sum(p.numel() for p in decoder.parameters())

    f, t = load_cached(records["camri"]["validation"][0], device)
    with_shape = decoder(f, output_size=t.shape[-3:])
    native = decoder(f)
    if native.shape[-3:] != tile_size or with_shape.shape != t.shape or not torch.equal(with_shape, native):
        raise RuntimeError("Level0 logits are not full-grid, or an interpolation branch fired unexpectedly")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    camri_ref = np.mean([float(r["dice"]) for r in csv.DictReader(open(ROOT / config["camri_metrics"])) if r["split"] == "validation"])
    camri_floor = camri_ref - config["camri_validation_max_drop"]
    best = -1; state = copy.deepcopy(decoder.state_dict()); best_epoch = 0; stale = 0; history = []
    safety_ineligible_epochs = []
    start = time.time(); peak = 0.
    aug_spec = {**DEFAULT_SPEC, **config.get("paper_style_augmentation", {})}

    for epoch in range(1, config["max_epochs"] + 1):
        decoder.train(); losses = []; epoch_start = time.time()
        for j, (domain, r) in enumerate(balanced_epoch_order(records["camri"]["train"], records["mouse"]["train"], config["seed"] + epoch)):
            image, target = load_raw_train(r)
            rng = random.Random(config["seed"] + epoch * 100 + j)
            image, target = augment_pair(image, target, rng, aug_spec)
            with torch.no_grad():
                features = encoder(image.to(device))
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = decoder(features, output_size=target.shape[-3:])
            loss, parts = training_loss(logits, target, config)
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        val = {d: aggregate(evaluate(decoder, records[d]["validation"], device)) for d in ("camri", "mouse")}
        score = (val["camri"]["dice"] + val["mouse"]["dice"]) / 2
        elapsed = time.time() - epoch_start; peak = max(peak, peak_mib())
        camri_safety_eligible = val["camri"]["dice"] >= camri_floor
        row = {"epoch": epoch, "loss": np.mean(losses), "epoch_seconds": elapsed, "balanced_validation_dice": score,
               "camri_safety_eligible": camri_safety_eligible,
               **{f"{d}_validation_{k}": v for d in val for k, v in val[d].items()}}
        history.append(row)
        print(f"epoch {epoch} CAMRI={val['camri']['dice']:.6f} Mouse={val['mouse']['dice']:.6f} balanced={score:.6f} "
              f"seconds={elapsed:.1f} camri_safety_eligible={camri_safety_eligible}", flush=True)
        if not camri_safety_eligible:
            safety_ineligible_epochs.append(epoch)
            continue
        if score > best + config["minimum_validation_improvement"]:
            best = score; state = copy.deepcopy(decoder.state_dict()); best_epoch = epoch; stale = 0
        else:
            stale += 1
        if stale >= config["early_stop_patience"]:
            break
    safety = best == -1

    torch.save({
        "decoder_state_dict": state, "epoch": best_epoch, "balanced_validation_dice": best,
        "initial_checkpoint": config["initial_checkpoint"], "config": config,
    }, out / "checkpoints/best_unetr_style_level0_full_width_augmented_decoder.pt")
    with (out / "training/history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0]); w.writeheader(); w.writerows(history)

    summary = {
        "device": str(device), "selected_epoch": best_epoch, "epochs_run": len(history),
        "best_balanced_validation_dice": best, "encoder_frozen": True, "one_query": True,
        "levels": list(LEVELS), "unetr_style_decoder_parameters": parameters,
        "level0_width": config["level0_width"], "tile_size": list(tile_size),
        "model_spacing_mm": config["model_spacing_mm"],
        "architecture_change": "none -- identical to outputs/unetr_style_level0_full_width_decoder. Sole variable: paper-style training augmentation (rotation/zoom/blur/noise/contrast/gamma/simulated-low-res) applied to raw images before the frozen encoder, replacing the prior feature-space flip+noise trick.",
        "augmentation_spec": aug_spec,
        "native_logits_shape": [1, 1, *tile_size], "final_logit_interpolation": False,
        "query_conditioning_at_level0": True, "level0_query_tokens_pooled_2x": True,
        "camri_safety_stop": safety, "camri_floor": camri_floor, "camri_reference": camri_ref,
        "safety_ineligible_epochs": safety_ineligible_epochs,
        "camri_safety_rule_semantics": "per-epoch selection eligibility filter, not a training-abort trigger",
        "mean_epoch_seconds": float(np.mean([r["epoch_seconds"] for r in history])),
        "peak_process_memory_mib": peak, "elapsed_seconds": time.time() - start,
        "transferred_baseline_tensors": len(shared), "new_parameter_keys": list(missing),
        "unexpected_keys": list(unexpected), "initial_checkpoint": config["initial_checkpoint"],
    }
    (out / "training/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
