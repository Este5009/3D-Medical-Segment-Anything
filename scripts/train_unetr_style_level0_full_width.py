#!/usr/bin/env python3
"""UNETR-style, real-skip-connection full-resolution level0 decoder --
controlled experiment.

Controlled intervention: the *only* change relative to
outputs/higher_resolution_true_level0 is decoder depth/architecture at and
near level0. TrueFullResolutionLevel0OneQueryMaskDecoder's single
depthwise+pointwise level0 fusion pass (effective receptive field ~3
model-grid voxels) is replaced by UnetrStyleLevel0OneQueryMaskDecoder's real
four-stage skip-connected residual upsampling chain, level4->3->2->1->0,
using monai.networks.blocks.UnetrUpBlock unmodified -- the exact block class
the original RS2-Net decoder itself is built from. Motivated by
outputs/level0_depth_diagnostic/ (residual error is small, scattered
1-4-voxel clusters and NOT concentrated on the untouched native-X axis --
consistent with a decoder local-context shortfall, not a resolution
shortfall). Resolution (192x128x160, model_spacing_mm unchanged), the
encoder, all hyperparameters, loss, and augmentation are identical to that
checkpoint's own training run -- a single isolated architectural variable.

Initialization: outputs/higher_resolution_true_level0/checkpoints/
best_higher_resolution_true_level0.pt (the current best model). Unlike the
prior two level0 experiments, this decoder is NOT spatially shape-agnostic
end to end: 75 of the old checkpoint's 102 decoder tensors transfer exactly
by name+shape (query, mask_embedding, mask_bias, projections.level1/2/3/4,
query_updates.level4/level3/level2/level1); the other 27 were either the
old decoder's now-removed level0-only modules, or (level0_query_projection.
weight, level0_embedding_projection.weight/bias) share a name with the new
architecture but differ in shape because the old checkpoint's level0_width
is 16 and this decoder's is 8 -- all correctly left behind, not corrupted.
The new up_blocks (real UnetrUpBlock residual convolutions), projections.
level0, and query_updates.level0 have no prior-checkpoint counterpart at
all and start at random initialization -- verified exactly by
verify_unetr_style_level0.py before this script is allowed to run.

level0_width=8 (not this project's prior level0_width=16 precedent) was
chosen from direct on-machine profiling during scoping, not by default or
guess: a single dense UnetrUpBlock(in=32,out=W) forward+backward pass at the
level0 grid measured 3.05s/4.51s/8.99s/13.89s at W=6/8/10/12, and did not
complete a single forward pass in >8 minutes at W=32 (the CPU Conv3d cost on
this machine is a sharp, non-linear function of channel count at this grid
size, not the smooth one initially assumed). level0_width=16 -- fine for the
old decoder's single depthwise+pointwise pass -- measured 206.71s for one
full training step of this real-residual-block design, infeasible for an
overnight run; level0_width=8 measured 7.59-10.62s/step, tractable.
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
from train_generalization_pilot import load_cached
from train_mixed_domain_decoder import balanced_epoch_order, cached_records, training_loss
from train_mouse_boundary_adaptation import augment, metric, aggregate
from train_query_decoder_overfit import load_json
from train_corrected_label_retraining import verification_gate
from verify_unetr_style_level0_full_width import verify as run_pretraining_gates, EXPECTED_NOT_TRANSFERRED, EXPECTED_TRANSFERRED_COUNT

LEVELS = ("level0", "level1", "level2", "level3", "level4")
BASELINE_DECODER_PARAMETERS = 355889  # outputs/higher_resolution_true_level0's TrueFullResolutionLevel0OneQueryMaskDecoder param count is 182081; this is UnetrStyleLevel0OneQueryMaskDecoder(level0_width=8)'s own param count, recorded post-hoc below for the real run.


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def prepare_cache(records, config, paths, device):
    cache = Path(config["corrected_cache"]); cache.mkdir(parents=True, exist_ok=True)
    tile_size = tuple(config["tile_size"])
    spacing = tuple(config["model_spacing_mm"])
    encoder = RS2NetEncoderAdapter(paths, image_size=tile_size, in_channels=1, out_channels=1, feature_size=48).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for domain in ("camri", "mouse"):
        for split in ("train", "validation"):
            for i, r in enumerate(records[domain][split], 1):
                dest = cache / f"{domain}_{split}_{r['subject']}.pt"
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
                    "preprocessed_shape": shape,
                    "label_interpolation": "nearest/order0/is_seg=True",
                    "model_spacing_mm": list(spacing),
                }, dest)
                print(f"cache {domain} {split} {i}/{len(records[domain][split])}", flush=True)
    del encoder


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
    ap.add_argument("--config", default="configs/unetr_style_level0_full_width.yaml")
    ap.add_argument("--skip-pretraining-gates", action="store_true")
    args = ap.parse_args()
    config = load_json(ROOT / args.config)
    tile_size = tuple(config["tile_size"])
    out = ROOT / config["output_directory"]
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "training").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    verification_gate(out)  # Same corrected-label-quality gate every run in this family uses; resolution/architecture-independent.

    if not args.skip_pretraining_gates:
        gate_report = run_pretraining_gates(args.config)
        if not gate_report.get("all_gates_passed"):
            raise RuntimeError("Pre-training verification gates failed; training aborted.")
        print("Pre-training verification gates passed; proceeding to training.\n", flush=True)

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    # MPS lacks Conv3d support in the installed torch 2.0.0, so CPU was this
    # project's only local option; a CUDA GPU (e.g. a rented pod) is strongly
    # preferred when present -- measured ~15x faster per training step for
    # this exact decoder (0.63s/step GPU vs 7.59-10.62s/step CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    records = cached_records(config)
    prepare_cache(records, config, paths, device)

    decoder = UnetrStyleLevel0OneQueryMaskDecoder(32, 4, level0_width=config["level0_width"])
    initial, missing, unexpected, shared = initialize(decoder, ROOT / config["initial_checkpoint"])
    decoder.to(device)
    parameters = sum(p.numel() for p in decoder.parameters())

    # Mandatory pre-training geometry gate (redundant with verify_*.py but
    # kept inline so `main()` alone is still safe to call directly).
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

    # Same CAMRI safety-rule semantics as every prior run in this family:
    # per-epoch selection eligibility filter, not a training-abort trigger.
    for epoch in range(1, config["max_epochs"] + 1):
        decoder.train(); losses = []; epoch_start = time.time()
        for j, (domain, r) in enumerate(balanced_epoch_order(records["camri"]["train"], records["mouse"]["train"], config["seed"] + epoch)):
            features, target = load_cached(r, device)
            features, target = augment(features, target, random.Random(config["seed"] + epoch * 100 + j), config["augmentation"])
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
    }, out / "checkpoints/best_unetr_style_level0_full_width_decoder.pt")
    with (out / "training/history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0]); w.writeheader(); w.writerows(history)

    summary = {
        "device": str(device), "selected_epoch": best_epoch, "epochs_run": len(history),
        "best_balanced_validation_dice": best, "encoder_frozen": True, "one_query": True,
        "levels": list(LEVELS), "unetr_style_decoder_parameters": parameters,
        "level0_width": config["level0_width"], "tile_size": list(tile_size),
        "model_spacing_mm": config["model_spacing_mm"],
        "architecture_change": "decoder depth only -- real UnetrUpBlock skip-connected residual upsampling chain level4->3->2->1->0 (level0_width=8 for the largest-grid stage; see docstring for the on-machine timing measurements that set this width), replacing the single depthwise+pointwise level0 fusion pass",
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
