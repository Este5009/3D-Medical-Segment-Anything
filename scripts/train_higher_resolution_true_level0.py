#!/usr/bin/env python3
"""Higher-resolution (192x128x160) TRUE level0 decoder controlled experiment.

Controlled intervention: the *only* change relative to
outputs/true_full_resolution_level0_decoder is the model-space input/target
grid, refined from 128x128x160 to 192x128x160 on model axis0 (-> native Y),
the single axis outputs/higher_resolution_encoder_diagnostic/ identified as
the dominant native-boundary-detail-loss axis. Model axis1 (-> native Z,
already finer than native) and axis2 (-> native X) keep the valid baseline's
spacing. The decoder class, architecture, and all hyperparameters are
otherwise identical to scripts/train_true_full_resolution_level0_decoder.py.

Because TrueFullResolutionLevel0OneQueryMaskDecoder's parameters (Conv3d,
Linear, MultiheadAttention, InstanceNorm3d) are all channel-dimension-only --
none depend on spatial D/H/W -- the entire 182,081-parameter checkpoint from
outputs/true_full_resolution_level0_decoder/checkpoints/best_true_level0_decoder.pt
transfers exactly (verified: 102/102 tensors, 0 new parameters; see
verify_higher_resolution_true_level0.py). This is a materially different,
cleaner initialization than the original true-level0 experiment (which had
26 freshly-initialized parameters and a large epoch-1 CAMRI collapse); this
run may not need the CAMRI safety-eligibility-filter machinery at all, but it
is kept for consistency with the established checkpoint-selection logic.
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
from models.query_mask_decoder import TrueFullResolutionLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_generalization_pilot import load_cached
from train_mixed_domain_decoder import balanced_epoch_order, cached_records, training_loss
from train_mouse_boundary_adaptation import augment, metric, aggregate
from train_query_decoder_overfit import load_json
from train_corrected_label_retraining import verification_gate
from verify_higher_resolution_true_level0 import verify as run_pretraining_gates

LEVELS = ("level0", "level1", "level2", "level3", "level4")
BASELINE_DECODER_PARAMETERS = 182081  # outputs/true_full_resolution_level0_decoder, this experiment's comparator.
EXPECTED_TRANSFERRED_TENSORS = 102  # ALL of TrueFullResolutionLevel0OneQueryMaskDecoder's tensors (spatially shape-agnostic).


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
    if set(shared) != set(old):
        raise RuntimeError(f"Not all baseline decoder tensors transferred: {set(old) - set(shared)}")
    return payload, missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/higher_resolution_true_level0.yaml")
    ap.add_argument("--skip-pretraining-gates", action="store_true")
    args = ap.parse_args()
    config = load_json(ROOT / args.config)
    tile_size = tuple(config["tile_size"])
    out = ROOT / config["output_directory"]
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "training").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    verification_gate(out)  # Same corrected-label-quality gate the baseline run uses; resolution-independent.

    if not args.skip_pretraining_gates:
        gate_report = run_pretraining_gates(args.config)
        if not gate_report.get("all_gates_passed"):
            raise RuntimeError("Pre-training verification gates failed; training aborted.")
        print("Pre-training verification gates passed; proceeding to training.\n", flush=True)

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    device = torch.device("cpu")  # MPS lacks Conv3d support in the installed torch 2.0.0.
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    records = cached_records(config)
    prepare_cache(records, config, paths, device)

    decoder = TrueFullResolutionLevel0OneQueryMaskDecoder(32, 4, level0_width=config["level0_width"])
    initial, missing, unexpected = initialize(decoder, ROOT / config["initial_checkpoint"])
    decoder.to(device)
    parameters = sum(p.numel() for p in decoder.parameters())
    transferred_baseline_tensors = len(initial["decoder_state_dict"])
    if transferred_baseline_tensors != EXPECTED_TRANSFERRED_TENSORS:
        raise RuntimeError(f"Unexpected baseline tensor count: {transferred_baseline_tensors} (expected {EXPECTED_TRANSFERRED_TENSORS})")
    if missing:
        raise RuntimeError(f"Expected zero new parameters (decoder is spatially shape-agnostic); got {missing}")

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

    # Same CAMRI safety-rule semantics as train_true_full_resolution_level0_decoder.py
    # (per-epoch selection eligibility filter, not a training-abort trigger).
    # See that script's inline comment for the full rationale; kept here for
    # checkpoint-selection consistency even though this run's 100%-transferred
    # initialization is not expected to need it.
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
    }, out / "checkpoints/best_higher_resolution_true_level0.pt")
    with (out / "training/history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0]); w.writeheader(); w.writerows(history)

    summary = {
        "device": str(device), "selected_epoch": best_epoch, "epochs_run": len(history),
        "best_balanced_validation_dice": best, "encoder_frozen": True, "one_query": True,
        "levels": list(LEVELS), "baseline_decoder_parameters": BASELINE_DECODER_PARAMETERS,
        "higher_resolution_decoder_parameters": parameters, "level0_width": config["level0_width"],
        "tile_size": list(tile_size), "model_spacing_mm": config["model_spacing_mm"],
        "architecture_change": "none -- input/target resolution only (128x128x160 -> 192x128x160, axis0/native-Y refined)",
        "native_logits_shape": [1, 1, *tile_size], "final_logit_interpolation": False,
        "query_conditioning_at_level0": True, "level0_query_tokens_pooled_2x": True,
        "camri_safety_stop": safety, "camri_floor": camri_floor, "camri_reference": camri_ref,
        "safety_ineligible_epochs": safety_ineligible_epochs,
        "camri_safety_rule_semantics": "per-epoch selection eligibility filter, not a training-abort trigger",
        "mean_epoch_seconds": float(np.mean([r["epoch_seconds"] for r in history])),
        "peak_process_memory_mib": peak, "elapsed_seconds": time.time() - start,
        "transferred_baseline_tensors": transferred_baseline_tensors,
        "new_parameter_keys": list(missing), "unexpected_keys": list(unexpected),
        "initial_checkpoint": config["initial_checkpoint"],
    }
    (out / "training/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
