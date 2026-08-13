#!/usr/bin/env python3
"""Genuine full-resolution level0 decoder ablation, initialized from the
converged corrected-label baseline (not the pre-correction generalization
pilot checkpoint the earlier, audited level0 experiment mistakenly used).

Controlled intervention: the *only* architectural change relative to
outputs/corrected_label_retraining is swapping MultiScaleOneQueryMaskDecoder
for TrueFullResolutionLevel0OneQueryMaskDecoder (models/query_mask_decoder.py).
That new decoder forms its mask logits once, directly on a fused full
model-grid (level0) feature field -- it contains no half-resolution mask
logits and no logit-space interpolation or residual skip (see the audit at
outputs/full_resolution_level0_decoder/implementation_audit.md for why that
distinction matters).

Every other variable is held fixed relative to
scripts/train_corrected_label_retraining.py: same split files, same
corrected nearest-neighbor label preprocessing, same augmentation, same loss,
same optimizer/lr/schedule, same seed, same checkpoint-selection rule, same
mask-preprocessing verification gate. The sole deliberate difference besides
the decoder class is the initialization checkpoint, which this experiment
requires to be the *converged* corrected-label decoder.
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

from corrected_label_preprocessing import preprocess_image_and_corrected_target
from models.query_mask_decoder import TrueFullResolutionLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_generalization_pilot import load_cached
from train_mixed_domain_decoder import balanced_epoch_order, cached_records, training_loss
from train_mouse_boundary_adaptation import augment, metric, aggregate
from train_query_decoder_overfit import choose_device, load_json
from train_corrected_label_retraining import verification_gate
from verify_true_full_resolution_level0_decoder import verify as run_pretraining_gates

LEVELS = ("level0", "level1", "level2", "level3", "level4")
OLD_DECODER_PARAMETERS = 170401  # MultiScaleOneQueryMaskDecoder, unchanged reference point.


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def prepare_cache(records, config, paths, device):
    cache = Path(config["corrected_cache"]); cache.mkdir(parents=True, exist_ok=True)
    encoder = RS2NetEncoderAdapter(paths, image_size=tuple(config["tile_size"]), in_channels=1, out_channels=1, feature_size=48).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for domain in ("camri", "mouse"):
        for split in ("train", "validation"):
            for i, r in enumerate(records[domain][split], 1):
                dest = cache / f"{domain}_{split}_{r['subject']}.pt"
                r["cache_path"] = str(dest)
                if dest.exists():
                    continue
                image, target, shape, _ = preprocess_image_and_corrected_target(Path(r["image_path"]), Path(r["mask_path"]), paths, tuple(config["tile_size"]))
                with torch.inference_mode():
                    features = encoder(image.to(device))
                torch.save({
                    "features": {k: features[k].cpu().half() for k in LEVELS},
                    "target": target.byte(),
                    "preprocessed_shape": shape,
                    "label_interpolation": "nearest/order0/is_seg=True",
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
    # Every corrected-label baseline tensor must transfer; only new level0
    # tensors may be missing, and nothing from the checkpoint may be dropped.
    if set(shared) != set(old):
        raise RuntimeError(f"Not all baseline decoder tensors transferred: {set(old) - set(shared)}")
    return payload, missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/true_full_resolution_level0_decoder.yaml")
    ap.add_argument("--skip-pretraining-gates", action="store_true", help="For resuming after gates already ran once; not used in the reported run.")
    args = ap.parse_args()
    config = load_json(ROOT / args.config)
    out = ROOT / config["output_directory"]
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "training").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    verification_gate(out)  # Same corrected-label-quality gate the baseline run uses; raises if labels regressed.

    if not args.skip_pretraining_gates:
        gate_report = run_pretraining_gates(args.config)
        if not gate_report.get("all_gates_passed"):
            raise RuntimeError("Pre-training verification gates failed; training aborted.")
        print("Pre-training verification gates passed; proceeding to training.\n", flush=True)

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    device = torch.device("cpu")  # MPS lacks Conv3d support in the installed torch 2.0.0; forced CPU, matching the corrected-label baseline's recorded execution device.
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    records = cached_records(config)
    prepare_cache(records, config, paths, device)

    decoder = TrueFullResolutionLevel0OneQueryMaskDecoder(32, 4, level0_width=config["level0_width"])
    initial, missing, unexpected = initialize(decoder, ROOT / config["initial_checkpoint"])
    decoder.to(device)
    parameters = sum(p.numel() for p in decoder.parameters())
    transferred_baseline_tensors = len(initial["decoder_state_dict"])
    if transferred_baseline_tensors != 76:
        # Sanity: the corrected-label checkpoint's decoder_state_dict must have
        # exactly the 76 entries MultiScaleOneQueryMaskDecoder produces.
        raise RuntimeError(f"Unexpected baseline tensor count: {transferred_baseline_tensors} (expected 76)")

    # Mandatory pre-training geometry gate (redundant with verify_*.py but
    # kept inline so `main()` alone is still safe to call directly).
    f, t = load_cached(records["camri"]["validation"][0], device)
    with_shape = decoder(f, output_size=t.shape[-3:])
    native = decoder(f)
    if native.shape[-3:] != (128, 128, 160) or with_shape.shape != t.shape or not torch.equal(with_shape, native):
        raise RuntimeError("Level0 logits are not full-grid, or an interpolation branch fired unexpectedly")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    camri_ref = np.mean([float(r["dice"]) for r in csv.DictReader(open(ROOT / config["camri_metrics"])) if r["split"] == "validation"])
    camri_floor = camri_ref - config["camri_validation_max_drop"]
    best = -1; state = copy.deepcopy(decoder.state_dict()); best_epoch = 0; stale = 0; history = []
    safety_ineligible_epochs = []
    start = time.time(); peak = 0.

    # NOTE on the CAMRI safety rule (disclosed deviation from
    # corrected_label_retraining's literal control flow; see
    # implementation_audit.md-style reporting in experiment_report.md):
    # in every prior experiment (label-correction only, and the earlier
    # flawed zero-init-residual level0 design) the shared decoder tensors are
    # either fully strict-loaded or the new branch is scale-gated to exactly
    # reproduce the baseline at epoch 0, so epoch 1 already starts near
    # camri_ref and the safety rule only ever fires on genuine mid-training
    # drift. This experiment's mask head is a *new*, freshly-initialized
    # full-resolution projection with no such guarantee (that guarantee is
    # precisely the residual-correction pattern this experiment was asked to
    # remove), so CAMRI performance necessarily collapses for the first few
    # epochs while the new head learns to use the transferred features. Using
    # the safety check as an immediate `break` (as literally copied from
    # train_corrected_label_retraining.py) aborted training after epoch 1
    # with a random, untrained decoder saved as "best" -- see
    # training/attempt1_camri_safety_aborted_epoch1/ for that run's artifacts.
    # The fix keeps the identical numeric criterion (camri_ref - 0.01) and
    # the identical "best balanced validation dice" selection rule, but
    # applies it as a per-epoch *eligibility filter* on which checkpoints may
    # be selected, rather than as a training-abort trigger. Ineligible epochs
    # are still logged, still count toward the max_epochs budget, but neither
    # update `best`/`state` nor consume early-stop patience (a warmup dip is
    # not "no improvement", it is "not yet evaluable").
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
            continue  # Not evaluable for selection or patience; keep training.
        if score > best + config["minimum_validation_improvement"]:
            best = score; state = copy.deepcopy(decoder.state_dict()); best_epoch = epoch; stale = 0
        else:
            stale += 1
        if stale >= config["early_stop_patience"]:
            break
    safety = best == -1  # True only if NO epoch within the full budget ever met the CAMRI safety floor.

    torch.save({
        "decoder_state_dict": state, "epoch": best_epoch, "balanced_validation_dice": best,
        "initial_checkpoint": config["initial_checkpoint"], "config": config,
    }, out / "checkpoints/best_true_level0_decoder.pt")
    with (out / "training/history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0]); w.writeheader(); w.writerows(history)

    summary = {
        "device": str(device), "selected_epoch": best_epoch, "epochs_run": len(history),
        "best_balanced_validation_dice": best, "encoder_frozen": True, "one_query": True,
        "levels": list(LEVELS), "old_decoder_parameters": OLD_DECODER_PARAMETERS,
        "true_level0_decoder_parameters": parameters, "level0_width": config["level0_width"],
        "architecture_change": "genuine full-grid level0 mask head (no half-resolution logit branch, no residual skip)",
        "native_logits_shape": [1, 1, 128, 128, 160], "final_logit_interpolation": False,
        "query_conditioning_at_level0": True, "level0_query_tokens_pooled_2x": True,
        "camri_safety_stop": safety, "camri_floor": camri_floor, "camri_reference": camri_ref,
        "safety_ineligible_epochs": safety_ineligible_epochs,
        "camri_safety_rule_semantics": "per-epoch selection eligibility filter, not a training-abort trigger (see inline comment in this script for why)",
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
