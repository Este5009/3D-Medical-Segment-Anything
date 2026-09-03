#!/usr/bin/env python3
"""embedding_dim / level0_width sweep on top of "current best"'s own exact
architecture (TrueFullResolutionLevel0OneQueryMaskDecoder) and training
recipe -- no decoder-depth change, no upsampling-mechanism change, no
augmentation change. Single variable: decoder width.

Motivation: "current best" (outputs/higher_resolution_true_level0) uses
embedding_dim=32, level0_width=16 -- level0_width deliberately narrower
than embedding_dim, a choice inherited from an earlier CPU-tractability
constraint (established this session: a real cost cliff on this project's
CPU at wider level0 channel counts). Direct inspection of the original
paper's own code (RS2/network/RSSNet.py) found their final, full-resolution
stage (decoder1) does NOT narrow at all -- in_channels=out_channels=
feature_size=48, the same width as every other near-final stage, all the
way through. Our level0_width=16 is 3x narrower than their 48 at the exact
stage that forms the final mask. On a rented GPU, the original CPU
constraint that motivated the narrowing no longer applies (profiled
directly before committing to this run: level0_width up to 96, matching
embedding_dim rather than half of it, costs no more than ~0.7s/training
step -- no cliff).

This sweep tests embedding_dim in {32, 48, 64, 96}, each with
level0_width == embedding_dim (matching the paper's own "don't narrow the
final stage" convention, not this project's historical half-width
convention). embedding_dim=32/level0_width=32 is included as a control
point at "current best"'s own embedding_dim, still wider than current
best's own level0_width=16.

No checkpoint transfer: every tensor in TrueFullResolutionLevel0OneQuery
MaskDecoder scales with embedding_dim or level0_width except mask_bias (a
single scalar), so a meaningful partial transfer isn't possible once width
changes -- each point trains from the class's own standard initialization
(matching how the query itself is already initialized: nn.init.normal_,
std=0.02), not from "current best"'s checkpoint. Same hyperparameters as
outputs/higher_resolution_true_level0's own recipe otherwise (learning
rate, weight decay, loss, feature-space flip+noise augmentation, same 39
training images, same locked split), so the widths are the only isolated
variable across all four points and against current best.
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

LEVELS = ("level0", "level1", "level2", "level3", "level4")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    config = load_json(ROOT / args.config)
    tile_size = tuple(config["tile_size"])
    out = ROOT / config["output_directory"]
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "training").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    verification_gate(out)  # Same corrected-label-quality gate every run in this project uses; width-independent.

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    records = cached_records(config)
    prepare_cache(records, config, paths, device)

    decoder = TrueFullResolutionLevel0OneQueryMaskDecoder(
        config["embedding_dim"], 4, level0_width=config["level0_width"]).to(device)
    parameters = sum(p.numel() for p in decoder.parameters())

    # Sanity gate: full-grid logits, no final-logit upsampling, matching every
    # prior true-level0 experiment's own inline check.
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
        "embedding_dim": config["embedding_dim"], "level0_width": config["level0_width"], "config": config,
    }, out / f"checkpoints/best_level0_width_sweep_emb{config['embedding_dim']}.pt")
    with (out / "training/history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0]); w.writeheader(); w.writerows(history)

    summary = {
        "device": str(device), "selected_epoch": best_epoch, "epochs_run": len(history),
        "best_balanced_validation_dice": best, "encoder_frozen": True, "one_query": True,
        "levels": list(LEVELS), "decoder_parameters": parameters,
        "embedding_dim": config["embedding_dim"], "level0_width": config["level0_width"],
        "tile_size": list(tile_size), "model_spacing_mm": config["model_spacing_mm"],
        "architecture_change": f"width sweep on current best's own architecture -- embedding_dim={config['embedding_dim']}, level0_width={config['level0_width']} (unreduced, matching the paper's own final-stage convention), vs current best's embedding_dim=32/level0_width=16. No decoder-depth or upsampling-mechanism change; trained from this class's own standard initialization, not transferred from any checkpoint (width change makes meaningful transfer impossible).",
        "native_logits_shape": [1, 1, *tile_size], "final_logit_interpolation": False,
        "camri_safety_stop": safety, "camri_floor": camri_floor, "camri_reference": camri_ref,
        "safety_ineligible_epochs": safety_ineligible_epochs,
        "camri_safety_rule_semantics": "per-epoch selection eligibility filter, not a training-abort trigger",
        "mean_epoch_seconds": float(np.mean([r["epoch_seconds"] for r in history])),
        "peak_process_memory_mib": peak, "elapsed_seconds": time.time() - start,
    }
    (out / "training/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
