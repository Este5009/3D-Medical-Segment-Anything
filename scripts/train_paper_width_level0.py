#!/usr/bin/env python3
"""PaperWidthLevel0OneQueryMaskDecoder training -- the literature-grounded
successor to the (partially invalidated) embedding_dim/level0_width sweep.

Two changes vs. current best (outputs/higher_resolution_true_level0),
both directly grounded in the paper's own code (RS2/network/RSSNet.py):

1. Upsampling mechanism: PaperUnetrUpBlock (trilinear interpolation + 1x1x1
   conv, matching the paper's own real UnetrUpBlock in
   RS2/network/up_block_unpooling.py) instead of a learned transposed
   convolution. Already validated in isolation: outputs/paper_style_level0_decoder/
   (verdict A).

2. Per-level channel widths match the paper's own real encoder/decoder
   channel progression exactly (level0=level1=48, level2=96, level3=192,
   level4=384 -- Figure 2 / RSSNet.py's own feature_size values), with NO
   projection-based compression anywhere in the canvas the mask is drawn
   from. This directly answers the user's own correction mid-session:
   level0/level1 must not exceed the paper's real 48 -- unlike the earlier,
   partially-invalidated sweep, which incorrectly let level0/level1 scale
   past 48 to 64/96 at some sweep points. Here, level0/level1 are pinned at
   48 by construction (PaperWidthLevel0OneQueryMaskDecoder.CHANNELS is not
   a free parameter), and level2/3/4 sit at their own paper-matched values.
   embedding_dim=32 is retained ONLY for the query vector and its per-level
   cross-attention bridges -- never for the canvas itself.

No checkpoint transfer: every canvas-pathway tensor has a materially
different shape from any prior checkpoint in this project (the whole point
of this experiment is removing the embedding_dim=32 canvas bottleneck), so
this trains from the class's own standard initialization, exactly like the
width sweep before it.

Resilience: this run follows a pod reset that silently destroyed the
in-progress width sweep (nothing had been synced back to the local machine).
To make sure that never costs real GPU time again, this script now writes
its checkpoint and full history to disk after EVERY epoch (not only at
early-stop/completion), so an external sync loop pulling the output
directory back to the local Mac never has more than one epoch's work to
lose, no matter when the pod dies.
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
from train_generalization_pilot import load_cached
from train_mixed_domain_decoder import balanced_epoch_order, cached_records, training_loss
from train_mouse_boundary_adaptation import augment, metric, aggregate
from train_query_decoder_overfit import load_json
from train_corrected_label_retraining import verification_gate

LEVELS = ("level0", "level1", "level2", "level3", "level4")
HISTORY_FIELDS = ("epoch", "loss", "epoch_seconds", "balanced_validation_dice", "camri_safety_eligible",
                   "camri_validation_dice", "camri_validation_iou", "camri_validation_precision", "camri_validation_recall",
                   "mouse_validation_dice", "mouse_validation_iou", "mouse_validation_precision", "mouse_validation_recall")


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


def save_checkpoint(path, state, epoch, best, config, parameters):
    torch.save({
        "decoder_state_dict": state, "epoch": epoch, "balanced_validation_dice": best,
        "embedding_dim": config["embedding_dim"], "decoder_parameters": parameters, "config": config,
    }, path)


def write_history(path, history):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS); w.writeheader(); w.writerows(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper_width_level0.yaml")
    args = ap.parse_args()
    config = load_json(ROOT / args.config)
    tile_size = tuple(config["tile_size"])
    out = ROOT / config["output_directory"]
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "training").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    verification_gate(out)

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    records = cached_records(config)
    prepare_cache(records, config, paths, device)

    decoder = PaperWidthLevel0OneQueryMaskDecoder(config["embedding_dim"], 4).to(device)
    parameters = sum(p.numel() for p in decoder.parameters())
    print(f"PaperWidthLevel0OneQueryMaskDecoder parameters: {parameters}", flush=True)

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
    checkpoint_path = out / "checkpoints/best_paper_width_level0_decoder.pt"
    latest_path = out / "checkpoints/latest_paper_width_level0_decoder.pt"
    history_path = out / "training/history.csv"

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

        # Persist every epoch regardless of improvement, so a pod interruption
        # never loses more than the epoch currently in flight.
        write_history(history_path, history)
        save_checkpoint(latest_path, decoder.state_dict(), epoch, score, config, parameters)

        if not camri_safety_eligible:
            safety_ineligible_epochs.append(epoch)
            continue
        if score > best + config["minimum_validation_improvement"]:
            best = score; state = copy.deepcopy(decoder.state_dict()); best_epoch = epoch; stale = 0
            save_checkpoint(checkpoint_path, state, best_epoch, best, config, parameters)
        else:
            stale += 1
        if stale >= config["early_stop_patience"]:
            break
    safety = best == -1

    if best == -1:
        # No epoch ever cleared the CAMRI safety floor; fall back to saving the
        # last epoch's weights under the "best" name so downstream evaluation
        # scripts still have something to load, clearly flagged as unsafe.
        save_checkpoint(checkpoint_path, decoder.state_dict(), config["max_epochs"], score, config, parameters)

    summary = {
        "device": str(device), "selected_epoch": best_epoch, "epochs_run": len(history),
        "best_balanced_validation_dice": best, "encoder_frozen": True, "one_query": True,
        "levels": list(LEVELS), "decoder_parameters": parameters,
        "embedding_dim": config["embedding_dim"], "canvas_widths": PaperWidthLevel0OneQueryMaskDecoder.CHANNELS,
        "tile_size": list(tile_size), "model_spacing_mm": config["model_spacing_mm"],
        "architecture_change": config["architecture_change"],
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
