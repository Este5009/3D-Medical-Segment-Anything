#!/usr/bin/env python3
"""TwoTaskLevel0OneQueryMaskDecoder joint training -- brain (CAMRI+Mouse)
and lesion (stroke) share every weight except their own starting query
vector. See configs/two_task_joint.yaml docstring for the full rationale
and what this is compared against.

Per-epoch balance: the brain training pool (CAMRI+Mouse combined, 39
subjects) is much smaller than the lesion training pool (223 subjects), so
each epoch samples a lesion subset the same size as the brain pool (a
different random subset each epoch, cycling through the full 223 over many
epochs) rather than letting lesion examples dominate every gradient step
50-to-1. Both tasks contribute equally to every epoch's gradient updates.

Caching: only the PREPROCESSED IMAGE + TARGET are cached to disk for both
domains (~11MB/subject), not the frozen encoder's own output features
(~420MB/subject at this resolution). Caching features for both domains
combined would need well over 100GB, which does not fit the pod's real
50GB network-volume quota (discovered the hard way mid-session tonight --
/workspace is NOT the effectively-unlimited pool `df -h` makes it look
like). The frozen encoder runs fresh on every training/eval step instead.

Resilience: writes checkpoint + history to disk after every epoch, same
practice as every other script tonight.
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
from models.query_mask_decoder import TwoTaskLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_mixed_domain_decoder import balanced_epoch_order, cached_records, training_loss
from train_mouse_boundary_adaptation import metric, aggregate
from train_query_decoder_overfit import load_json
from train_corrected_label_retraining import verification_gate
from train_stroke_lesion_only import stroke_records, robust_save, augment_lesion_image

LEVELS = ("level0", "level1", "level2", "level3", "level4")


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def prepare_brain_cache(records, config, paths):
    """Preprocessed image+target only -- see module docstring."""
    cache = Path(config["corrected_cache"]); cache.mkdir(parents=True, exist_ok=True)
    tile_size = tuple(config["tile_size"])
    spacing = tuple(config["model_spacing_mm"])
    for domain in ("camri", "mouse"):
        for split in ("train", "validation"):
            for i, r in enumerate(records[domain][split], 1):
                dest = cache / f"{domain}_{split}_{r['subject']}.pt"
                r["cache_path"] = str(dest)
                if dest.exists():
                    continue
                image, target, shape, _ = preprocess_image_and_corrected_target_at_spacing(
                    Path(r["image_path"]), Path(r["mask_path"]), paths, spacing, tile_size)
                robust_save({"image": image.half(), "target": target.byte()}, dest)
                print(f"cache {domain} {split} {i}/{len(records[domain][split])}", flush=True)


def load_cached_image(record, device):
    payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
    image = payload["image"].to(device).float()
    target = payload["target"].to(device).float()
    return image, target


def augment_brain_image(image, target, rng, config):
    """Same recipe as augment_lesion_image -- flip + light intensity
    jitter, now applied pre-encoder for both tasks identically."""
    return augment_lesion_image(image, target, rng, config)


@torch.inference_mode()
def evaluate_brain(encoder, decoder, records, device):
    decoder.eval(); out = []
    for r in records:
        image, t = load_cached_image(r, device)
        f = encoder(image)
        logits = decoder(f, task="brain", output_size=t.shape[-3:])
        out.append({"subject": r["subject"], **metric(logits, t)})
    return out


@torch.inference_mode()
def evaluate_lesion(encoder, decoder, records, device):
    decoder.eval(); out = []
    for r in records:
        image, t = load_cached_image(r, device)
        f = encoder(image)
        logits = decoder(f, task="lesion", output_size=t.shape[-3:])
        out.append({"subject": r["subject"], **metric(logits, t)})
    return out


def save_checkpoint(path, state, epoch, best, config, parameters):
    torch.save({
        "decoder_state_dict": state, "epoch": epoch, "balanced_validation_dice": best,
        "embedding_dim": config["embedding_dim"], "decoder_parameters": parameters, "config": config,
    }, path)


def write_history(path, history):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_task_joint.yaml")
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
    # Instantiating the encoder first registers RS2's own package onto
    # sys.path, which the preprocessing pipeline's imports depend on --
    # must happen before either prepare_*_cache call, not after.
    encoder = RS2NetEncoderAdapter(paths, image_size=tile_size, in_channels=1, out_channels=1, feature_size=48).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    brain_records = cached_records(config)
    prepare_brain_cache(brain_records, config, paths)
    lesion_records = stroke_records(config)
    # stroke_records/prepare_cache from train_stroke_lesion_only already
    # cache image+target only, matching this script's own brain caching.
    from train_stroke_lesion_only import prepare_cache as prepare_stroke_cache
    prepare_stroke_cache(lesion_records, config, paths, device)
    brain_train_n = len(brain_records["camri"]["train"]) + len(brain_records["mouse"]["train"])
    print(f"brain train pool={brain_train_n} lesion train pool={len(lesion_records['train'])} "
          f"(lesion subsampled to {brain_train_n}/epoch for balance)", flush=True)

    decoder = TwoTaskLevel0OneQueryMaskDecoder(config["embedding_dim"], 4).to(device)
    parameters = sum(p.numel() for p in decoder.parameters())
    print(f"TwoTaskLevel0OneQueryMaskDecoder parameters: {parameters}", flush=True)

    ib, tb = load_cached_image(brain_records["camri"]["validation"][0], device)
    il, tl = load_cached_image(lesion_records["validation"][0], device)
    with torch.no_grad():
        fb = encoder(ib); fl = encoder(il)
        ob = decoder(fb, task="brain", output_size=tb.shape[-3:])
        ol = decoder(fl, task="lesion", output_size=tl.shape[-3:])
    if ob.shape != tb.shape or ol.shape != tl.shape or torch.equal(ob, ol):
        raise RuntimeError("Sanity gate failed: task outputs wrong shape or identical across tasks")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    camri_ref = np.mean([float(r["dice"]) for r in csv.DictReader(open(ROOT / config["camri_metrics"])) if r["split"] == "validation"])
    camri_floor = camri_ref - config["camri_validation_max_drop"]
    best = -1; state = copy.deepcopy(decoder.state_dict()); best_epoch = 0; stale = 0; history = []
    safety_ineligible_epochs = []
    start = time.time(); peak = 0.
    checkpoint_path = out / "checkpoints/best_two_task_joint_decoder.pt"
    latest_path = out / "checkpoints/latest_two_task_joint_decoder.pt"
    history_path = out / "training/history.csv"

    for epoch in range(1, config["max_epochs"] + 1):
        decoder.train(); losses = []; epoch_start = time.time()
        rng_epoch = random.Random(config["seed"] + epoch)
        brain_order = [("brain", r) for _, r in balanced_epoch_order(brain_records["camri"]["train"], brain_records["mouse"]["train"], config["seed"] + epoch)]
        lesion_pool = lesion_records["train"][:]
        rng_epoch.shuffle(lesion_pool)
        lesion_order = [("lesion", r) for r in lesion_pool[:brain_train_n]]
        combined = brain_order + lesion_order
        rng_epoch.shuffle(combined)

        for j, (task, r) in enumerate(combined):
            step_rng = random.Random(config["seed"] + epoch * 1000 + j)
            image, target = load_cached_image(r, device)
            image, target = augment_lesion_image(image, target, step_rng, config)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                features = encoder(image)
                logits = decoder(features, task=task, output_size=target.shape[-3:])
            loss, parts = training_loss(logits.float(), target, config)
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))

        val_camri = aggregate(evaluate_brain(encoder, decoder, brain_records["camri"]["validation"], device))
        val_mouse = aggregate(evaluate_brain(encoder, decoder, brain_records["mouse"]["validation"], device))
        val_lesion = aggregate(evaluate_lesion(encoder, decoder, lesion_records["validation"], device))
        score = (val_camri["dice"] + val_mouse["dice"] + val_lesion["dice"]) / 3
        elapsed = time.time() - epoch_start; peak = max(peak, peak_mib())
        camri_safety_eligible = val_camri["dice"] >= camri_floor
        row = {"epoch": epoch, "loss": np.mean(losses), "epoch_seconds": elapsed, "balanced_validation_dice": score,
               "camri_safety_eligible": camri_safety_eligible,
               **{f"camri_validation_{k}": v for k, v in val_camri.items()},
               **{f"mouse_validation_{k}": v for k, v in val_mouse.items()},
               **{f"lesion_validation_{k}": v for k, v in val_lesion.items()}}
        history.append(row)
        print(f"epoch {epoch} CAMRI={val_camri['dice']:.6f} Mouse={val_mouse['dice']:.6f} Lesion={val_lesion['dice']:.6f} "
              f"balanced={score:.6f} seconds={elapsed:.1f} camri_safety_eligible={camri_safety_eligible}", flush=True)

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
    if safety:
        save_checkpoint(checkpoint_path, decoder.state_dict(), config["max_epochs"], score, config, parameters)

    summary = {
        "device": str(device), "selected_epoch": best_epoch, "epochs_run": len(history),
        "best_balanced_validation_dice": best, "encoder_frozen": True, "two_query": True,
        "levels": list(LEVELS), "decoder_parameters": parameters,
        "embedding_dim": config["embedding_dim"], "tile_size": list(tile_size),
        "model_spacing_mm": config["model_spacing_mm"], "architecture_change": config["architecture_change"],
        "brain_train_pool": brain_train_n, "lesion_train_pool": len(lesion_records["train"]),
        "camri_safety_stop": safety, "camri_floor": camri_floor, "camri_reference": camri_ref,
        "safety_ineligible_epochs": safety_ineligible_epochs,
        "mean_epoch_seconds": float(np.mean([r["epoch_seconds"] for r in history])),
        "peak_process_memory_mib": peak, "elapsed_seconds": time.time() - start,
    }
    (out / "training/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
