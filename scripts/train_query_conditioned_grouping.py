#!/usr/bin/env python3
"""Validation-only selection of a compact query-conditioned grouping residual."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.query_conditioned_grouping import QueryConditionedGroupingDecoder
from models.query_mask_decoder import MultiScaleOneQueryMaskDecoder, dice_bce_boundary_loss
from train_generalization_pilot import load_cached
from train_mixed_domain_decoder import balanced_epoch_order, cached_records, ensure_feature_cache
from train_mouse_boundary_adaptation import aggregate, augment, metric
from train_query_decoder_overfit import choose_device, load_json


def load_frozen_decoder(config: dict, device: torch.device) -> MultiScaleOneQueryMaskDecoder:
    checkpoint = torch.load(ROOT / config["baseline_checkpoint"], map_location="cpu", weights_only=False)
    decoder = MultiScaleOneQueryMaskDecoder(32, 4)
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    decoder.to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    assert tuple(decoder.query.shape) == (1, 1, 32)
    assert not any(parameter.requires_grad for parameter in decoder.parameters())
    return decoder


def forward_cached(grouping, decoder, features, target, use_query):
    """Run frozen baseline and trainable residual without constructing baseline gradients."""
    with torch.no_grad():
        initial = decoder(features, output_size=None)
    final, correction = grouping(
        features["level2"], initial, decoder.query.detach(),
        use_query=use_query, output_size=target.shape[-3:]
    )
    return final, correction


@torch.inference_mode()
def evaluate(grouping, decoder, records, device, use_query):
    grouping.eval()
    result = []
    for record in records:
        features, target = load_cached(record, device)
        logits, correction = forward_cached(grouping, decoder, features, target, use_query)
        result.append({
            "subject": record["subject"],
            **metric(logits, target),
            "mean_absolute_correction": float(correction.abs().mean()),
        })
    return result


def baseline_validation(decoder, records, device):
    class Identity(torch.nn.Module):
        @torch.inference_mode()
        def __call__(self, feature_map, target):
            return decoder(feature_map, output_size=target.shape[-3:])
    result = {}
    for domain in ("camri", "mouse"):
        rows = []
        for record in records[domain]["validation"]:
            features, target = load_cached(record, device)
            rows.append({"subject": record["subject"], **metric(Identity()(features, target), target)})
        result[domain] = aggregate(rows)
    return result


def train_ablation(name, options, config, records, decoder, device, output):
    torch.manual_seed(config["seed"])
    model = QueryConditionedGroupingDecoder(
        feature_channels=config["feature_channels"],
        hidden_channels=config["hidden_channels"],
        use_coordinates=options["use_coordinates"],
        max_correction=config["max_correction"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history = []
    baseline = baseline_validation(decoder, records, device)
    baseline_score = np.mean([baseline[d]["dice"] for d in ("camri", "mouse")])

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        loss_rows = []
        order = balanced_epoch_order(
            records["camri"]["train"], records["mouse"]["train"], config["seed"] + epoch
        )
        for index, (_, record) in enumerate(order):
            features, target = load_cached(record, device)
            features, target = augment(
                features, target, random.Random(config["seed"] + epoch * 100 + index),
                {"flip_probability": 0.5, "feature_scale": 0.03, "feature_noise_std": 0.005},
            )
            optimizer.zero_grad(set_to_none=True)
            logits, correction = forward_cached(model, decoder, features, target, options["use_query"])
            segmentation_loss, parts = dice_bce_boundary_loss(
                logits, target,
                boundary_weight=config["boundary_weight"], boundary_width=1,
            )
            residual_loss = correction.abs().mean()
            loss = segmentation_loss + config["residual_weight"] * residual_loss
            loss.backward()
            optimizer.step()
            loss_rows.append({
                "loss": float(loss.detach()),
                **{key: float(value) for key, value in parts.items()},
                "residual_loss": float(residual_loss.detach()),
            })

        validation = {
            domain: aggregate(evaluate(
                model, decoder, records[domain]["validation"], device, options["use_query"]
            ))
            for domain in ("camri", "mouse")
        }
        score = np.mean([validation[d]["dice"] for d in ("camri", "mouse")])
        row = {
            "ablation": name, "epoch": epoch,
            **{key: float(np.mean([item[key] for item in loss_rows])) for key in loss_rows[0]},
            "balanced_validation_dice": float(score),
            **{f"{domain}_validation_{key}": value
               for domain in validation for key, value in validation[domain].items()},
        }
        history.append(row)
        print(
            f"{name} epoch {epoch}: CAMRI={validation['camri']['dice']:.5f} "
            f"Mouse={validation['mouse']['dice']:.5f} balanced={score:.5f}", flush=True
        )
        # Safety stop is relative to the unchanged model on the same validation records.
        destabilized = any(
            validation[d]["dice"] < baseline[d]["dice"] - config["validation_safety_drop"]
            for d in ("camri", "mouse")
        )
        if destabilized:
            print(f"{name}: validation safety stop", flush=True)
            break
        if score > best_score + config["minimum_validation_improvement"]:
            best_score, best_epoch, stale = float(score), epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config["early_stop_patience"]:
            break

    model.load_state_dict(best_state)
    final = {
        domain: aggregate(evaluate(
            model, decoder, records[domain]["validation"], device, options["use_query"]
        ))
        for domain in ("camri", "mouse")
    }
    history_path = output / "ablations" / f"{name}_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    torch.save({
        "grouping_state_dict": best_state,
        "ablation": name,
        "options": options,
        "best_epoch": best_epoch,
        "validation": final,
        "baseline_checkpoint": config["baseline_checkpoint"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }, output / "ablations" / f"{name}.pt")
    return {
        "name": name, "options": options, "best_epoch": best_epoch,
        "epochs_run": len(history), "balanced_validation_dice": float(np.mean(
            [final[d]["dice"] for d in ("camri", "mouse")]
        )),
        "validation": final, "parameter_count": sum(p.numel() for p in model.parameters()),
        "baseline_balanced_validation_dice": float(baseline_score),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/query_conditioned_3d_grouping.yaml")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    output = ROOT / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "configuration.json").write_text(json.dumps(config, indent=2))
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    device = choose_device()
    records = cached_records(config)
    ensure_feature_cache(records, config, device)
    decoder = load_frozen_decoder(config, device)
    started = time.time()
    results = [
        train_ablation(name, options, config, records, decoder, device, output)
        for name, options in config["ablations"].items()
    ]
    # Only validation results select the learned candidate.
    selected = max(results, key=lambda item: item["balanced_validation_dice"])
    source = torch.load(output / "ablations" / f"{selected['name']}.pt", map_location="cpu", weights_only=False)
    torch.save(source, output / "best_grouping_checkpoint.pt")
    summary = {
        "device": str(device), "encoder_frozen": True, "baseline_decoder_frozen": True,
        "exactly_one_existing_query": True, "selection_data": "validation only",
        "affinity_objective": "not used in this first controlled residual experiment",
        "ablations": results, "selected_ablation": selected["name"],
        "elapsed_seconds": time.time() - started,
    }
    (output / "validation_ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
