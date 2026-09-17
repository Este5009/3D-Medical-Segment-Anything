#!/usr/bin/env python3
"""Synthetic noise/blur/intensity robustness test for the ORIGINAL RS2-Net
model (their own encoder AND their own decoder, no query mechanism) -- the
noise-robustness counterpart to test_rs2net_baseline_invariance.py. Same
corruption model and severities as this project's own test_noise_robustness.py,
run on CAMRI + mouse brain data (RS2-Net has no lesion task) at RS2-Net's own
native resolution (128x128x160, 0.25x0.200x0.160mm), so the result is directly
comparable to outputs/noise_robustness/brain/summary_by_corruption.csv.

Answers directly: does RS2-Net's own decoder -- trained on their own,
larger, multi-center dataset -- hold up better or worse than this project's
decoder under the exact same synthetic corruption?
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from corrected_label_preprocessing import preprocess_image_and_corrected_target
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import load_json
from test_noise_robustness import apply_corruption, NOISE_STDS, BLUR_SIGMAS, INTENSITY_SEVERITIES, TRAINING_ENVELOPE
import space_invariance_common as sic

NATIVE_TILE = (128, 128, 160)


def subject_records(domain, limit):
    if domain == "camri":
        split = load_json(ROOT / "outputs/generalization_pilot/subject_split.json")
        src = {r["subject"]: r for r in csv.DictReader(open(ROOT / "outputs/generalization_pilot/metrics.csv")) if r["split"] == "test"}
        ids = [s for s in split.get("test", []) if s in src]
        records = [{"sid": s, "image": Path(src[s]["image_path"]), "mask": Path(src[s]["mask_path"])} for s in ids]
    else:
        split = load_json(ROOT / "outputs/mouse_boundary_adaptation/split.json")
        src = {r["scan_id"]: r for r in csv.DictReader(open(ROOT / "outputs/mouse_external_evaluation/subject_metrics.csv"))}
        records = [{"sid": s, "image": Path(src[s]["image_path"]), "mask": Path(src[s]["ground_truth_path"])}
                   for s in split["test"]["scans"] if s in src]
    return records[:limit] if limit else records


def run_subject(network, image_path, mask_path, paths, device, threshold, seed):
    data, target, _, _ = preprocess_image_and_corrected_target(image_path, mask_path, paths, NATIVE_TILE)
    x = data.to(device)
    gt = target[0, 0].numpy() > 0.5
    rng = random.Random(seed)

    with torch.inference_mode():
        clean_logits = network(x)
    clean_mask = (clean_logits.sigmoid()[0, 0].cpu().numpy() > threshold)
    clean_dice = sic.dice(clean_mask, gt)

    rows = []
    for kind, severities in (("noise", NOISE_STDS), ("blur", BLUR_SIGMAS), ("intensity", INTENSITY_SEVERITIES)):
        for sev in severities:
            x_c = apply_corruption(x, kind, sev, rng, device)
            with torch.inference_mode():
                logits_c = network(x_c)
            mask_c = (logits_c.sigmoid()[0, 0].cpu().numpy() > threshold)
            dice_c = sic.dice(mask_c, gt)
            rows.append({"corruption": kind, "severity": sev,
                        "within_training_envelope": sev <= TRAINING_ENVELOPE[kind],
                        "clean_dice": clean_dice, "corrupted_dice": dice_c,
                        "dice_delta": dice_c - clean_dice})
    return rows, clean_dice


def aggregate(rows):
    keys = sorted(set((r["corruption"], r["severity"]) for r in rows))
    out = []
    for kind, sev in keys:
        q = [r for r in rows if r["corruption"] == kind and r["severity"] == sev]
        out.append({"corruption": kind, "severity": sev,
                    "within_training_envelope": q[0]["within_training_envelope"], "n": len(q),
                    "mean_clean_dice": float(np.mean([r["clean_dice"] for r in q])),
                    "mean_corrupted_dice": float(np.mean([r["corrupted_dice"] for r in q])),
                    "min_corrupted_dice": float(np.min([r["corrupted_dice"] for r in q])),
                    "mean_dice_delta": float(np.mean([r["dice_delta"] for r in q])),
                    "worst_dice_delta": float(np.min([r["dice_delta"] for r in q]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("camri", "mouse", "both"), default="both")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--out", default="outputs/rs2net_baseline_noise_robustness")
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_cfg = load_json(ROOT / "configs/rs2net_encoder.yaml")
    paths = RS2NetPaths.from_config(enc_cfg)
    adapter = RS2NetEncoderAdapter(paths, image_size=NATIVE_TILE, in_channels=1, out_channels=1, feature_size=48)
    network = adapter.network.to(device).eval()
    for p in network.parameters():
        p.requires_grad_(False)

    threshold = 0.5
    domains = ("camri", "mouse") if args.domain == "both" else (args.domain,)
    all_summaries = {}
    for domain in domains:
        out = ROOT / args.out / domain
        out.mkdir(parents=True, exist_ok=True)
        records = subject_records(domain, args.limit)
        rows = []
        for i, r in enumerate(records, 1):
            subj_rows, clean_dice = run_subject(network, r["image"], r["mask"], paths, device, threshold, args.seed + i)
            for row in subj_rows:
                row["subject"] = r["sid"]
            rows += subj_rows
            print(f"[rs2net-baseline-noise {domain} {i}/{len(records)}] {r['sid']} clean_dice={clean_dice:.4f} "
                  f"worst_corrupted={min(row['corrupted_dice'] for row in subj_rows):.4f}", flush=True)

        sic.write_csv(out / "per_subject_corruption_metrics.csv", rows)
        summary = aggregate(rows)
        sic.write_csv(out / "summary_by_corruption.csv", summary)
        (out / "summary.json").write_text(json.dumps({"domain": domain, "subjects": len(records), "rows": summary}, indent=2))
        all_summaries[domain] = summary
        print(json.dumps(summary, indent=2))

    (ROOT / args.out / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    write_comparison_report(all_summaries, ROOT / args.out / "report.md")


def write_comparison_report(summaries, path):
    own_path = ROOT / "outputs/noise_robustness/brain/summary_by_corruption.csv"
    own_rows = list(csv.DictReader(open(own_path))) if own_path.exists() else []
    own_by_key = {(r["corruption"], r["severity"]): r for r in own_rows}

    lines = ["# RS2-Net baseline noise/blur/intensity robustness vs. this project's model",
             "",
             "Same corruption model and severities as `test_noise_robustness.py`, run against "
             "the ORIGINAL RS2-Net model (their own encoder AND their own decoder) on CAMRI + "
             "mouse brain data -- RS2-Net has no lesion task. Compared directly against this "
             "project's own brain-query result (`outputs/noise_robustness/brain/`).",
             ""]
    for domain, summary in summaries.items():
        lines += [f"## {domain}", "",
                  "| corruption | severity | RS2-Net baseline mean Dice | this project's model mean Dice | difference |",
                  "|---|---|---|---|---|"]
        for r in summary:
            key = (r["corruption"], str(r["severity"]))
            own = own_by_key.get(key)
            own_dice = float(own["mean_corrupted_dice"]) if own else float("nan")
            diff = r["mean_corrupted_dice"] - own_dice
            lines.append(f"| {r['corruption']} | {r['severity']:g} | {r['mean_corrupted_dice']:.4f} | "
                         f"{own_dice:.4f} | {diff:+.4f} |")
        lines.append("")
    lines.append("_Positive difference = RS2-Net's own decoder scored higher than this project's "
                 "decoder at that corruption/severity, on brain segmentation._")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
