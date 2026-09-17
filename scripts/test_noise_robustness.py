#!/usr/bin/env python3
"""Synthetic image-quality robustness test (advisor-requested): how much does
the current two-task model's accuracy degrade under Gaussian noise, Gaussian
blur, and random intensity (brightness/contrast) shifts -- separately from
the space-invariance work, which is about POSE, not image QUALITY.

Unlike the rotation/translation tests, corruption here is not invertible, so
there is no "self-consistency" metric -- each corrupted prediction is scored
directly against the expert mask, and reported as a degradation from that
same subject's own clean-image Dice.

Corruption model, applied directly to the preprocessed (already-normalized)
image tensor, before the frozen encoder ever sees it:
  noise      x + N(0, std)                         std in {0.05, 0.1, 0.2, 0.4}
  blur       separable 3D Gaussian smoothing         sigma (voxels) in {0.5, 1, 2, 4}
  intensity  x * (1 + U(-w, w)) + U(-w, w) * 0.5    w (severity) in {0.1, 0.25, 0.5, 0.75}

Severity levels are chosen to straddle the encoder's OWN original training
augmentation (see outputs/summarized_diagnosis/rs2net_original_training_technique.md:
noise p=0.1, blur sigma 0.5-1.0 at p=0.2, brightness x0.75-1.25 at p=0.15) --
so the lower severities sit inside what the encoder was already trained to
tolerate, and the higher ones deliberately go well past it, to find where
the model actually breaks.

Inference only, no GPU-memory concerns (one forward pass per severity, same
grid the model already runs at).
"""
from __future__ import annotations
import argparse
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
import torch.nn.functional as F

from train_query_decoder_overfit import load_json
import space_invariance_common as sic

NOISE_STDS = (0.05, 0.1, 0.2, 0.4)
BLUR_SIGMAS = (0.5, 1.0, 2.0, 4.0)
INTENSITY_SEVERITIES = (0.1, 0.25, 0.5, 0.75)
TRAINING_ENVELOPE = {"noise": 0.1, "blur": 1.0, "intensity": 0.25}  # from the encoder's own recipe -- see docstring


def gaussian_kernel_1d(sigma, device):
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum()), radius


def gaussian_blur3d(x, sigma, device):
    """Separable 3D Gaussian blur via three 1D depthwise convolutions."""
    if sigma <= 0:
        return x
    k, r = gaussian_kernel_1d(sigma, device)
    out = x
    for dim, kshape in ((2, (-1, 1, 1)), (3, (1, -1, 1)), (4, (1, 1, -1))):
        kernel = k.view(*[1, 1] + [1 if i != dim else -1 for i in range(2, 5)])
        pad = [0, 0, 0, 0, 0, 0]
        pad_idx = (4 - dim) * 2
        pad[pad_idx] = pad[pad_idx + 1] = r
        out = F.conv3d(F.pad(out, pad, mode="replicate"), kernel)
    return out


def apply_corruption(x, kind, severity, rng, device):
    g = torch.Generator(device="cpu").manual_seed(rng.randint(0, 2**31 - 1))
    if kind == "noise":
        noise = torch.randn(x.shape, generator=g).to(device) * severity
        return x + noise
    if kind == "blur":
        return gaussian_blur3d(x, severity, device)
    if kind == "intensity":
        mult = 1.0 + (torch.rand(1, generator=g).item() * 2 - 1) * severity
        add = (torch.rand(1, generator=g).item() * 2 - 1) * severity * 0.5
        return x * mult + add
    raise ValueError(kind)


def run_subject(model, x, gt, threshold, device, seed):
    rng = random.Random(seed)
    with torch.inference_mode():
        clean_logits = model(x.to(device), gt.shape)
    clean_mask = (clean_logits.sigmoid()[0, 0].cpu().numpy() > threshold)
    clean_dice = sic.dice(clean_mask, gt)

    rows = []
    for kind, severities in (("noise", NOISE_STDS), ("blur", BLUR_SIGMAS), ("intensity", INTENSITY_SEVERITIES)):
        for sev in severities:
            x_c = apply_corruption(x.to(device), kind, sev, rng, device)
            with torch.inference_mode():
                logits_c = model(x_c, gt.shape)
            mask_c = (logits_c.sigmoid()[0, 0].cpu().numpy() > threshold)
            dice_c = sic.dice(mask_c, gt)
            rows.append({"corruption": kind, "severity": sev,
                        "within_training_envelope": sev <= TRAINING_ENVELOPE[kind],
                        "clean_dice": clean_dice, "corrupted_dice": dice_c,
                        "dice_delta": dice_c - clean_dice})
    return rows, clean_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_task_joint.yaml")
    ap.add_argument("--task", choices=("brain", "lesion", "both"), default="both")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--out", default="outputs/noise_robustness")
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    config = load_json(ROOT / args.config)
    tile = tuple(config["tile_size"])
    threshold = config.get("inference_threshold", 0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = ("brain", "lesion") if args.task == "both" else (args.task,)

    all_summaries = {}
    for task in tasks:
        out = ROOT / args.out / task
        out.mkdir(parents=True, exist_ok=True)
        model, paths = sic.load_model(config, task, device)
        records = sic.subject_records(config, task, args.limit)

        rows = []
        for i, r in enumerate(records, 1):
            x, gt = sic.preprocess_to_grid(r["image"], r["mask"], paths, tile)
            subj_rows, clean_dice = run_subject(model, x, gt, threshold, device, args.seed + i)
            for row in subj_rows:
                row["subject"] = r["sid"]
            rows += subj_rows
            print(f"[{task} {i}/{len(records)}] {r['sid']} clean_dice={clean_dice:.4f} "
                  f"worst_corrupted={min(row['corrupted_dice'] for row in subj_rows):.4f}", flush=True)

        sic.write_csv(out / "per_subject_corruption_metrics.csv", rows)
        summary = aggregate(rows)
        sic.write_csv(out / "summary_by_corruption.csv", summary)
        (out / "summary.json").write_text(json.dumps({"task": task, "subjects": len(records), "rows": summary}, indent=2))
        all_summaries[task] = summary
        make_figure(summary, task, out / "noise_robustness.png")
        print(json.dumps(summary, indent=2))

    (ROOT / args.out / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    write_report(all_summaries, ROOT / args.out / "report.md")


def aggregate(rows):
    keys = sorted(set((r["corruption"], r["severity"]) for r in rows))
    out = []
    for kind, sev in keys:
        q = [r for r in rows if r["corruption"] == kind and r["severity"] == sev]
        out.append({"corruption": kind, "severity": sev,
                    "within_training_envelope": q[0]["within_training_envelope"],
                    "n": len(q),
                    "mean_clean_dice": float(np.mean([r["clean_dice"] for r in q])),
                    "mean_corrupted_dice": float(np.mean([r["corrupted_dice"] for r in q])),
                    "min_corrupted_dice": float(np.min([r["corrupted_dice"] for r in q])),
                    "mean_dice_delta": float(np.mean([r["dice_delta"] for r in q])),
                    "worst_dice_delta": float(np.min([r["dice_delta"] for r in q]))})
    return out


def make_figure(summary, task, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, kind, xlabel in zip(axes, ("noise", "blur", "intensity"),
                                ("Gaussian noise std", "blur sigma (voxels)", "intensity severity")):
        q = sorted([r for r in summary if r["corruption"] == kind], key=lambda r: r["severity"])
        xs = [r["severity"] for r in q]
        means = [r["mean_corrupted_dice"] for r in q]
        mins = [r["min_corrupted_dice"] for r in q]
        clean = q[0]["mean_clean_dice"] if q else None
        ax.axhline(clean, color="gray", ls="--", lw=1, label="clean-image Dice")
        ax.plot(xs, means, "o-", color="#C2620E", label="mean corrupted Dice")
        ax.plot(xs, mins, "o--", color="#A23434", label="worst-subject Dice")
        envelope = TRAINING_ENVELOPE[kind]
        ax.axvline(envelope, color="#2F7D50", ls=":", lw=1.3, label="encoder's own training envelope")
        ax.set(title=f"{task}: {kind}", xlabel=xlabel, ylabel="Dice vs expert")
        ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{task} query -- robustness to synthetic image degradation", fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(summaries, path):
    lines = ["# Synthetic noise / blur / intensity robustness",
             "",
             "Advisor-requested: how good is the current model for degraded images? Each "
             "corrupted prediction is scored directly against the expert mask (not "
             "self-consistency -- corruption isn't invertible) and compared to that same "
             "subject's own clean-image Dice. The **training envelope** line marks where "
             "the frozen encoder's own original training augmentation tops out "
             "(see `rs2net_original_training_technique.md`): noise std 0.1, blur sigma "
             "1.0, intensity severity 0.25 -- severities at or below that line are inside "
             "what the encoder has already seen; above it is genuinely out of distribution.",
             ""]
    for task, summary in summaries.items():
        lines += [f"## {task} query", "",
                  "| corruption | severity | mean Dice | worst subject | delta vs clean | in training envelope |",
                  "|---|---|---|---|---|---|"]
        for r in summary:
            lines.append(f"| {r['corruption']} | {r['severity']:g} | {r['mean_corrupted_dice']:.4f} | "
                         f"{r['min_corrupted_dice']:.4f} | {r['mean_dice_delta']:+.4f} | "
                         f"{'yes' if r['within_training_envelope'] else 'no'} |")
        lines.append("")
    lines += ["_Generated by `scripts/test_noise_robustness.py`; raw per-subject numbers in "
              "the CSVs beside this file._"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
