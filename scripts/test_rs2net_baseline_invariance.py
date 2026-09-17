#!/usr/bin/env python3
"""Space-invariance test for the ORIGINAL RS2-Net model (their own encoder AND
their own decoder, end to end) -- not this project's TwoTaskLevel0OneQueryMask
Decoder. Brain segmentation only (CAMRI + mouse); RS2-Net has no lesion task.

Motivation: rs2net_original_training_technique.md established that the paper's
own training used real rotation augmentation (+/-30 deg per axis, 20% of
samples) -- unlike this project's own decoder, which uses none. This script
asks the direct question that comparison raises: does their rotation-
augmented full pipeline actually hold up better under the exact same
rotation/translation stress test this project already ran on its own model?

Same method as test_orientation_invariance.py / test_translation_invariance.py:
baseline prediction, transform the input, run the model, transform the output
back into the reference frame with the exact inverse, compare. Runs at RS2-Net's
own native resolution (128x128x160, 0.25x0.200x0.160 mm spacing) -- NOT this
project's higher-resolution grid -- since that is what their model, and their
rotation-augmented training, actually target.

Per-stage detail is narrower than the two-task model's report: RS2-Net's own
decoder (RSSNet.decoder1-4) is a different architecture with no query
mechanism and no equivalent "canvas" decomposition, so only the shared,
comparable stages are tapped: the five encoder feature levels (identical
extraction to RS2NetEncoderAdapter) and the final output. This still directly
answers the question that matters: does their encoder's own rotation-augmented
training measurably reduce the encoder-level drift this project's frozen-
encoder analysis already measured?

Inference only, no GPU-memory concerns beyond a single forward pass per
transform (their patch size IS the whole grid, same as this project's model).
"""
from __future__ import annotations
import argparse
import csv
import functools
import json
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

from corrected_label_preprocessing import preprocess_image_and_corrected_target
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import load_json
from spatial_transform_utils import rotation_grid, translation_grid, apply_grid
import space_invariance_common as sic  # dice(), transform_sort_key(), write_csv()

NATIVE_TILE = (128, 128, 160)
NATIVE_SPACING = (0.25, 0.20000000298023224, 0.1599999964237213)
ENCODER_STAGES = ("encoder_level0", "encoder_level1", "encoder_level2", "encoder_level3", "encoder_level4")
STAGES = ("input",) + ENCODER_STAGES + ("output_logits",)
AXES = ("axis0", "axis1", "axis2")


def encoder_taps(network, volume):
    """Identical extraction to RS2NetEncoderAdapter.forward -- the same five
    levels this project's own encoder-drift analysis already measured, so the
    two are directly comparable."""
    hidden = network.swinViT(volume, network.normalize)
    return {
        "encoder_level0": network.encoder1(volume),
        "encoder_level1": network.encoder2(hidden[0]),
        "encoder_level2": network.encoder3(hidden[1]),
        "encoder_level3": hidden[2],
        "encoder_level4": network.feature(hidden[3]),
    }


@torch.inference_mode()
def full_pipeline_taps(network, volume):
    taps = {"input": volume}
    taps.update(encoder_taps(network, volume))
    taps["output_logits"] = network(volume)
    return taps


def build_transforms(mode, magnitudes):
    if mode == "rotation":
        transforms = [{"label": "resampling_floor",
                       "fwd_grid_fn": lambda shape, spacing, device: rotation_grid(shape, spacing, "axis2", 0.0, device),
                       "inv_grid_fn": lambda shape, spacing, device: rotation_grid(shape, spacing, "axis2", 0.0, device, invert=True)}]
        for axis in AXES:
            for ang in magnitudes:
                transforms.append({
                    "label": f"{axis}_{ang:+.0f}deg",
                    "fwd_grid_fn": functools.partial(lambda shape, spacing, device, a=axis, g=ang: rotation_grid(shape, spacing, a, g, device)),
                    "inv_grid_fn": functools.partial(lambda shape, spacing, device, a=axis, g=ang: rotation_grid(shape, spacing, a, g, device, invert=True)),
                })
    else:
        transforms = [{"label": "resampling_floor",
                       "fwd_grid_fn": lambda shape, spacing, device: rotation_grid(shape, spacing, "axis2", 0.0, device),
                       "inv_grid_fn": lambda shape, spacing, device: rotation_grid(shape, spacing, "axis2", 0.0, device, invert=True)}]
        for axis in AXES:
            for mm in magnitudes:
                transforms.append({
                    "label": f"{axis}_{mm:+.0f}mm",
                    "fwd_grid_fn": functools.partial(lambda shape, spacing, device, a=axis, s=mm: translation_grid(shape, spacing, a, s, device)),
                    "inv_grid_fn": functools.partial(lambda shape, spacing, device, a=axis, s=mm: translation_grid(shape, spacing, a, s, device, invert=True)),
                })
    return transforms


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


def run_subject(network, image_path, mask_path, paths, transforms, device, threshold):
    data, target, _, _ = preprocess_image_and_corrected_target(image_path, mask_path, paths, NATIVE_TILE)
    x = data.to(device)
    gt = target[0, 0].numpy() > 0.5
    grid_shape = tuple(x.shape[-3:])

    base = full_pipeline_taps(network, x)
    base_mask = (base["output_logits"].sigmoid()[0, 0].cpu().numpy() > threshold)
    base_dice_gt = sic.dice(base_mask, gt)

    stage_rows, mask_rows = [], []
    for tf in transforms:
        g_fwd, _ = sic.stage_grid(x, grid_shape, NATIVE_SPACING, tf["fwd_grid_fn"], device)
        x_t = apply_grid(x, g_fwd, mode="bilinear")
        t = full_pipeline_taps(network, x_t)

        for stage in STAGES:
            a = base[stage]; b = t[stage]
            g_inv, _ = sic.stage_grid(a, grid_shape, NATIVE_SPACING, tf["inv_grid_fn"], device)
            b_aligned = apply_grid(b.float(), g_inv, mode="bilinear")
            rel_l2, cos = sic.compare_spatial(a, b_aligned, device)
            stage_rows.append({"transform": tf["label"], "stage": stage, "relative_l2": rel_l2, "cosine": cos})

        g_inv, _ = sic.stage_grid(t["output_logits"], grid_shape, NATIVE_SPACING, tf["inv_grid_fn"], device)
        logits_aligned = apply_grid(t["output_logits"], g_inv, mode="bilinear")
        mask_t = (logits_aligned.sigmoid()[0, 0].cpu().numpy() > threshold)
        mask_rows.append({"transform": tf["label"],
                          "self_consistency_dice": sic.dice(mask_t, base_mask),
                          "transformed_vs_expert_dice": sic.dice(mask_t, gt),
                          "baseline_vs_expert_dice": base_dice_gt,
                          "accuracy_delta": sic.dice(mask_t, gt) - base_dice_gt})
    return stage_rows, mask_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("rotation", "translation"), required=True)
    ap.add_argument("--domain", choices=("camri", "mouse", "both"), default="both")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--magnitudes", type=float, nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    magnitudes = args.magnitudes or ([5.0, 10.0, 20.0, 30.0] if args.mode == "rotation" else [1.0, 2.0, 4.0, 6.0])
    out_root = ROOT / (args.out or f"outputs/rs2net_baseline_invariance/{args.mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_cfg = load_json(ROOT / "configs/rs2net_encoder.yaml")
    paths = RS2NetPaths.from_config(enc_cfg)
    adapter = RS2NetEncoderAdapter(paths, image_size=NATIVE_TILE, in_channels=1, out_channels=1, feature_size=48)
    network = adapter.network.to(device).eval()
    for p in network.parameters():
        p.requires_grad_(False)

    transforms = build_transforms(args.mode, magnitudes)
    threshold = 0.5
    domains = ("camri", "mouse") if args.domain == "both" else (args.domain,)

    all_summaries = {}
    for domain in domains:
        out = out_root / domain
        out.mkdir(parents=True, exist_ok=True)
        records = subject_records(domain, args.limit)
        stage_rows, mask_rows = [], []
        for i, r in enumerate(records, 1):
            s_rows, m_rows = run_subject(network, r["image"], r["mask"], paths, transforms, device, threshold)
            for row in s_rows + m_rows:
                row["subject"] = r["sid"]
            stage_rows += s_rows; mask_rows += m_rows
            print(f"[rs2net-baseline {domain} {i}/{len(records)}] {r['sid']} "
                  f"mean self-Dice={np.mean([m['self_consistency_dice'] for m in m_rows]):.4f}", flush=True)

        sic.write_csv(out / "per_transform_stage_metrics.csv", stage_rows)
        sic.write_csv(out / "per_transform_mask_metrics.csv", mask_rows)
        stage_summary = sic.aggregate_stage(stage_rows)
        mask_summary = sic.aggregate_mask(mask_rows)
        sic.write_csv(out / "summary_by_stage.csv", stage_summary)
        sic.write_csv(out / "summary_by_mask.csv", mask_summary)
        summary = {"domain": domain, "model": "RS2-Net original (encoder+decoder, no query)",
                   "mode": args.mode, "subjects": len(records), "magnitudes": magnitudes,
                   "by_stage": stage_summary, "by_mask": mask_summary}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries[domain] = summary
        make_figure(stage_summary, mask_summary, magnitudes, domain, args.mode, out / "rs2net_baseline_invariance.png")
        print(json.dumps(mask_summary, indent=2))

    (out_root / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    write_comparison_report(all_summaries, args.mode, out_root / "report.md")


def make_figure(stage_summary, mask_summary, magnitudes, domain, mode, path):
    unit = "deg" if mode == "rotation" else "mm"
    floor = {s["stage"]: s["mean_relative_l2"] for s in stage_summary if s["transform"] == "resampling_floor"}
    model_rows = [s for s in stage_summary if s["transform"] != "resampling_floor"]
    by_stage = {}
    for s in model_rows:
        by_stage.setdefault(s["stage"], []).append(s["mean_relative_l2"])
    stages = list(STAGES)
    means = [np.mean(by_stage.get(s, [np.nan])) for s in stages]
    floors = [floor.get(s, np.nan) for s in stages]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    xs = np.arange(len(stages))
    ax[0].bar(xs - 0.2, floors, 0.4, label="resampling floor")
    ax[0].bar(xs + 0.2, means, 0.4, label=f"under {mode}")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(stages, rotation=40, ha="right")
    ax[0].set(title=f"RS2-Net baseline ({domain}): per-stage drift", ylabel="relative L2")
    ax[0].legend(); ax[0].grid(alpha=.3, axis="y")

    for axis in AXES:
        pts = []
        for m in magnitudes:
            q = [x for x in mask_summary if x["transform"] == f"{axis}_{m:+.0f}{unit}"]
            if q:
                pts.append((m, q[0]["mean_self_consistency_dice"]))
        if pts:
            ax[1].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=axis)
    ax[1].set(title=f"RS2-Net baseline ({domain}): self-consistency vs {mode}", xlabel=f"{mode} ({unit})",
             ylabel="Dice")
    ax[1].set_ylim(0, 1.02); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_comparison_report(summaries, mode, path):
    """Direct numeric comparison against this project's own two-task model,
    read from the already-committed orientation/translation invariance
    reports for the SAME mode, brain query, mouse domain (the only domain
    that test covered)."""
    own_path = ROOT / f"outputs/{'orientation' if mode == 'rotation' else 'translation'}_invariance/brain/summary_by_mask.csv"
    own_rows = list(csv.DictReader(open(own_path))) if own_path.exists() else []
    own_model_tfs = [r for r in own_rows if r["transform"] != "resampling_floor"]
    own_mean = float(np.mean([float(r["mean_self_consistency_dice"]) for r in own_model_tfs])) if own_model_tfs else float("nan")

    lines = [f"# RS2-Net baseline {mode}-invariance vs. this project's model",
             "",
             "Same rotation/translation stress test as `test_orientation_invariance.py` / "
             "`test_translation_invariance.py`, run against the ORIGINAL RS2-Net model "
             "(their own encoder AND their own decoder, no query mechanism) instead of "
             "this project's TwoTaskLevel0OneQueryMaskDecoder. Brain segmentation only "
             "(CAMRI + mouse) -- RS2-Net has no lesion task.",
             "",
             "| domain | mean self-consistency Dice | mean accuracy delta vs expert |",
             "|---|---|---|"]
    for domain, s in summaries.items():
        rows = [m for m in s["by_mask"] if m["transform"] != "resampling_floor"]
        lines.append(f"| {domain} | {np.mean([m['mean_self_consistency_dice'] for m in rows]):.4f} | "
                     f"{np.mean([m['mean_accuracy_delta'] for m in rows]):+.4f} |")
    lines += ["",
              "## Versus this project's own two-task model (brain query, mouse domain)",
              "",
              f"- RS2-Net baseline (their encoder + their decoder, rotation-augmented training): "
              f"see table above.",
              f"- This project's model (frozen RS2-Net encoder + this project's own decoder, "
              f"NO rotation augmentation): mean self-consistency Dice **{own_mean:.4f}** "
              f"(`outputs/{'orientation' if mode=='rotation' else 'translation'}_invariance/brain/summary_by_mask.csv`).",
              "",
              "If the baseline number above is meaningfully higher than this project's own "
              f"{own_mean:.4f}, that is direct evidence the paper's rotation-augmented "
              "training measurably helps -- and a concrete case for adding the same kind of "
              "augmentation to this project's own decoder training. If the two are close, "
              "the gap this project measured is coming from somewhere other than missing "
              "rotation augmentation (e.g. the decoder's skip-fusion compounding, or the "
              "lesion task specifically, which this baseline cannot speak to at all).",
              ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
