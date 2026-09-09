#!/usr/bin/env python3
"""Orientation-invariance test: does rotating the same anatomy change the
model's prediction, and where in the architecture does the sensitivity enter?

For every test subject and every (axis, angle) the input volume is rotated by
a true anatomical rotation about the volume centre (spatial_transform_utils,
anisotropic-spacing-correct), pushed through the pipeline, and every tapped
stage is mapped back into the original reference frame and compared with the
untransformed baseline. See space_invariance_common for the stage list and
the resampling-floor control.

Outputs (under outputs/orientation_invariance/<task>/):
  per_transform_stage_metrics.csv   every subject x transform x stage
  per_transform_mask_metrics.csv    every subject x transform (Dice numbers)
  summary_by_stage.csv              mean drift per stage per transform
  summary_by_mask.csv               mean self-consistency + accuracy delta
  summary.json
  orientation_invariance.png        stage-drift + angle-response figures
  report.md                         full written explanation
"""
from __future__ import annotations
import argparse
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

from train_query_decoder_overfit import load_json
from higher_resolution_preprocessing import HIGHER_RESOLUTION_SPACING
from spatial_transform_utils import rotation_grid
import space_invariance_common as sic

AXES = ("axis0", "axis1", "axis2")
AXIS_ANATOMY = {"axis0": "L-R (sagittal-plane roll)", "axis1": "D-V (coronal-plane)", "axis2": "R-C (axial-plane yaw)"}
DEFAULT_ANGLES = (5.0, 10.0, 20.0, 30.0)


def build_transforms(angles):
    transforms = [{
        "label": "resampling_floor",
        "fwd_grid_fn": lambda shape, spacing, device: rotation_grid(shape, spacing, "axis2", 0.0, device),
        "inv_grid_fn": lambda shape, spacing, device: rotation_grid(shape, spacing, "axis2", 0.0, device, invert=True),
    }]
    for axis in AXES:
        for ang in angles:
            transforms.append({
                "label": f"{axis}_{ang:+.0f}deg",
                "fwd_grid_fn": functools.partial(
                    lambda shape, spacing, device, a=axis, g=ang: rotation_grid(shape, spacing, a, g, device)),
                "inv_grid_fn": functools.partial(
                    lambda shape, spacing, device, a=axis, g=ang: rotation_grid(shape, spacing, a, g, device, invert=True)),
            })
    return transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_task_joint.yaml")
    ap.add_argument("--task", choices=("brain", "lesion", "both"), default="both")
    ap.add_argument("--limit", type=int, default=15, help="subjects per task (0 = all)")
    ap.add_argument("--angles", type=float, nargs="+", default=list(DEFAULT_ANGLES))
    ap.add_argument("--out", default="outputs/orientation_invariance")
    args = ap.parse_args()

    config = load_json(ROOT / args.config)
    tile = tuple(config["tile_size"])
    spacing = tuple(config.get("model_spacing_mm", HIGHER_RESOLUTION_SPACING))
    threshold = config.get("inference_threshold", 0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = ("brain", "lesion") if args.task == "both" else (args.task,)
    transforms = build_transforms(args.angles)

    all_summaries = {}
    for task in tasks:
        out = ROOT / args.out / task
        out.mkdir(parents=True, exist_ok=True)
        model, paths = sic.load_model(config, task, device)
        records = sic.subject_records(config, task, args.limit)

        stage_rows, mask_rows = [], []
        for i, r in enumerate(records, 1):
            x, gt = sic.preprocess_to_grid(r["image"], r["mask"], paths, tile)
            s_rows, m_rows = sic.run_subject(model, x, gt, task, spacing, transforms, device, threshold)
            for row in s_rows + m_rows:
                row["subject"] = r["sid"]
            stage_rows += s_rows
            mask_rows += m_rows
            done = [m for m in m_rows]
            print(f"[{task} {i}/{len(records)}] {r['sid']} "
                  f"mean self-Dice={np.mean([m['self_consistency_dice'] for m in done]):.4f}", flush=True)

        sic.write_csv(out / "per_transform_stage_metrics.csv", stage_rows)
        sic.write_csv(out / "per_transform_mask_metrics.csv", mask_rows)
        stage_summary = sic.aggregate_stage(stage_rows)
        mask_summary = sic.aggregate_mask(mask_rows)
        sic.write_csv(out / "summary_by_stage.csv", stage_summary)
        sic.write_csv(out / "summary_by_mask.csv", mask_summary)

        summary = {"task": task, "subjects": len(records), "angles": args.angles,
                   "axes": {a: AXIS_ANATOMY[a] for a in AXES},
                   "by_stage": stage_summary, "by_mask": mask_summary}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries[task] = summary
        make_figure(stage_summary, mask_summary, args.angles, task, out / "orientation_invariance.png")
        write_report(summary, out / "report.md")
        print(json.dumps(mask_summary, indent=2))

    (ROOT / args.out / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    if len(all_summaries) > 1:
        sic.combined_report("rotation", "deg", args.angles, all_summaries,
                            ROOT / args.out / "report.md")


def make_figure(stage_summary, mask_summary, angles, task, path):
    stages = list(sic.SPATIAL_STAGES) + ["query_mask_embedding"]
    floor = {s["stage"]: s["mean_relative_l2"] for s in stage_summary if s["transform"] == "resampling_floor"}
    # average model transforms (exclude the floor) per stage
    model_rows = [s for s in stage_summary if s["transform"] != "resampling_floor"]
    by_stage = {}
    for s in model_rows:
        by_stage.setdefault(s["stage"], []).append(s["mean_relative_l2"])
    means = [np.mean(by_stage.get(s, [np.nan])) for s in stages]
    floors = [floor.get(s, np.nan) for s in stages]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    xs = np.arange(len(stages))
    ax[0].bar(xs - 0.2, floors, 0.4, label="resampling floor (no model)")
    ax[0].bar(xs + 0.2, means, 0.4, label="under rotation")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(stages, rotation=45, ha="right")
    ax[0].set(title=f"{task}: per-stage drift (relative L2, restored vs baseline)", ylabel="relative L2")
    ax[0].legend(); ax[0].grid(alpha=.3, axis="y")

    for axis in AXES:
        pts = []
        for ang in angles:
            q = [m for m in mask_summary if m["transform"] == f"{axis}_{ang:+.0f}deg"]
            if q:
                pts.append((ang, q[0]["mean_self_consistency_dice"]))
        if pts:
            ax[1].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=AXIS_ANATOMY[axis])
    ax[1].set(title=f"{task}: end-to-end self-consistency vs rotation", xlabel="rotation (deg)",
             ylabel="Dice(rotated→restored, baseline)")
    ax[1].set_ylim(0, 1.02); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(summary, path):
    task = summary["task"]
    bs = {(s["transform"], s["stage"]): s for s in summary["by_stage"]}
    bm = {m["transform"]: m for m in summary["by_mask"]}
    floor = {st: bs[("resampling_floor", st)] for _, st in bs if ("resampling_floor", st) in bs}

    model_tfs = sorted({t for t, _ in bs} - {"resampling_floor"}, key=sic.transform_sort_key)

    def stage_line(stage):
        vals = [bs[(t, stage)]["mean_relative_l2"] for t in model_tfs if (t, stage) in bs]
        f = floor.get(stage, {}).get("mean_relative_l2", float("nan"))
        return f"| `{stage}` | {f:.4f} | {np.mean(vals):.4f} | {np.max(vals):.4f} |"

    lines = [
        f"# Orientation-invariance test - {task} query",
        "",
        f"**Subjects:** {summary['subjects']} held-out test cases. "
        f"**Rotations:** {', '.join(f'{a:g}deg' for a in summary['angles'])} about each of the three "
        f"anatomical axes ({', '.join(f'{k}={v}' for k, v in summary['axes'].items())}), "
        f"applied about the volume centre in true physical millimetres.",
        "",
        "## How the pipeline is built (and where invariance could break)",
        "",
        "1. **Input** - the scan is resampled onto a fixed model grid "
        "(192x128x160 voxels, 0.167x0.200x0.160 mm).",
        "2. **Frozen RS2-Net encoder** - a Swin-based 3D encoder, weights frozen. It emits a "
        "five-level feature pyramid: `level0` at full resolution (48 channels) down to `level4` at "
        "1/16 resolution (384 channels). Convolutional/attention encoders are only *approximately* "
        "translation-equivariant (padding, strided downsampling) and are **not** rotation-equivariant "
        "by construction, so this is the first place pose can leak in.",
        "3. **Decoder canvas trunk** (trainable, shared by both queries) - starts from the `level4` "
        "features and upsamples coarse->fine (`canvas_level4` -> ... -> `canvas_level0`), each step = "
        "trilinear interpolation + 1x1x1 conv + skip-concat + residual block. The `canvas_level0` "
        "tensor (48 channels, full resolution) is the spatial map the final decision reads.",
        "4. **Query cross-attention** - the task query vector (`query_brain` or `query_lesion`) "
        "cross-attends to every canvas level in turn, producing one non-spatial `query_mask_embedding` "
        "vector. If the mechanism is pose-invariant this vector should barely move when the input is "
        "rotated; if it moves a lot, the query is 'seeing' orientation.",
        "5. **Output** - `output_logits = einsum(level0_embedding, canvas_level0)` + bias, then a "
        "0.5 threshold gives `output_mask`.",
        "",
        "## Per-stage drift under rotation",
        "",
        "Relative L2 between each stage's baseline output and its rotated output after the rotation "
        "is undone (0 = identical). The **resampling floor** column is the same measurement with no "
        "model in the loop (transform the input forward then straight back) - drift at or near the "
        "floor is interpolation artefact, not model sensitivity.",
        "",
        "| stage | resampling floor | mean under rotation | max under rotation |",
        "|---|---|---|---|",
    ]
    for stage in sic.SPATIAL_STAGES:
        lines.append(stage_line(stage))
    # query embedding
    q_vals = [bs[(t, "query_mask_embedding")] for t in model_tfs if (t, "query_mask_embedding") in bs]
    if q_vals:
        lines += [
            "",
            f"**Query embedding** (`query_mask_embedding`, compared directly - no re-alignment): "
            f"mean cosine similarity to baseline {np.mean([q['mean_cosine'] for q in q_vals]):.4f}, "
            f"mean relative L2 {np.mean([q['mean_relative_l2'] for q in q_vals]):.4f}.",
        ]

    lines += [
        "",
        "## End-to-end effect on the prediction",
        "",
        "| rotation | self-consistency Dice | min | mean accuracy delta vs expert | worst |",
        "|---|---|---|---|---|",
    ]
    for tf in model_tfs:
        m = bm[tf]
        lines.append(f"| `{tf}` | {m['mean_self_consistency_dice']:.4f} | {m['min_self_consistency_dice']:.4f} "
                     f"| {m['mean_accuracy_delta']:+.4f} | {m['worst_accuracy_delta']:+.4f} |")
    fl = bm.get("resampling_floor")
    if fl:
        lines.append(f"| `resampling_floor` | {fl['mean_self_consistency_dice']:.4f} | "
                     f"{fl['min_self_consistency_dice']:.4f} | {fl['mean_accuracy_delta']:+.4f} | "
                     f"{fl['worst_accuracy_delta']:+.4f} |")

    lines += [
        "",
        "## Reading this",
        "",
        "- **Self-consistency Dice** is the orientation-invariance number: 1.0 = the rotated scan "
        "produces exactly the same mask. The gap between it and the `resampling_floor` row is the "
        "part attributable to the model rather than to interpolation.",
        "- The per-stage table shows *where* drift is introduced. Compare the encoder rows against "
        "the canvas rows: if drift is already large at `encoder_level4` and only grows mildly "
        "afterwards, the frozen encoder is the dominant source and the trainable decoder mostly "
        "inherits it. If drift jumps inside the canvas trunk, the decoder is adding sensitivity.",
        "- The **query embedding** cosine says whether the conditioning mechanism itself is "
        "orientation-sensitive: a cosine near 1.0 means the query reads essentially the same summary "
        "regardless of pose; a lower cosine means orientation changes what the query attends to.",
        "",
        f"_Generated by `scripts/test_orientation_invariance.py`; raw per-subject numbers in the "
        f"CSVs beside this file._",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
