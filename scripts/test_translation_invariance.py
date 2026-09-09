#!/usr/bin/env python3
"""Translation-invariance test: does moving the same anatomy to a different
position inside the volume change the model's prediction, and where in the
architecture does the sensitivity enter?

Same design as test_orientation_invariance.py but the transform is a pure
shift (in physical millimetres, reported per axis) about the volume centre
instead of a rotation. A pure shift is not distorted by anisotropic spacing
the way a rotation is, but it still probes real behaviour: strided
downsampling in the frozen encoder makes CNN/Swin features only
*approximately* translation-equivariant, and any absolute-position signal in
the decoder trunk would show up here.

Outputs (under outputs/translation_invariance/<task>/): identical file set to
the orientation test - see that script's docstring.
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
from spatial_transform_utils import translation_grid, rotation_grid
import space_invariance_common as sic

AXES = ("axis0", "axis1", "axis2")
AXIS_ANATOMY = {"axis0": "L-R", "axis1": "D-V", "axis2": "R-C"}
DEFAULT_SHIFTS_MM = (1.0, 2.0, 4.0, 6.0)


def build_transforms(shifts_mm):
    transforms = [{
        "label": "resampling_floor",
        "fwd_grid_fn": lambda shape, spacing, device: rotation_grid(shape, spacing, "axis2", 0.0, device),
        "inv_grid_fn": lambda shape, spacing, device: rotation_grid(shape, spacing, "axis2", 0.0, device, invert=True),
    }]
    for axis in AXES:
        for mm in shifts_mm:
            transforms.append({
                "label": f"{axis}_{mm:+.0f}mm",
                "fwd_grid_fn": functools.partial(
                    lambda shape, spacing, device, a=axis, s=mm: translation_grid(shape, spacing, a, s, device)),
                "inv_grid_fn": functools.partial(
                    lambda shape, spacing, device, a=axis, s=mm: translation_grid(shape, spacing, a, s, device, invert=True)),
            })
    return transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_task_joint.yaml")
    ap.add_argument("--task", choices=("brain", "lesion", "both"), default="both")
    ap.add_argument("--limit", type=int, default=15, help="subjects per task (0 = all)")
    ap.add_argument("--shifts-mm", type=float, nargs="+", default=list(DEFAULT_SHIFTS_MM))
    ap.add_argument("--out", default="outputs/translation_invariance")
    args = ap.parse_args()

    config = load_json(ROOT / args.config)
    tile = tuple(config["tile_size"])
    spacing = tuple(config.get("model_spacing_mm", HIGHER_RESOLUTION_SPACING))
    threshold = config.get("inference_threshold", 0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = ("brain", "lesion") if args.task == "both" else (args.task,)
    transforms = build_transforms(args.shifts_mm)

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
            print(f"[{task} {i}/{len(records)}] {r['sid']} "
                  f"mean self-Dice={np.mean([m['self_consistency_dice'] for m in m_rows]):.4f}", flush=True)

        sic.write_csv(out / "per_transform_stage_metrics.csv", stage_rows)
        sic.write_csv(out / "per_transform_mask_metrics.csv", mask_rows)
        stage_summary = sic.aggregate_stage(stage_rows)
        mask_summary = sic.aggregate_mask(mask_rows)
        sic.write_csv(out / "summary_by_stage.csv", stage_summary)
        sic.write_csv(out / "summary_by_mask.csv", mask_summary)

        summary = {"task": task, "subjects": len(records), "shifts_mm": args.shifts_mm,
                   "axes": AXIS_ANATOMY, "by_stage": stage_summary, "by_mask": mask_summary}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries[task] = summary
        make_figure(stage_summary, mask_summary, args.shifts_mm, task, out / "translation_invariance.png")
        write_report(summary, out / "report.md")
        print(json.dumps(mask_summary, indent=2))

    (ROOT / args.out / "summary.json").write_text(json.dumps(all_summaries, indent=2))
    if len(all_summaries) > 1:
        sic.combined_report("translation", "mm", args.shifts_mm, all_summaries,
                            ROOT / args.out / "report.md")


def make_figure(stage_summary, mask_summary, shifts, task, path):
    stages = list(sic.SPATIAL_STAGES) + ["query_mask_embedding"]
    floor = {s["stage"]: s["mean_relative_l2"] for s in stage_summary if s["transform"] == "resampling_floor"}
    model_rows = [s for s in stage_summary if s["transform"] != "resampling_floor"]
    by_stage = {}
    for s in model_rows:
        by_stage.setdefault(s["stage"], []).append(s["mean_relative_l2"])
    means = [np.mean(by_stage.get(s, [np.nan])) for s in stages]
    floors = [floor.get(s, np.nan) for s in stages]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    xs = np.arange(len(stages))
    ax[0].bar(xs - 0.2, floors, 0.4, label="resampling floor (no model)")
    ax[0].bar(xs + 0.2, means, 0.4, label="under translation")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(stages, rotation=45, ha="right")
    ax[0].set(title=f"{task}: per-stage drift (relative L2, restored vs baseline)", ylabel="relative L2")
    ax[0].legend(); ax[0].grid(alpha=.3, axis="y")

    for axis in AXES:
        pts = []
        for mm in shifts:
            q = [m for m in mask_summary if m["transform"] == f"{axis}_{mm:+.0f}mm"]
            if q:
                pts.append((mm, q[0]["mean_self_consistency_dice"]))
        if pts:
            ax[1].plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=AXIS_ANATOMY[axis])
    ax[1].set(title=f"{task}: end-to-end self-consistency vs shift", xlabel="shift (mm)",
             ylabel="Dice(shifted→restored, baseline)")
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
        f"# Translation-invariance test - {task} query",
        "",
        f"**Subjects:** {summary['subjects']} held-out test cases. "
        f"**Shifts:** {', '.join(f'{s:g} mm' for s in summary['shifts_mm'])} along each of the three "
        f"grid axes ({', '.join(f'{k}={v}' for k, v in summary['axes'].items())}), about the volume centre.",
        "",
        "## How the pipeline is built (and where invariance could break)",
        "",
        "1. **Input** - resampled onto a fixed 192x128x160 grid at 0.167x0.200x0.160 mm.",
        "2. **Frozen RS2-Net encoder** - Swin-based 3D encoder, frozen. Five-level pyramid `level0` "
        "(full res, 48 ch) .. `level4` (1/16 res, 384 ch). Strided downsampling makes such encoders "
        "only *approximately* translation-equivariant: a shift that is not a multiple of a level's "
        "stride lands the anatomy on a different sub-grid phase, so features shift by a rounded "
        "amount and interpolate differently.",
        "3. **Decoder canvas trunk** (trainable, shared) - `canvas_level4` -> ... -> `canvas_level0` "
        "by trilinear upsample + 1x1x1 conv + skip-concat + residual block. Purely convolutional, so "
        "it carries no explicit position signal - any translation sensitivity here is inherited from "
        "the encoder or from boundary padding.",
        "4. **Query cross-attention** - `query_brain`/`query_lesion` attends to each canvas level, "
        "giving one non-spatial `query_mask_embedding`.",
        "5. **Output** - dot product of the query embedding with `canvas_level0`, + bias, 0.5 threshold.",
        "",
        "## Per-stage drift under translation",
        "",
        "Relative L2 between each stage's baseline output and its shifted output after the shift is "
        "undone. The **resampling floor** is the same with no model in the loop.",
        "",
        "| stage | resampling floor | mean under translation | max under translation |",
        "|---|---|---|---|",
    ]
    for stage in sic.SPATIAL_STAGES:
        lines.append(stage_line(stage))

    q_vals = [bs[(t, "query_mask_embedding")] for t in model_tfs if (t, "query_mask_embedding") in bs]
    if q_vals:
        lines += ["",
                  f"**Query embedding**: mean cosine similarity to baseline "
                  f"{np.mean([q['mean_cosine'] for q in q_vals]):.4f}, mean relative L2 "
                  f"{np.mean([q['mean_relative_l2'] for q in q_vals]):.4f}."]

    lines += ["",
              "## End-to-end effect on the prediction",
              "",
              "| shift | self-consistency Dice | min | mean accuracy delta vs expert | worst |",
              "|---|---|---|---|---|"]
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
        "- **Self-consistency Dice** near 1.0 means the shifted scan yields the same mask. The gap "
        "to `resampling_floor` is the model-attributable part.",
        "- If drift rises smoothly with shift size and tracks the encoder rows, the sub-grid-phase "
        "effect of strided downsampling is the explanation and it is expected to be small. A sharp "
        "jump at a particular shift, or drift concentrated in the canvas trunk, would indicate a "
        "learned position dependence worth investigating.",
        "- Compare axis by axis: the grid is anisotropic, so the same millimetre shift is a "
        "different number of voxels on each axis and may hit stride boundaries differently.",
        "",
        f"_Generated by `scripts/test_translation_invariance.py`; raw per-subject numbers in the "
        f"CSVs beside this file._",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
