#!/usr/bin/env python3
"""Orientation-invariance test for TwoTaskLevel0OneQueryMaskDecoder.

Question (from the advisor): is the model "space invariant"? This script
answers the ORIENTATION half -- does rotating the same anatomy change the
prediction? -- and localizes WHERE in the pipeline any sensitivity is
introduced (frozen encoder features vs. final output), not just end-to-end
Dice.

Method, per subject and per (axis, angle):
  1. Preprocess the case onto the model grid (192x128x160,
     0.167x0.200x0.160 mm) -> reference tensor ``x`` and grid-space expert
     mask ``gt``.
  2. Baseline prediction ``pred0`` from ``x``.
  3. Rotate ``x`` by ``angle`` about ``axis`` using
     spatial_transform_utils (anisotropic-spacing-correct, true anatomical
     rotation about the volume centre), run the model, then rotate the
     output logits back into the reference frame with the exact inverse
     grid. This isolates the model's behaviour from the resampling: an
     ideally orientation-invariant model gives the same mask after the
     round trip.
  4. Report three Dice numbers: self-consistency (rotated-and-restored
     prediction vs. baseline prediction), rotated-vs-expert, and
     baseline-vs-expert. Self-consistency is the invariance metric; the
     other two show whether rotation helps or hurts real accuracy.
  5. Per-stage probe: encode the centre tile of ``x`` and of the rotated
     ``x``, rotate the rotated features back at their own (downsampled)
     resolution, and report per-level relative L2 and cosine similarity.
     Compared against the same relative L2 on the final logits, this shows
     how much drift the encoder alone contributes versus the decoder.

Not a training script: every prediction is produced under
``torch.inference_mode`` from a strictly loaded checkpoint.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
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

import evaluate_external_holdout as eeh
eeh.FEATURE_NAMES = ("level0", "level1", "level2", "level3", "level4")
from evaluate_external_holdout import sliding_window_logits
from higher_resolution_preprocessing import HIGHER_RESOLUTION_SPACING
from models.query_mask_decoder import TaskFixedFrozenEncoderQueryModel, TwoTaskLevel0OneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import load_json
from spatial_transform_utils import rotation_grid, apply_grid

FEATURE_NAMES = ("level0", "level1", "level2", "level3", "level4")
AXES = ("axis0", "axis1", "axis2")
DEFAULT_ANGLES = (-30.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 30.0)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * (a & b).sum() / denom)


def preprocess_with_seg(image_path, mask_path, paths, tile_size):
    """Same as evaluate_external_holdout.preprocess but also returns the
    expert segmentation already on the model grid, so rotation metrics can
    be scored against ground truth without a native-space round trip."""
    from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from RS2.utilities.plans_handling.plans_handler import PlansManager

    json_root = paths.baseline_project / "RS2" / "jsons"
    plans = load_json(json_root / "plans.json"); dataset = load_json(json_root / "dataset.json")
    manager = PlansManager(plans); configuration = manager.get_configuration("3d_fullres")
    data, segmentation, properties = DefaultPreprocessor(verbose=False).run_case(
        [str(image_path)], str(mask_path), manager, configuration, dataset
    )
    image = torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0)
    seg = torch.from_numpy((np.asarray(segmentation) > 0).astype(np.float32)).unsqueeze(0)
    return image, seg, properties


def probe_encoder(model, x, x_rot, shape, axis, angle, device):
    """Per-level relative L2 and cosine similarity between the baseline
    encoder features and the rotated features rotated back to the
    reference frame, at each level's own resolution."""
    with torch.inference_mode():
        f0 = model.encode(x.to(device))
        fr = model.encode(x_rot.to(device))
    out = {}
    for name in FEATURE_NAMES:
        a = f0[name]; b = fr[name]
        fshape = tuple(a.shape[-3:])
        scale = [s / fs for s, fs in zip(shape, fshape)]
        fspacing = [sp * sc for sp, sc in zip(HIGHER_RESOLUTION_SPACING, scale)]
        g_inv = rotation_grid(fshape, fspacing, axis, angle, device, invert=True)
        b_aligned = apply_grid(b.float(), g_inv, mode="bilinear")
        # ignore the thin border the inverse resample cannot fill
        m = torch.zeros(fshape, dtype=torch.bool, device=device)
        m[2:-2, 2:-2, 2:-2] = True
        a_v = a.float()[..., m]; b_v = b_aligned[..., m]
        rel_l2 = float((a_v - b_v).norm() / a_v.norm().clamp_min(1e-8))
        cos = float(torch.nn.functional.cosine_similarity(a_v.reshape(-1), b_v.reshape(-1), dim=0))
        out[name] = {"relative_l2": rel_l2, "cosine": cos}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_task_joint.yaml")
    ap.add_argument("--task", choices=("brain", "lesion"), default="lesion")
    ap.add_argument("--limit", type=int, default=12, help="subjects to test (0 = all)")
    ap.add_argument("--angles", type=float, nargs="+", default=list(DEFAULT_ANGLES))
    ap.add_argument("--out", default="outputs/orientation_invariance")
    args = ap.parse_args()

    config = load_json(ROOT / args.config)
    tile = tuple(config["tile_size"])
    spacing = tuple(config.get("model_spacing_mm", HIGHER_RESOLUTION_SPACING))
    OUT = ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    ck = torch.load(ROOT / config["output_directory"] / "checkpoints/best_two_task_joint_decoder.pt",
                    map_location="cpu", weights_only=False)
    decoder = TwoTaskLevel0OneQueryMaskDecoder(config["embedding_dim"], 4)
    decoder.load_state_dict(ck["decoder_state_dict"], strict=True)
    encoder = RS2NetEncoderAdapter(paths, image_size=tile, in_channels=1, out_channels=1, feature_size=48)
    model = TaskFixedFrozenEncoderQueryModel(encoder, decoder, args.task).to(device).eval()

    # subject list
    if args.task == "lesion":
        split = load_json(ROOT / config["stroke_split"])
        raw = Path(config["stroke_raw_root"])
        records = [{"sid": s, "image": raw / s / "t2.nii", "mask": raw / s / "masklesion_manual.nii"}
                   for s in split["test"]]
    else:
        split = load_json(ROOT / config["mouse_split"])
        src = {r["scan_id"]: r for r in csv.DictReader(open(ROOT / config["mouse_metrics"]))}
        records = [{"sid": s, "image": Path(src[s]["image_path"]), "mask": Path(src[s]["ground_truth_path"])}
                   for s in split["test"]["scans"]]
    if args.limit:
        records = records[: args.limit]

    rows, probe_rows = [], []
    for i, r in enumerate(records, 1):
        x, gt, _ = preprocess_with_seg(r["image"], r["mask"], paths, tile)
        shape = tuple(x.shape[-3:])
        gt_np = gt[0, 0].numpy() > 0.5
        with torch.inference_mode():
            base_logits = sliding_window_logits(model, x, tile, device)
        pred0 = (base_logits.sigmoid()[0, 0].cpu().numpy() > config["inference_threshold"])
        base_dice_gt = dice(pred0, gt_np)

        for axis in AXES:
            for angle in args.angles:
                g_fwd = rotation_grid(shape, spacing, axis, angle, device, invert=False)
                g_inv = rotation_grid(shape, spacing, axis, angle, device, invert=True)
                x_rot = apply_grid(x.to(device), g_fwd, mode="bilinear").cpu()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    logits_rot = sliding_window_logits(model, x_rot, tile, device)
                logits_aligned = apply_grid(logits_rot, g_inv, mode="bilinear")
                pred_rot = (logits_aligned.sigmoid()[0, 0].cpu().numpy() > config["inference_threshold"])

                border = np.zeros(shape, bool); border[4:-4, 4:-4, 4:-4] = True
                self_dice = dice(pred_rot & border, pred0 & border)
                rot_dice_gt = dice(pred_rot, gt_np)
                out_rel_l2 = float(np.linalg.norm((logits_aligned - base_logits).cpu().numpy()[..., border])
                                   / max(np.linalg.norm(base_logits.cpu().numpy()[..., border]), 1e-8))
                rows.append({"subject": r["sid"], "task": args.task, "axis": axis, "angle_deg": angle,
                             "self_consistency_dice": self_dice, "rotated_vs_expert_dice": rot_dice_gt,
                             "baseline_vs_expert_dice": base_dice_gt,
                             "accuracy_delta": rot_dice_gt - base_dice_gt,
                             "output_logit_relative_l2": out_rel_l2,
                             "seconds": time.perf_counter() - t0})

                enc = probe_encoder(model, x, x_rot, shape, axis, angle, device)
                for lvl, v in enc.items():
                    probe_rows.append({"subject": r["sid"], "task": args.task, "axis": axis,
                                       "angle_deg": angle, "stage": lvl, **v})
                probe_rows.append({"subject": r["sid"], "task": args.task, "axis": axis,
                                   "angle_deg": angle, "stage": "output_logits",
                                   "relative_l2": out_rel_l2, "cosine": float("nan")})
                print(f"[{i}/{len(records)}] {r['sid']} {axis} {angle:+.0f}deg "
                      f"self={self_dice:.4f} rotΔacc={rot_dice_gt - base_dice_gt:+.4f}", flush=True)

    def write(name, data):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

    write("per_rotation_metrics.csv", rows)
    write("per_stage_probe.csv", probe_rows)

    # aggregate: mean self-consistency and accuracy delta by |angle|
    agg = {}
    for row in rows:
        k = abs(row["angle_deg"])
        agg.setdefault(k, []).append(row)
    summary = []
    for k in sorted(agg):
        q = agg[k]
        summary.append({"abs_angle_deg": k, "n": len(q),
                        "mean_self_consistency_dice": float(np.mean([x["self_consistency_dice"] for x in q])),
                        "min_self_consistency_dice": float(np.min([x["self_consistency_dice"] for x in q])),
                        "mean_accuracy_delta": float(np.mean([x["accuracy_delta"] for x in q])),
                        "mean_output_relative_l2": float(np.mean([x["output_logit_relative_l2"] for x in q]))})
    write("summary_by_angle.csv", summary)

    stage_order = list(FEATURE_NAMES) + ["output_logits"]
    stage_summary = []
    for stage in stage_order:
        q = [x for x in probe_rows if x["stage"] == stage]
        stage_summary.append({"stage": stage, "n": len(q),
                              "mean_relative_l2": float(np.mean([x["relative_l2"] for x in q])),
                              "mean_cosine": float(np.nanmean([x["cosine"] for x in q]))})
    write("summary_by_stage.csv", stage_summary)

    (OUT / "summary.json").write_text(json.dumps(
        {"task": args.task, "subjects": len(records), "angles": args.angles,
         "by_angle": summary, "by_stage": stage_summary}, indent=2))

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot([s["abs_angle_deg"] for s in summary], [s["mean_self_consistency_dice"] for s in summary],
               "o-", label="mean")
    ax[0].plot([s["abs_angle_deg"] for s in summary], [s["min_self_consistency_dice"] for s in summary],
               "o--", label="min")
    ax[0].set(title="Self-consistency vs. rotation magnitude", xlabel="|rotation| (deg)",
              ylabel="Dice(rotated→restored, baseline)"); ax[0].set_ylim(0, 1.02); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].bar([s["stage"] for s in stage_summary], [s["mean_relative_l2"] for s in stage_summary])
    ax[1].set(title="Where rotation drift enters", ylabel="mean relative L2 (rotated→restored vs baseline)")
    ax[1].tick_params(axis="x", rotation=30)
    fig.savefig(OUT / "orientation_invariance.png", dpi=200, bbox_inches="tight")

    print(json.dumps(summary, indent=2))
    print(json.dumps(stage_summary, indent=2))


if __name__ == "__main__":
    main()
