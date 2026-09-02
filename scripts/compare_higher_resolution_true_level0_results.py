#!/usr/bin/env python3
"""Paired 128x128x160-baseline versus 192x128x160-higher-resolution TRUE
level0 diagnostic evidence.

Adapted from scripts/compare_true_full_resolution_level0_results.py, pointed
at outputs/higher_resolution_true_level0/. Analysis logic (contour geometry,
factor-N quantization, probability detail, paired figures) is reused
unchanged from diagnose_model_spatial_resolution.py and
analyze_boundary_error_diagnostics.py so all three level0 experiments remain
comparable on identical methodology.
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from analyze_boundary_error_diagnostics import align_probability_to_raw
from diagnose_model_spatial_resolution import digital_contour_statistics, quantization_rows, effective_native_footprint, probability_detail_statistics

OUT = ROOT / "outputs/higher_resolution_true_level0"


def rows(p):
    return list(csv.DictReader(open(p)))


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def geometry(records):
    geometry, quant, probability = [], [], []
    for i, r in enumerate(records, 1):
        gt = np.asarray(nib.load(r["ground_truth_path"]).dataobj) > 0
        spacing = tuple(map(float, nib.load(r["ground_truth_path"]).header.get_zooms()[:3]))
        foot, _ = effective_native_footprint(spacing)
        for condition, pred_key, raw_key, prob_key in (
            ("baseline_128", "baseline_filtered_prediction_path", "baseline_raw_prediction_path", "baseline_probability_path"),
            ("higher_res_192", "higher_res_filtered_prediction_path", "higher_res_raw_prediction_path", "higher_res_probability_path"),
        ):
            pred = np.asarray(nib.load(r[pred_key]).dataobj) > 0
            raw = np.asarray(nib.load(r[raw_key]).dataobj) > 0
            per = defaultdict(list)
            for z in np.where(gt.any((0, 1)))[0]:
                for kind, mask in (("expert", gt[:, :, z]), ("prediction", pred[:, :, z])):
                    q = digital_contour_statistics(mask)
                    if q:
                        for k, v in q.items():
                            per[(kind, k)].append(v)
            item = {"condition": condition, "domain": r["domain"], "subject": r["subject"]}
            for kind in ("expert", "prediction"):
                for k in sorted({x[1] for x in per}):
                    item[f"{kind}_{k}"] = float(np.mean(per[(kind, k)]))
            for k in sorted({x[1] for x in per}):
                item[f"prediction_expert_{k}_ratio"] = item[f"prediction_{k}"] / max(item[f"expert_{k}"], 1e-12)
            geometry.append(item)
            qq = quantization_rows(r["domain"], r["subject"], gt, pred, foot)
            for x in qq:
                x["condition"] = condition
            quant += qq
            saved = np.asarray(nib.load(r[prob_key]).dataobj, np.float32)
            prob, _ = align_probability_to_raw(saved, raw)
            ps = probability_detail_statistics(prob, pred, gt)
            probability.append({"condition": condition, "domain": r["domain"], "subject": r["subject"], **ps})
        print(f"geometry {i}/{len(records)}", flush=True)
    return geometry, quant, probability


def figure(r, role, path):
    im = np.asarray(nib.load(r["image_path"]).dataobj, float)
    gt = np.asarray(nib.load(r["ground_truth_path"]).dataobj) > 0
    base = np.asarray(nib.load(r["baseline_filtered_prediction_path"]).dataobj) > 0
    new = np.asarray(nib.load(r["higher_res_filtered_prediction_path"]).dataobj) > 0
    rb = np.asarray(nib.load(r["baseline_raw_prediction_path"]).dataobj) > 0
    rn = np.asarray(nib.load(r["higher_res_raw_prediction_path"]).dataobj) > 0
    pb, _ = align_probability_to_raw(np.asarray(nib.load(r["baseline_probability_path"]).dataobj, np.float32), rb)
    pn, _ = align_probability_to_raw(np.asarray(nib.load(r["higher_res_probability_path"]).dataobj, np.float32), rn)
    score = (base ^ gt).sum((0, 1)) + (new ^ gt).sum((0, 1)); z = int(score.argmax())
    q = np.argwhere(gt[:, :, z] | base[:, :, z] | new[:, :, z])
    lo = np.maximum(q.min(0) - 6, 0); hi = np.minimum(q.max(0) + 7, gt.shape[:2])
    a, b = np.percentile(im[:, :, z], [1, 99]); m = np.clip((im[:, :, z] - a) / max(b - a, 1e-8), 0, 1)

    def err(p):
        x = np.stack([m] * 3, -1); p = p[:, :, z]; x[p & ~gt[:, :, z]] = (1, 0, 0); x[~p & gt[:, :, z]] = (1, 1, 0); return x

    fig, ax = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    panels = [m, gt[:, :, z], base[:, :, z], new[:, :, z], err(base), err(new), pb[:, :, z], pn[:, :, z],
              m[lo[0]:hi[0], lo[1]:hi[1]], m[lo[0]:hi[0], lo[1]:hi[1]],
              (base ^ gt)[:, :, z][lo[0]:hi[0], lo[1]:hi[1]], (new ^ gt)[:, :, z][lo[0]:hi[0], lo[1]:hi[1]]]
    titles = ["MRI", "Expert mask", "Baseline (128x128x160)", "Higher-res (192x128x160)", "Baseline FP/FN", "Higher-res FP/FN",
              "Baseline probability", "Higher-res probability", "Baseline boundary zoom", "Higher-res boundary zoom",
              "Baseline error zoom", "Higher-res error zoom"]
    for a0, x, t in zip(ax.flat, panels, titles):
        a0.imshow(x.T if x.ndim == 2 else x.transpose(1, 0, 2), cmap="gray" if x.ndim == 2 else None, origin="lower", interpolation="nearest")
        a0.set_title(t); a0.axis("off")
    for a0, p in ((ax[2, 0], base), (ax[2, 1], new)):
        a0.contour(gt[lo[0]:hi[0], lo[1]:hi[1], z].T, colors="#24c96b", linewidths=1)
        a0.contour(p[lo[0]:hi[0], lo[1]:hi[1], z].T, colors="#00d5e7", linewidths=1)
    fig.legend(handles=[Line2D([0], [0], color="#24c96b", label="Expert contour"), Line2D([0], [0], color="#00d5e7", label="Prediction contour"),
                         Patch(color="red", label="FP"), Patch(color="yellow", label="FN")], loc="lower center", ncol=4)
    fig.suptitle(f"{r['domain']} {r['subject']} | {role} | slice {z} | Dice {float(r['baseline_filtered_dice']):.4f}→{float(r['higher_res_filtered_dice']):.4f} | spacing {nib.load(r['ground_truth_path']).header.get_zooms()[:3]} mm")
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig); return z


def main():
    rec = rows(OUT / "per_subject_comparison.csv")
    geom, quant, prob = geometry(rec)
    write(OUT / "contour_geometry_comparison.csv", geom)
    write(OUT / "quantization_comparison.csv", quant)
    write(OUT / "probability_detail_comparison.csv", prob)

    metric = [r for r in rows(OUT / "native_metrics.csv") if r["condition"] in ("baseline_filtered", "higher_res_filtered")]
    write(OUT / "baseline_vs_higher_res_metrics.csv", metric)

    paired = []
    for r in rec:
        paired.append({
            "domain": r["domain"], "subject": r["subject"],
            "baseline_dice": r["baseline_filtered_dice"], "higher_res_dice": r["higher_res_filtered_dice"],
            "dice_change": float(r["higher_res_filtered_dice"]) - float(r["baseline_filtered_dice"]),
            "baseline_hd95_mm": r["baseline_filtered_hd95_mm"], "higher_res_hd95_mm": r["higher_res_filtered_hd95_mm"],
            "baseline_assd_mm": r["baseline_filtered_assd_mm"], "higher_res_assd_mm": r["higher_res_filtered_assd_mm"],
        })
    write(OUT / "paired_subject_changes.csv", paired)

    manifest = []
    for domain in ("CAMRI", "Mouse"):
        q = [r for r in rec if r["domain"] == domain]
        g = [x for x in geom if x["domain"] == domain and x["condition"] == "baseline_128"]
        pixel = max(g, key=lambda x: x["prediction_expert_axis_run_max_ratio"])["subject"]
        roles = {
            "largest_improvement": max(q, key=lambda x: float(x["higher_res_filtered_dice"]) - float(x["baseline_filtered_dice"])),
            "largest_regression": min(q, key=lambda x: float(x["higher_res_filtered_dice"]) - float(x["baseline_filtered_dice"])),
            "median_change": sorted(q, key=lambda x: float(x["higher_res_filtered_dice"]) - float(x["baseline_filtered_dice"]))[len(q) // 2],
            "previously_pixelated": next(x for x in q if x["subject"] == pixel),
            "worst_baseline": min(q, key=lambda x: float(x["baseline_filtered_dice"])),
        }
        curved = "064" if domain == "CAMRI" else "POLYIC_20190629_polyic_mo__E4_P1_4"
        roles["high_curvature"] = next((x for x in q if x["subject"] == curved), roles["median_change"])
        for role, r in roles.items():
            p = OUT / "figures" / domain.lower() / f"{role}_{r['subject']}.png"
            z = figure(r, role, p)
            manifest.append({"domain": domain, "role": role, "subject": r["subject"], "slice": z, "path": str(p)})
    write(OUT / "figures/manifest.csv", manifest)

    h = rows(OUT / "training/history.csv")
    selected_epoch = json.loads((OUT / "training/summary.json").read_text())["selected_epoch"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([int(x["epoch"]) for x in h], [float(x["camri_validation_dice"]) for x in h], label="CAMRI validation")
    ax.plot([int(x["epoch"]) for x in h], [float(x["mouse_validation_dice"]) for x in h], label="Mouse validation")
    ax.axvline(selected_epoch, color="black", ls="--", label=f"selected epoch {selected_epoch}")
    ax.set(title="Higher-resolution (192x128x160) TRUE level0 validation history", xlabel="Epoch", ylabel="Volumetric Dice", ylim=(.9, 1))
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(OUT / "training/learning_curves.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    print(json.dumps({"records": len(rec), "figures": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
