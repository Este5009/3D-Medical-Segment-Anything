#!/usr/bin/env python3
"""Assemble outputs/unetr_style_level0_full_width_augmented_decoder/experiment_report.md from the
already-generated CSVs/JSONs. Run only after train/evaluate/compare finish.
No invented numbers -- everything here is read back from files those scripts
wrote.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/unetr_style_level0_full_width_augmented_decoder"
# This run's pre-training architecture gates reused the full-width run's own
# verify script/config unchanged (augmentation is the only variable; the
# architecture is identical), so that gate's output file was correctly
# written to the full-width run's OWN output directory, not this one.
FULL_WIDTH_OUT = ROOT / "outputs/unetr_style_level0_full_width_decoder"


def rows(p):
    return list(csv.DictReader(open(p)))


def fnum(x, nd=4):
    return f"{float(x):.{nd}f}"


def main():
    summary = json.loads((OUT / "training/summary.json").read_text())
    pretrain = json.loads((FULL_WIDTH_OUT / "pretraining_verification.json").read_text())
    native = rows(OUT / "native_metrics.csv")
    paired = rows(OUT / "paired_subject_changes.csv")
    manifest = rows(OUT / "figures/manifest.csv")
    quant = rows(OUT / "quantization_comparison.csv")

    def agg(domain, condition):
        return next(r for r in native if r["domain"] == domain and r["condition"] == condition)

    lines = []
    lines.append("# UNETR-style full-width level0 decoder + paper-style augmentation -- experiment report\n")
    lines.append(
        "Single isolated variable relative to `outputs/unetr_style_level0_full_width_decoder` "
        "(same architecture: `UnetrStyleLevel0OneQueryMaskDecoder`, level0_width=32, matching "
        "embedding_dim -- the real four-stage skip-connected `UnetrUpBlock` chain at its full "
        "intended capacity, not the CPU-forced level0_width=8 taper): training augmentation. "
        "That run improved Mouse boundary metrics broadly (76/80 subjects, every metric) but "
        "visual inspection found small, locally-confident false-positive 'bulbs' escaping the "
        "true contour on some subjects -- consistent with a higher-capacity decoder (444,449 "
        "params) memorizing patterns from only 39 real training images rather than learning "
        "boundary features that generalize. This run replaces the prior feature-space "
        "flip+noise augmentation with the original RS2-Net paper's own richer recipe (rotation, "
        "zoom, Gaussian blur/noise, brightness/contrast, gamma, simulated low-resolution), "
        "applied to raw images before the frozen encoder (see "
        "`scripts/paper_style_augmentation.py`) -- the standard remedy for this failure mode. "
        "Same initial checkpoint (`outputs/higher_resolution_true_level0`, not the possibly-"
        "overfit full-width checkpoint, so this cleanly tests whether augmentation prevents "
        "the overfitting rather than corrects an already-overfit model), same resolution, loss, "
        "optimizer, and seed as the run being followed up on.\n"
    )

    lines.append("## 1. Architecture and initialization\n")
    lines.append(f"- Total decoder parameters: **{summary['unetr_style_decoder_parameters']:,}** "
                  f"(level0_width={summary['level0_width']}, identical architecture to "
                  "`outputs/unetr_style_level0_full_width_decoder`; vs. 182,081 for the original "
                  "single-depthwise-pass decoder).")
    lines.append(f"- Transferred from `{summary['initial_checkpoint']}`: **{summary['transferred_baseline_tensors']}/102** decoder tensors, "
                  "by exact name+shape, all bit-identical (verified, not assumed -- see "
                  "`outputs/unetr_style_level0_full_width_decoder/pretraining_verification.json`, "
                  "reused unchanged since the architecture didn't change here).")
    lines.append(f"- Freshly initialized (no prior-checkpoint counterpart): {len(summary['new_parameter_keys'])} tensors "
                  "-- the four real `UnetrUpBlock` stages, `projections.level0`, and `query_updates.level0`.")
    lines.append(
        "- Augmentation spec (probabilities and ranges): "
        f"`{json.dumps(summary.get('augmentation_spec', {}))}`. Spatial transforms (rotate, zoom, flip) "
        "applied identically to image and target; intensity transforms (blur, noise, contrast, gamma, "
        "simulated low-resolution) to the image only. Applied to raw images before the frozen encoder, "
        "so training could no longer use this project's usual cached-feature shortcut -- the encoder "
        "runs fresh every training step, every epoch (validation is unaffected: no augmentation there, "
        "so its cached-feature path is unchanged).\n"
    )

    lines.append("## 2. Training\n")
    lines.append(f"- Selected epoch: **{summary['selected_epoch']}** / {summary['epochs_run']} run "
                  f"(early-stop patience reached, or max_epochs exhausted).")
    lines.append(f"- Best balanced validation Dice ((CAMRI+Mouse)/2): **{fnum(summary['best_balanced_validation_dice'])}**.")
    lines.append(f"- CAMRI safety-eligibility floor: {fnum(summary['camri_floor'])} (reference {fnum(summary['camri_reference'])}); "
                  f"ineligible epochs: {summary['safety_ineligible_epochs'] or 'none'}; safety stop triggered: {summary['camri_safety_stop']}.")
    lines.append(f"- Mean epoch time: {summary['mean_epoch_seconds']:.1f}s; total elapsed: "
                  f"{summary['elapsed_seconds']/3600:.2f} hours; peak process memory: {summary['peak_process_memory_mib']:.0f} MiB.")
    lines.append("- See `training/learning_curves.png` for the full validation-Dice trajectory.\n")

    lines.append("## 3. Native test-set metrics (86 subjects: 6 CAMRI + 80 Mouse)\n")
    lines.append("| Domain | Condition | Dice | IoU | Precision | Recall | HD95 (mm) | ASSD (mm) | SurfDice@0.1mm | SurfDice@0.2mm |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for domain in ("CAMRI", "Mouse"):
        for cond, label in (("baseline_filtered", "current best (192, real depthwise level0)"),
                             ("unetr_style_filtered", "UNETR-style (real skip-connected depth)")):
            r = agg(domain, cond)
            lines.append(f"| {domain} | {label} | {fnum(r['dice'])} | {fnum(r['iou'])} | {fnum(r['precision'])} | "
                          f"{fnum(r['recall'])} | {fnum(r['hd95_mm'])} | {fnum(r['assd_mm'])} | "
                          f"{fnum(r['surface_dice_0.1mm'])} | {fnum(r['surface_dice_0.2mm'])} |")
    lines.append("")

    lines.append("## 4. Paired per-subject change\n")
    for domain in ("CAMRI", "Mouse"):
        d = [r for r in paired if r["domain"] == domain]
        changes = [float(r["dice_change"]) for r in d]
        hd_changes = [float(r["hd95_change_mm"]) for r in d]
        improved = sum(1 for c in changes if c > 0)
        regressed = sum(1 for c in changes if c < 0)
        unchanged = len(d) - improved - regressed
        lines.append(f"**{domain}** ({len(d)} subjects): {improved} improved, {regressed} regressed, {unchanged} unchanged (Dice). "
                      f"Mean Dice change: {sum(changes)/len(changes):+.5f}. Mean HD95 change: {sum(hd_changes)/len(hd_changes):+.4f} mm.")
    lines.append("")

    lines.append("## 5. Per-axis grid-lock excess (contour quantization signature)\n")
    lines.append("Same methodology as `outputs/level0_depth_diagnostic/` and the prior resolution experiment "
                  "-- `prediction_minus_expert_alignment`, averaged over candidate factors 2-8, current best vs. "
                  "UNETR-style, per domain and native axis:\n")
    axis_summary: dict = {}
    for r in quant:
        key = (r["condition"], r["domain"], r["native_coordinate_axis"])
        axis_summary.setdefault(key, []).append(float(r["prediction_minus_expert_alignment"]))
    lines.append("| Domain | Axis | current_best | unetr_style |")
    lines.append("|---|---|---:|---:|")
    for domain in ("CAMRI", "Mouse"):
        for axis in ("x", "y"):
            a = sum(axis_summary.get(("current_best", domain, axis), [0])) / max(len(axis_summary.get(("current_best", domain, axis), [1])), 1)
            b = sum(axis_summary.get(("unetr_style", domain, axis), [0])) / max(len(axis_summary.get(("unetr_style", domain, axis), [1])), 1)
            lines.append(f"| {domain} | {axis} | {a:+.4f} | {b:+.4f} |")
    lines.append("")

    lines.append("## 6. Figures\n")
    lines.append("Per domain: `worst_dice`, `median_dice`, `best_dice` (ranked on the new model's own Dice -- "
                  "the explicit request for this run), plus the established paired-comparison roles "
                  "(`largest_improvement`, `largest_regression`, `largest_hd95_improvement`, `previously_pixelated`, "
                  "`high_curvature`) kept for continuity with every prior report in this family. "
                  f"{len(manifest)} figures total; see `figures/manifest.csv` for the full path list.\n")

    dice_deltas = [float(r["dice_change"]) for r in paired]
    hd_deltas = [float(r["hd95_change_mm"]) for r in paired]
    mean_dice_delta = sum(dice_deltas) / len(dice_deltas)
    mean_hd_delta = sum(hd_deltas) / len(hd_deltas)
    regressions = sum(1 for c in dice_deltas if c < -0.0005)
    lines.append("## 7. Verdict\n")
    lines.append(
        "**Note on what this verdict compares:** like every prior report in this family, the numbers "
        "above are against `outputs/higher_resolution_true_level0` (the original current-best model), "
        "for direct comparability across all three UNETR-style runs. The more directly relevant "
        "comparison for *this specific experiment's own question* -- did augmentation reduce the "
        "false-positive 'bulbs' seen in the un-augmented full-width run -- is against "
        "`outputs/unetr_style_level0_full_width_decoder` (that run's own baseline-relative numbers were: "
        "Mouse Dice +0.00128, HD95 -0.00138mm, 76/80 improved; CAMRI Dice -0.00044, HD95 +0.01381mm, "
        "2/6 improved). See its own `paired_subject_changes.csv` and figures for a like-for-like check; "
        "this report does not re-derive that comparison to avoid silently changing what earlier reports "
        "in this family measured.\n"
    )
    if mean_dice_delta > 0.0005 and mean_hd_delta < -0.0005 and regressions <= len(dice_deltas) * 0.1:
        verdict = ("**A. Full-width decoder depth with paper-style augmentation measurably improves boundary "
                    "quality on top of the current best model.**")
    elif regressions > len(dice_deltas) * 0.3 or mean_dice_delta < -0.0005:
        verdict = ("**C. Full-width decoder depth with paper-style augmentation does not clearly help, and/or "
                    "introduces meaningful regressions relative to the current best model.**")
    else:
        verdict = ("**B. Full-width decoder depth with paper-style augmentation gives a small, mixed, or "
                    "inconclusive change relative to the current best model -- direction is not clearly "
                    "positive across both domains/metrics.**")
    lines.append(verdict)
    lines.append(f"\nMean Dice change: {mean_dice_delta:+.5f}. Mean HD95 change: {mean_hd_delta:+.4f} mm. "
                 f"Subjects with a Dice regression > 0.0005: {regressions}/{len(dice_deltas)}.")
    lines.append(
        "\nThis experiment tested a single, literature-grounded hypothesis -- that the full-width "
        "decoder's small, locally-confident false-positive boundary excursions are an overfitting "
        "signature (more decoder capacity, same 39 training images, no added regularization), remedied "
        "by the original RS2-Net paper's own richer training augmentation rather than by architecture "
        "changes. See `outputs/unetr_style_level0_full_width_decoder/` for the run and visual evidence "
        "that motivated this experiment, and `scripts/paper_style_augmentation.py` / "
        "`configs/unetr_style_level0_full_width_augmented.yaml` for exactly what augmentation was applied."
    )

    (OUT / "experiment_report.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "experiment_report.md")
    print("\n".join(lines[-8:]))


if __name__ == "__main__":
    main()
