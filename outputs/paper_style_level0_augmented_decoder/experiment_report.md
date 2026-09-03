# UNETR-style full-width level0 decoder + paper-style augmentation -- experiment report

Single isolated variable relative to `outputs/unetr_style_level0_full_width_decoder` (same architecture: `PaperStyleLevel0OneQueryMaskDecoder`, level0_width=32, matching embedding_dim -- the real four-stage skip-connected `UnetrUpBlock` chain at its full intended capacity, not the CPU-forced level0_width=8 taper): training augmentation. That run improved Mouse boundary metrics broadly (76/80 subjects, every metric) but visual inspection found small, locally-confident false-positive 'bulbs' escaping the true contour on some subjects -- consistent with a higher-capacity decoder (444,449 params) memorizing patterns from only 39 real training images rather than learning boundary features that generalize. This run replaces the prior feature-space flip+noise augmentation with the original RS2-Net paper's own richer recipe (rotation, zoom, Gaussian blur/noise, brightness/contrast, gamma, simulated low-resolution), applied to raw images before the frozen encoder (see `scripts/paper_style_augmentation.py`) -- the standard remedy for this failure mode. Same initial checkpoint (`outputs/higher_resolution_true_level0`, not the possibly-overfit full-width checkpoint, so this cleanly tests whether augmentation prevents the overfitting rather than corrects an already-overfit model), same resolution, loss, optimizer, and seed as the run being followed up on.

## 1. Architecture and initialization

- Total decoder parameters: **415,777** (level0_width=32, identical architecture to `outputs/unetr_style_level0_full_width_decoder`; vs. 182,081 for the original single-depthwise-pass decoder).
- Transferred from `outputs/higher_resolution_true_level0/checkpoints/best_higher_resolution_true_level0.pt`: **75/102** decoder tensors, by exact name+shape, all bit-identical (verified, not assumed -- see `outputs/unetr_style_level0_full_width_decoder/pretraining_verification.json`, reused unchanged since the architecture didn't change here).
- Freshly initialized (no prior-checkpoint counterpart): 21 tensors -- the four real `UnetrUpBlock` stages, `projections.level0`, and `query_updates.level0`.
- Augmentation spec (probabilities and ranges): `{"flip_probability": 0.5, "rotate_probability": 0.2, "rotate_range_deg": 15, "zoom_probability": 0.2, "zoom_range": [0.85, 1.15], "blur_probability": 0.15, "blur_sigma_range": [0.5, 1.5], "noise_probability": 0.15, "noise_std_fraction": 0.05, "contrast_probability": 0.15, "contrast_range": [0.75, 1.25], "gamma_probability": 0.15, "gamma_range": [0.7, 1.5], "low_res_probability": 0.15, "low_res_scale_range": [0.5, 1.0]}`. Spatial transforms (rotate, zoom, flip) applied identically to image and target; intensity transforms (blur, noise, contrast, gamma, simulated low-resolution) to the image only. Applied to raw images before the frozen encoder, so training could no longer use this project's usual cached-feature shortcut -- the encoder runs fresh every training step, every epoch (validation is unaffected: no augmentation there, so its cached-feature path is unchanged).

## 2. Training

- Selected epoch: **21** / 26 run (early-stop patience reached, or max_epochs exhausted).
- Best balanced validation Dice ((CAMRI+Mouse)/2): **0.9807**.
- CAMRI safety-eligibility floor: 0.9728 (reference 0.9828); ineligible epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]; safety stop triggered: False.
- Mean epoch time: 53.7s; total elapsed: 0.39 hours; peak process memory: 3179 MiB.
- See `training/learning_curves.png` for the full validation-Dice trajectory.

## 3. Native test-set metrics (86 subjects: 6 CAMRI + 80 Mouse)

| Domain | Condition | Dice | IoU | Precision | Recall | HD95 (mm) | ASSD (mm) | SurfDice@0.1mm | SurfDice@0.2mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMRI | current best (192, real depthwise level0) | 0.9888 | 0.9779 | 0.9889 | 0.9888 | 0.1333 | 0.0178 | 0.8943 | 0.9390 |
| CAMRI | UNETR-style (real skip-connected depth) | 0.9863 | 0.9730 | 0.9824 | 0.9903 | 0.1333 | 0.0292 | 0.8715 | 0.9169 |
| Mouse | current best (192, real depthwise level0) | 0.9795 | 0.9599 | 0.9783 | 0.9808 | 0.1151 | 0.0260 | 0.7986 | 0.9710 |
| Mouse | UNETR-style (real skip-connected depth) | 0.9802 | 0.9611 | 0.9794 | 0.9810 | 0.1124 | 0.0251 | 0.8091 | 0.9716 |

## 4. Paired per-subject change

**CAMRI** (6 subjects): 0 improved, 6 regressed, 0 unchanged (Dice). Mean Dice change: -0.00251. Mean HD95 change: +0.0000 mm.
**Mouse** (80 subjects): 59 improved, 21 regressed, 0 unchanged (Dice). Mean Dice change: +0.00063. Mean HD95 change: -0.0027 mm.

## 5. Per-axis grid-lock excess (contour quantization signature)

Same methodology as `outputs/level0_depth_diagnostic/` and the prior resolution experiment -- `prediction_minus_expert_alignment`, averaged over candidate factors 2-8, current best vs. UNETR-style, per domain and native axis:

| Domain | Axis | current_best | unetr_style |
|---|---|---:|---:|
| CAMRI | x | -0.0115 | -0.0137 |
| CAMRI | y | +0.0153 | +0.0123 |
| Mouse | x | +0.0015 | +0.0012 |
| Mouse | y | +0.0360 | +0.0343 |

## 6. Figures

Per domain: `worst_dice`, `median_dice`, `best_dice` (ranked on the new model's own Dice -- the explicit request for this run), plus the established paired-comparison roles (`largest_improvement`, `largest_regression`, `largest_hd95_improvement`, `previously_pixelated`, `high_curvature`) kept for continuity with every prior report in this family. 16 figures total; see `figures/manifest.csv` for the full path list.

## 7. Verdict

**Note on what this verdict compares:** like every prior report in this family, the numbers above are against `outputs/higher_resolution_true_level0` (the original current-best model), for direct comparability across all three UNETR-style runs. The more directly relevant comparison for *this specific experiment's own question* -- did augmentation reduce the false-positive 'bulbs' seen in the un-augmented full-width run -- is against `outputs/unetr_style_level0_full_width_decoder` (that run's own baseline-relative numbers were: Mouse Dice +0.00128, HD95 -0.00138mm, 76/80 improved; CAMRI Dice -0.00044, HD95 +0.01381mm, 2/6 improved). See its own `paired_subject_changes.csv` and figures for a like-for-like check; this report does not re-derive that comparison to avoid silently changing what earlier reports in this family measured.

**B. Full-width decoder depth with paper-style augmentation gives a small, mixed, or inconclusive change relative to the current best model -- direction is not clearly positive across both domains/metrics.**

Mean Dice change: +0.00041. Mean HD95 change: -0.0025 mm. Subjects with a Dice regression > 0.0005: 12/86.

This experiment tested a single, literature-grounded hypothesis -- that the full-width decoder's small, locally-confident false-positive boundary excursions are an overfitting signature (more decoder capacity, same 39 training images, no added regularization), remedied by the original RS2-Net paper's own richer training augmentation rather than by architecture changes. See `outputs/unetr_style_level0_full_width_decoder/` for the run and visual evidence that motivated this experiment, and `scripts/paper_style_augmentation.py` / `configs/paper_style_level0_augmented.yaml` for exactly what augmentation was applied.
