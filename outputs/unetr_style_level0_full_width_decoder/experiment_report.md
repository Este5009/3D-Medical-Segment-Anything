# UNETR-style real-decoder-depth level0 decoder -- experiment report

Single isolated architectural variable relative to `outputs/higher_resolution_true_level0` (192x128x160, current best model): `TrueFullResolutionLevel0OneQueryMaskDecoder`'s single depthwise+pointwise level0 fusion pass is replaced by `UnetrStyleLevel0OneQueryMaskDecoder`'s real four-stage skip-connected residual upsampling chain, level4->3->2->1->0, using `monai.networks.blocks.UnetrUpBlock` unmodified -- the exact block class the original RS2-Net decoder itself is built from. Motivated by `outputs/level0_depth_diagnostic/` (residual error is small, scattered 1-4-voxel clusters, not axis-locked quantization on the untouched native-X axis -- a decoder local-context signature, not a resolution shortfall). Resolution, spacing, tile size, encoder, hyperparameters, loss, and augmentation are all unchanged from the initialization checkpoint's own training run.

## 1. Architecture and initialization

- Total decoder parameters: **444,449** (vs. 182,081 for the current-best decoder; 355,889 measured directly, level0_width=32).
- Transferred from `outputs/higher_resolution_true_level0/checkpoints/best_higher_resolution_true_level0.pt`: **75/102** decoder tensors, by exact name+shape, all bit-identical (verified, not assumed -- see `pretraining_verification.json`).
- Freshly initialized (no prior-checkpoint counterpart): 21 tensors -- the four real `UnetrUpBlock` stages, `projections.level0`, and `query_updates.level0`.
- `level0_width=8` (not this project's prior `level0_width=16` precedent) was set from direct on-machine profiling before committing to training: a single dense `UnetrUpBlock(in=32,out=W)` forward+backward pass at the level0 grid measured 3.05s/4.51s/8.99s/13.89s at W=6/8/10/12, and did not complete a single forward pass in over 8 minutes at W=32; `level0_width=16` (fine for a single depthwise+pointwise pass) measured 206.71s for one full training step of this real-residual-block design -- infeasible for an overnight run. `level0_width=8` measured 7.59-10.62s/step in scoping and {'CAMRI': 0.16835546493530273, 'Mouse': 0.002568960189819336} seconds/sample forward-only against the real checkpoint and cached features -- tractable.

## 2. Training

- Selected epoch: **34** / 35 run (early-stop patience reached, or max_epochs exhausted).
- Best balanced validation Dice ((CAMRI+Mouse)/2): **0.9804**.
- CAMRI safety-eligibility floor: 0.9728 (reference 0.9828); ineligible epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]; safety stop triggered: False.
- Mean epoch time: 61.7s; total elapsed: 0.60 hours; peak process memory: 3023 MiB.
- See `training/learning_curves.png` for the full validation-Dice trajectory.

## 3. Native test-set metrics (86 subjects: 6 CAMRI + 80 Mouse)

| Domain | Condition | Dice | IoU | Precision | Recall | HD95 (mm) | ASSD (mm) | SurfDice@0.1mm | SurfDice@0.2mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMRI | current best (192, real depthwise level0) | 0.9888 | 0.9779 | 0.9889 | 0.9888 | 0.1333 | 0.0178 | 0.8943 | 0.9390 |
| CAMRI | UNETR-style (real skip-connected depth) | 0.9884 | 0.9771 | 0.9875 | 0.9893 | 0.1471 | 0.0381 | 0.8925 | 0.9379 |
| Mouse | current best (192, real depthwise level0) | 0.9795 | 0.9599 | 0.9783 | 0.9808 | 0.1151 | 0.0260 | 0.7986 | 0.9710 |
| Mouse | UNETR-style (real skip-connected depth) | 0.9808 | 0.9624 | 0.9792 | 0.9824 | 0.1137 | 0.0244 | 0.8149 | 0.9741 |

## 4. Paired per-subject change

**CAMRI** (6 subjects): 2 improved, 4 regressed, 0 unchanged (Dice). Mean Dice change: -0.00044. Mean HD95 change: +0.0138 mm.
**Mouse** (80 subjects): 76 improved, 4 regressed, 0 unchanged (Dice). Mean Dice change: +0.00128. Mean HD95 change: -0.0014 mm.

## 5. Per-axis grid-lock excess (contour quantization signature)

Same methodology as `outputs/level0_depth_diagnostic/` and the prior resolution experiment -- `prediction_minus_expert_alignment`, averaged over candidate factors 2-8, current best vs. UNETR-style, per domain and native axis:

| Domain | Axis | current_best | unetr_style |
|---|---|---:|---:|
| CAMRI | x | -0.0115 | -0.0131 |
| CAMRI | y | +0.0153 | +0.0127 |
| Mouse | x | +0.0015 | +0.0014 |
| Mouse | y | +0.0360 | +0.0345 |

## 6. Figures

Per domain: `worst_dice`, `median_dice`, `best_dice` (ranked on the new model's own Dice -- the explicit request for this run), plus the established paired-comparison roles (`largest_improvement`, `largest_regression`, `largest_hd95_improvement`, `previously_pixelated`, `high_curvature`) kept for continuity with every prior report in this family. 16 figures total; see `figures/manifest.csv` for the full path list.

## 7. Verdict

**B. Real decoder depth gives a small, mixed, or inconclusive change relative to the current best model -- direction is not clearly positive across both domains/metrics.**

Mean Dice change: +0.00116. Mean HD95 change: -0.0003 mm. Subjects with a Dice regression > 0.0005: 4/86.

This experiment tested a single, literature-grounded hypothesis -- the residual pixelation is a decoder local-context shortfall, addressed by giving level0 the same real, skip-connected, multi-stage residual depth the original paper's own decoder has, rather than a single lightweight conv pass. See `outputs/level0_depth_diagnostic/findings.md` for the evidence that motivated this experiment, and `configs/unetr_style_level0_full_width.yaml` / this decoder's docstring for the on-machine timing measurements that set `level0_width=8`.
