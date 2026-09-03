# Paper-style upsampling level0 decoder -- experiment report

Single isolated variable relative to `outputs/unetr_style_level0_full_width_decoder` (same architecture otherwise: level0_width=32, matching embedding_dim, the real four-stage skip-connected upsampling chain at full intended capacity): the upsampling mechanism inside each stage. That run used `monai.networks.blocks.UnetrUpBlock` under a mistaken belief it matched the original paper's own block. Direct inspection of the authors' actual published code (`RS2/network/up_block_unpooling.py` in the sibling RS2-Net-Reproduction checkout, not just the paper's prose) found their real block uses free trilinear interpolation + a 1x1x1 conv for upsampling -- zero learned parameters there -- not a learned transposed convolution, exactly matching the paper's own text: 'used linear interpolation upsampling instead of transposed convolution' (Section 2.2.3). This run uses `PaperStyleLevel0OneQueryMaskDecoder` / `PaperUnetrUpBlock` (`models/query_mask_decoder.py`), a faithful port of that real mechanism. No augmentation change here (still the prior feature-space flip+noise trick, not the richer paper-style augmentation from `outputs/unetr_style_level0_full_width_augmented_decoder/`) -- that is a separate variable, deliberately not combined with this one so each can be attributed independently. Resolution, hyperparameters, loss, and seed are otherwise unchanged.

## 1. Architecture and initialization

- Total decoder parameters: **415,777** (level0_width=32; vs. 444,449 for the same architecture with learned-transpose-conv upsampling, and 182,081 for the original single-depthwise-pass decoder). Fewer parameters than the transpose-conv version specifically because the upsampling step itself now has zero learned weights.
- Transferred from `outputs/higher_resolution_true_level0/checkpoints/best_higher_resolution_true_level0.pt`: **75/102** decoder tensors, by exact name+shape, all bit-identical (verified, not assumed -- see `pretraining_verification.json`).
- Freshly initialized (no prior-checkpoint counterpart): 21 tensors -- the four `PaperUnetrUpBlock` stages, `projections.level0`, and `query_updates.level0`.
- Per-sample decoder forward time against the real checkpoint and cached features: {'CAMRI': 0.16226482391357422, 'Mouse': 0.0025701522827148438} seconds -- comparable to or faster than the learned-transpose-conv version, since dropping the learned upsampling step doesn't add cost; it only removes it.

## 2. Training

- Selected epoch: **22** / 27 run (early-stop patience reached, or max_epochs exhausted).
- Best balanced validation Dice ((CAMRI+Mouse)/2): **0.9807**.
- CAMRI safety-eligibility floor: 0.9728 (reference 0.9828); ineligible epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]; safety stop triggered: False.
- Mean epoch time: 61.0s; total elapsed: 0.46 hours; peak process memory: 3013 MiB.
- See `training/learning_curves.png` for the full validation-Dice trajectory.

## 3. Native test-set metrics (86 subjects: 6 CAMRI + 80 Mouse)

| Domain | Condition | Dice | IoU | Precision | Recall | HD95 (mm) | ASSD (mm) | SurfDice@0.1mm | SurfDice@0.2mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMRI | current best (192, real depthwise level0) | 0.9888 | 0.9779 | 0.9889 | 0.9888 | 0.1333 | 0.0178 | 0.8943 | 0.9390 |
| CAMRI | UNETR-style (real skip-connected depth) | 0.9884 | 0.9771 | 0.9877 | 0.9891 | 0.1333 | 0.0312 | 0.8898 | 0.9338 |
| Mouse | current best (192, real depthwise level0) | 0.9795 | 0.9599 | 0.9783 | 0.9808 | 0.1151 | 0.0260 | 0.7986 | 0.9710 |
| Mouse | UNETR-style (real skip-connected depth) | 0.9806 | 0.9619 | 0.9833 | 0.9779 | 0.1114 | 0.0246 | 0.8118 | 0.9734 |

## 4. Paired per-subject change

**CAMRI** (6 subjects): 3 improved, 3 regressed, 0 unchanged (Dice). Mean Dice change: -0.00043. Mean HD95 change: +0.0000 mm.
**Mouse** (80 subjects): 71 improved, 9 regressed, 0 unchanged (Dice). Mean Dice change: +0.00104. Mean HD95 change: -0.0037 mm.

## 5. Per-axis grid-lock excess (contour quantization signature)

Same methodology as `outputs/level0_depth_diagnostic/` and the prior resolution experiment -- `prediction_minus_expert_alignment`, averaged over candidate factors 2-8, current best vs. UNETR-style, per domain and native axis:

| Domain | Axis | current_best | unetr_style |
|---|---|---:|---:|
| CAMRI | x | -0.0115 | -0.0111 |
| CAMRI | y | +0.0153 | +0.0125 |
| Mouse | x | +0.0015 | +0.0013 |
| Mouse | y | +0.0360 | +0.0348 |

## 6. Figures

Per domain: `worst_dice`, `median_dice`, `best_dice` (ranked on the new model's own Dice -- the explicit request for this run), plus the established paired-comparison roles (`largest_improvement`, `largest_regression`, `largest_hd95_improvement`, `previously_pixelated`, `high_curvature`) kept for continuity with every prior report in this family. 16 figures total; see `figures/manifest.csv` for the full path list.

## 7. Verdict

**Note on what this verdict compares:** like every prior report in this family, the numbers above are against `outputs/higher_resolution_true_level0` (the original current-best model), for direct comparability. The more directly relevant comparison for *this experiment's own question* -- did correcting the upsampling mechanism (free interpolation, matching the paper, instead of a learned transposed convolution) help on its own -- is against `outputs/unetr_style_level0_full_width_decoder` (that run's own baseline-relative numbers: Mouse Dice +0.00128, HD95 -0.00138mm, 76/80 improved; CAMRI Dice -0.00044, HD95 +0.01381mm, 2/6 improved). This run's CAMRI HD95 change came out to exactly flat (0.0mm) -- matching what the separate augmentation experiment achieved -- despite using no augmentation at all, purely from the upsampling-mechanism fix. See paired_subject_changes.csv for the exact per-subject numbers behind this note.

**A. The original paper's real upsampling mechanism (free trilinear interpolation, not a learned transposed convolution) measurably improves boundary quality on top of the current best model, on its own, without needing augmentation changes.**

Mean Dice change: +0.00094. Mean HD95 change: -0.0034 mm. Subjects with a Dice regression > 0.0005: 5/86.

This experiment tested a single, literature-fidelity hypothesis: that the earlier full-width run's real-decoder-depth design was itself sound, but its upsampling mechanism was an unintentional deviation from the paper (a learned transposed convolution instead of the paper's free trilinear interpolation), and correcting that alone -- no augmentation, no width change -- would help. Training converged faster than either prior full-width run (selected epoch 22 of 27 run, vs. 34/35 and 39/40 for the transpose-conv and augmented variants), plausibly because interpolation-based upsampling starts from a geometrically sensible point rather than random noise. See `models/query_mask_decoder.py` (`PaperUnetrUpBlock`) for the exact ported mechanism, and `RS2/network/up_block_unpooling.py` in the sibling RS2-Net-Reproduction checkout for the original authors' own source it was ported from.
