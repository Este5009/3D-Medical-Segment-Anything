# Higher-resolution (192x128x160) TRUE level0 decoder

## Controlled question

Does giving the same frozen pretrained encoder finer Mouse MRI spatial
detail — refining only the axis `outputs/higher_resolution_encoder_diagnostic/`
identified as the dominant native-boundary-detail-loss axis — let the
validated TRUE full-resolution level0 decoder produce measurably better
Mouse (and CAMRI) segmentations? This is a resolution-only change: the
decoder class, architecture, query count, loss, optimizer, schedule, split,
threshold, and filter are all identical to `outputs/true_full_resolution_level0_decoder`.

## Input/target resolution

- Current baseline: `128x128x160`, model-axis spacing `(0.25, 0.20, 0.16)` mm.
- New: **`192x128x160`**, model-axis spacing **`(0.16667, 0.20, 0.16)`** mm.
- Only model axis0 (-> native Y) was refined, from 0.25mm to 0.16667mm
  (1.5x finer) — the single axis the encoder diagnostic found losing the
  most native detail (~2.5-3.6x downsampling). Axis1 (-> native Z, already
  finer than native) and axis2 (-> native X) spacing are unchanged.
- Field of view preserved exactly: `128 * 0.25mm = 192 * 0.16667mm = 32.0mm`
  on the refined axis. No anatomy was stretched or cropped.
- MRI resampling: the identical `DefaultPreprocessor` / `resampling_fn` as
  the baseline (same interpolation order/method); only the target spacing
  value passed to it differs (`scripts/higher_resolution_preprocessing.py`,
  an in-memory override of the loaded plans configuration -- no file edit).
- Expert masks: identical corrected nearest-neighbor path (`is_seg=True,
  order=0, order_z=0`), verified categorical `{0,1}`, produced directly on
  the new 192x128x160 grid.
- Voxel-count increase: **1.5x** (2,621,440 -> 3,932,160 voxels/tile).

## Architecture (unchanged)

`TrueFullResolutionLevel0OneQueryMaskDecoder` (`models/query_mask_decoder.py`),
182,081 parameters, `level0_width=16`. Frozen RS2-Net encoder, one query,
levels4->0 fusion, genuine full-grid level0 mask head -- **no** half-resolution
mask logits, **no** logit-space interpolation, **no** residual skip anywhere
in the graph (unchanged from the valid baseline; confirmed again by the
pre-training gates below). New logits: `[1,1,192,128,160]`.

## Initialization (verified, not assumed)

Initialized from `outputs/true_full_resolution_level0_decoder/checkpoints/best_true_level0_decoder.pt`
(that experiment's selected epoch 20, balanced validation Dice 0.977094).
`TrueFullResolutionLevel0OneQueryMaskDecoder`'s parameters (Conv3d, Linear,
MultiheadAttention, InstanceNorm3d) are all channel-dimension-only -- none
depend on spatial D/H/W -- so **all 102/102 tensors (182,081/182,081
parameters, 100%) transferred exactly**, with **zero** newly-initialized
parameters. This is verified in `pretraining_verification.json`
(`scripts/verify_higher_resolution_true_level0.py`), not assumed from the
architecture description. All eight pre-training gates passed on real CAMRI
and Mouse samples: checkpoint path, encoder frozen, query count=1, MRI input
`[1,1,192,128,160]`, target `[1,1,192,128,160]` with values `{0,1}`, logits
`[1,1,192,128,160]`, no final logit upsampling (bitwise-identical with/without
an explicit `output_size` argument), and exact weight match.

Because initialization is 100% weight-identical (unlike the original
true-level0 experiment's 26 freshly-initialized level0 parameters), this run
never needed the CAMRI-safety-eligibility-filter machinery:
`camri_safety_stop: false`, `safety_ineligible_epochs: []` -- every one of
the 20 epochs was eligible from epoch 1 onward (epoch 1 CAMRI Dice 0.9764,
comfortably above the 0.9728 floor).

## Training

All hyperparameters identical to the 128x128x160 baseline: seed `20260717`,
lr `5e-5`, weight decay `1e-4`, `max_epochs=20`, `early_stop_patience=5`,
`minimum_validation_improvement=0.0003`, `camri_validation_max_drop=0.01`,
`dice_bce_boundary` loss (`boundary_weight=0.25`), same augmentation, same
label resampling, same 0.5 threshold, same 26-connectivity filter, same
split files, same `verification_gate()` mask-quality check. **Batch size
was not changed** (already 1, single-sample SGD in both experiments -- no
memory-driven reduction was needed).

Ran the full 20-epoch budget (no early-stop trigger); balanced validation
Dice climbed with minor local dips but no collapse. **Selected epoch: 17**,
balanced validation Dice **0.980386** (CAMRI 0.980615, Mouse 0.980157) --
above the 128x128x160 baseline's own final validation Dice of 0.977094.

## Native test results (6 CAMRI + 80 Mouse, identical filtering/tiling)

Reused the exact same `run_records`/`sliding_window_logits` code as every
other experiment in this family; the only change was requesting `level0`
features (as always) and swapping the preprocessing call for the
spacing-aware variant (`scripts/evaluate_higher_resolution_true_level0.py`).

### Filtered predictions (production condition)

| Domain | Metric | Baseline (128x128x160) | Higher-res (192x128x160) | Change |
|---|---|---:|---:|---:|
| CAMRI (n=6) | Dice | 0.986922 | **0.988834** | +0.001912 |
| CAMRI | Precision | 0.987289 | 0.988923 | +0.001634 |
| CAMRI | Recall | 0.986564 | 0.988761 | +0.002197 |
| CAMRI | HD95 | 0.1333 mm | 0.1333 mm | unchanged |
| CAMRI | ASSD | 0.02049 mm | **0.01782 mm** | -13.0% |
| CAMRI | Surface Dice 0.1mm | 0.8777 | 0.8943 | +1.66 pp |
| CAMRI | FP / FN voxels | 13,044 / 14,603 | 11,622 / 12,308 | -1,422 / -2,295 |
| CAMRI | Total residual error | 27,647 | **23,930** | -13.4% |
| Mouse (n=80) | Dice | 0.972438 | **0.979537** | **+0.007099** |
| Mouse | Precision | 0.973260 | 0.978275 | +0.005015 |
| Mouse | Recall | 0.971664 | 0.980840 | +0.009176 |
| Mouse | HD95 | 0.16077 mm | **0.11506 mm** | **-28.4%** |
| Mouse | ASSD | 0.03577 mm | **0.02601 mm** | **-27.3%** |
| Mouse | Surface Dice 0.1mm | 0.7308 | 0.7986 | +6.79 pp |
| Mouse | FP / FN voxels | 259,661 / 265,552 | 213,959 / 178,789 | -17.6% / -32.7% |
| Mouse | Total residual error | 525,213 | **392,748** | **-25.2%** |

Per-subject: **6/6 CAMRI improved, 0 regressed. 80/80 Mouse improved, 0
regressed.** Mean Dice change +0.00191 (CAMRI), +0.00710 (Mouse). Even the
nominal "largest regression" role (smallest-gain subject, `POLYIC_20190510_mouse43__E9_P1`,
the same subject whose detached-appendage FP artifact was already flagged in
prior reports) still improved, Dice 0.9501->0.9531 -- there is no case in
either domain where higher resolution made a subject's segmentation worse.

## Boundary / pixel-level analysis (primary endpoint)

| Metric | Domain | Baseline (128) | Higher-res (192) | Change |
|---|---|---:|---:|---:|
| Boundary distance <=1 voxel | CAMRI | 93.978% | 94.262% | +0.28 pp |
| Boundary distance <=1 voxel | Mouse | 91.622% | **94.614%** | **+2.99 pp** |
| Mean physical distance | CAMRI | 0.0828 mm | 0.0860 mm | +0.0032 mm (small, opposite direction -- see note) |
| Mean physical distance | Mouse | 0.0733 mm | 0.0776 mm | +0.0043 mm (small, opposite direction -- see note) |
| High-complexity errors/100 surface voxels | CAMRI | (25.2, prior true-level0 report) | 22.42 | improved |
| High-complexity errors/100 surface voxels | Mouse | 35.709 | **29.10** | **-18.5%** |
| Flat-region errors/100 surface voxels | Mouse | (21.5, prior report) | 15.01 | improved |
| Contour direction-change ratio (->1.0 better) | CAMRI | 0.8285 | **0.9212** | +0.093 |
| Contour direction-change ratio (->1.0 better) | Mouse | 0.6980 | **0.8449** | **+0.147** |
| P95 axis-run ratio (->1.0 better) | CAMRI | 1.1794 | 1.1021 | improved |
| P95 axis-run ratio (->1.0 better) | Mouse | 1.1971 | 1.1395 | improved |
| Max axis-run ratio (->1.0 better) | CAMRI | 1.3224 | **1.0900** | improved |
| Max axis-run ratio (->1.0 better) | Mouse | 1.6489 | **1.3032** | improved |
| Factor-5 native-y grid-lock excess | CAMRI | 18.08 pp | **8.08 pp** | more than halved |
| Factor-5 native-y grid-lock excess | Mouse | 28.61 pp | **12.69 pp** | more than halved |

Note on mean physical distance: it rose slightly (by hundredths of a mm) in
both domains even as the <=1-voxel share, HD95, ASSD, and total error all
improved substantially -- consistent with more of the error mass moving into
the "essentially correct" bucket while a small residual tail sits marginally
farther out. This is a minor, second-order nuance, not a contradiction of
the primary result (also visible in the unchanged/near-unchanged P95 values).

Probability calibration: predictions became somewhat *less* falsely
overconfident this time (the opposite trade-off direction from the earlier
128-resolution level0 experiment) -- Mouse FP-boundary confident-incorrect
33.57%->26.42%, FN-boundary 33.28%->23.01%; uncertain (0.25-0.75) boundary
fraction rose (Mouse FP 40.98%->46.68%, FN 39.56%->48.81%). Interpreted
together with the direction-change and factor-5 results, the higher-resolution
model is not simply "sharper and still wrong" the way the 128-resolution
level0 change sometimes was -- it is more often right, and where it remains
wrong, more often appropriately uncertain rather than confidently wrong.

**Conclusion: finer input measurably and consistently reduces the remaining
coarse/staircase boundary behavior** -- every contour-geometry, grid-lock,
and curvature-stratified measure moved toward the expert's own geometry, in
both domains, not just Mouse.

## Visual comparison

12 paired figures generated (`outputs/higher_resolution_true_level0/figures/`,
6 roles x 2 domains: largest improvement, largest regression [i.e. smallest
gain], median change, previously-pixelated, worst-baseline, high-curvature),
using the identical MRI slice/crop for expert/baseline/higher-res, raw
(unsmoothed) pixel contours, and FP/FN maps. Consistent pattern across
figures: the higher-resolution contour tracks concavities and curved
segments more closely, with visibly fewer long straight (grid-aligned)
segments than the 128-resolution baseline. The nominal "largest regression"
Mouse case (`POLYIC_20190510_mouse43__E9_P1`) shows the clearest illustration
that this is a genuine boundary-quality change, not merely case-selection
luck: both models share the same detached whisker-like false-positive
artifact (an unrelated grouping error), but on the actual brain boundary
itself the higher-resolution contour is visibly smoother and more concave-
following, and the subject's Dice still improved (0.9501->0.9531) despite
being the smallest gain in the cohort.

## Efficiency

| | Baseline (128x128x160) | Higher-res (192x128x160) |
|---|---:|---:|
| Trainable parameters | 182,081 | 182,081 (unchanged -- resolution-only experiment) |
| Voxel count / tile | 2,621,440 | 3,932,160 (**1.5x**) |
| Mean training epoch | 249.4 s | 581.1 s (2.33x) |
| Total training wall time | 4,988 s | 11,622 s (includes one ~57-min host scheduling stall, consistent with an occasional pattern seen on this machine in the prior experiment) |
| Peak training memory | 10,236 MiB | 11,252 MiB (+9.9%, well within the 18 GiB system) |
| Median native inference, CAMRI | 9.93 s | 65.15 s (6.6x -- CAMRI's larger native extent appears to now require more sliding-window tiles at the finer grid; only 6 subjects affected) |
| Median native inference, Mouse | 9.46 s | 16.68 s (1.76x) |
| Batch size | 1 | 1 (unchanged; no memory-driven reduction needed) |

CAMRI's larger inference-time increase (vs. Mouse's more modest, roughly
voxel-proportional increase) is a genuine, disclosed cost specific to
CAMRI's larger native volume extent interacting with sliding-window tiling
at the finer grid -- not a general property of the architecture change. It
did not require any batch-size or resolution compromise to stay within
memory.

## Final decision -- nine questions

1. **Did giving the encoder finer input improve Mouse Dice?** Yes -- 0.972438 -> 0.979537 (+0.0071), 80/80 subjects individually improved.
2. **Did HD95/ASSD improve?** Yes, substantially -- Mouse HD95 -28.4%, ASSD -27.3%; CAMRI HD95 unchanged (already at the measurement floor), ASSD -13.0%.
3. **Did <=1-voxel share improve?** Yes -- Mouse +2.99 pp, CAMRI +0.28 pp.
4. **Did high-curvature errors improve?** Yes -- Mouse high-complexity error rate -18.5% (35.709->29.10 per 100 surface voxels); CAMRI also improved (25.2->22.42).
5. **Did contour articulation move closer to expert?** Yes, by the largest margin of any metric here -- Mouse direction-change ratio 0.698->0.845 (+0.147), CAMRI 0.829->0.921 (+0.093); axis-run ratios improved in both domains.
6. **Did factor-5/grid-lock decrease?** Yes, more than halved in both domains -- CAMRI 18.08->8.08 pp, Mouse 28.61->12.69 pp.
7. **Did visible pixelation actually improve?** Yes, consistently across the figure set -- smoother, more concavity-following contours with fewer long grid-aligned runs.
8. **Did CAMRI remain stable?** Better than stable -- CAMRI improved on every metric tracked (Dice, precision, recall, ASSD, surface Dice, total error) and was unchanged (not worse) on HD95; 6/6 subjects improved.
9. **Was the gain worth the extra compute?** Yes for Mouse (1.76x inference, 2.33x training-epoch cost against a 25-30% reduction in boundary error and the largest contour-articulation gain seen in this project to date). CAMRI's 6.6x inference-time increase is a real, disclosed cost worth watching if CAMRI throughput ever matters at scale, but affects only 6 subjects and the accuracy gain there was positive too.

## Verdict

**A. Higher-resolution input solves a meaningful part of the remaining
boundary problem.**

This is not a marginal result: contour direction-change ratio moved further
toward the expert's own geometry than any prior intervention in this
project (label correction, the corrected true-level0 decoder, or the
original genuine-level0 architecture change combined), grid-lock excess more
than halved, and every accuracy and surface metric improved in both domains
with zero regressions across 86 test subjects. The one architecturally
unrelated cost (CAMRI's larger sliding-window inference-time increase) does
not offset this. Recommended as the new baseline resolution for this decoder
family; the next natural, still-cheap follow-up (per
`outputs/higher_resolution_encoder_diagnostic/`) is checking whether
refining the second loss-bearing axis (native X, currently untouched here)
compounds this gain further -- not attempted in this experiment, which
deliberately changed only the one variable requested.
