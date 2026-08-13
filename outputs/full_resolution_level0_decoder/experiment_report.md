# Corrected-label full-resolution level0 decoder ablation

## Controlled question

This experiment changed one architectural factor: the corrected-label one-query
decoder was extended from level1 (64×64×80 learned logits, followed by 2×
interpolation) to level0 (128×128×160 learned residual logits). The RS2-Net
encoder remained frozen, the learned-query count remained one, and labels,
splits, initialization checkpoint, augmentation, loss, optimizer, threshold,
native export, and largest-26-connected-component filtering were held fixed.

The selected checkpoint is `checkpoints/best_level0_decoder.pt` at epoch 20.
The primary comparator is the corrected-label epoch-17 checkpoint, not the
historical damaged-label model.

## Architecture trace

Old: levels4→3→2→1 FPN and query attention → 32-channel level1 mask head →
64×64×80 logits → trilinear output interpolation.

New: the identical old path plus 48→8 level0 lateral projection, 32→8 level1
top-down projection, additive depthwise/pointwise refinement, one level0 query
update, and an 8-channel full-grid query/voxel dot-product residual. Only the
global level0 query-attention tokens are average-pooled 2× to control memory;
the mask field is not pooled. The output is natively `[1,1,128,128,160]` and no
final logit interpolation occurs at the training grid. A zero-initialized
learned residual scale made the transferred model exactly reproduce the old
decoder before training.

- Old trainable parameters: 170,401.
- New trainable parameters: 180,466 (+10,065; +5.91%).
- Encoder: frozen throughout.
- Query count: exactly one.

## Untouched native test results after identical filtering

| Domain | Model | Dice | IoU | Precision | Recall | HD95 (mm) | ASSD (mm) | FP | FN | Total error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAMRI (n=6) | corrected baseline | 0.987029 | 0.974391 | 0.985825 | 0.988249 | 0.1333 | 0.01997 | 14,607 | 12,791 | 27,398 |
| CAMRI (n=6) | level0 | 0.986841 | 0.974024 | 0.988121 | 0.985573 | 0.1333 | 0.02081 | 12,461 | 15,651 | 28,112 |
| Mouse (n=80) | corrected baseline | 0.969536 | 0.940890 | 0.966415 | 0.972760 | 0.2009 | 0.04214 | 331,053 | 254,256 | 585,309 |
| Mouse (n=80) | level0 | 0.964860 | 0.932158 | 0.954529 | 0.975543 | 0.2649 | 0.05524 | 452,000 | 228,375 | 680,375 |

CAMRI Dice changed by -0.000188 (three subjects improved and three regressed).
Mouse Dice changed by -0.004676; only 4/80 subjects improved and 76/80
regressed. CAMRI HD95 was unchanged and ASSD worsened slightly. Mouse HD95,
ASSD, and total error worsened. Mouse behavior shifted strongly toward FP.

For historical context only, the old damaged-label filtered Dice was 0.982626
CAMRI and 0.965685 Mouse. Correct label preprocessing remains beneficial, but
that does not make level0 beneficial relative to its valid corrected comparator.

## Boundary-distance and geometric endpoints

Combined residual-error bins changed as follows:

| Distance to expert surface | corrected baseline | level0 |
|---|---:|---:|
| ≤1 voxel | 89.288% | 83.301% |
| >1–2 voxels | 7.829% | 11.437% |
| >2–3 voxels | 2.159% | 3.742% |
| >3–5 voxels | 0.636% | 1.272% |
| >5 voxels | 0.088% | 0.248% |

Combined physical mean/P95 distance worsened from 0.0883/0.300 mm to
0.1315/0.400 mm. CAMRI alone became slightly more concentrated within one
voxel (93.766→94.533%) with lower mean physical distance (0.0914→0.0775 mm),
but its total error still increased. Mouse drove the overall regression:
≤1-voxel errors fell 89.079→82.837% and mean/P95 rose
0.0881/0.300→0.1338/0.400 mm. Thus the outward-error tail was not reduced.

Errors per 100 expert-surface voxels in high-complexity regions changed
25.232→26.101 CAMRI and 39.700→46.721 Mouse. Flat-region values changed
4.641→4.644 and 24.523→28.972. Curved anatomy did not benefit.

Contour direction-change/expert ratios moved only 0.8207→0.8243 CAMRI and
0.6866→0.6976 Mouse. Maximum axis-run ratios decreased 1.326→1.256 CAMRI and
1.663→1.591 Mouse, but Mouse P95 axis-run ratio worsened 1.175→1.205. Factor-5
native-y alignment excess decreased from 22.03 to 15.94 percentage points for
CAMRI, but barely changed from 29.69 to 29.30 points for Mouse. This is weak,
inconsistent evidence of finer contour rendering, not disappearance of the
factor-5 lattice or a clinically meaningful boundary improvement.

## Probability behavior

The level0 output is a genuinely learned full-grid residual, not merely a
higher-resolution visualization. Nevertheless, probability evidence did not
show useful finer localization. Boundary-band intermediate probability fell
11.65→10.77% CAMRI and 20.72→19.02% Mouse; Mouse boundary gradient magnitude
also fell 0.1978→0.1921 per native pixel. Confident incorrect boundary FP rose
27.48→38.30% on Mouse. Probability isocontour subvoxel-coordinate percentages
were essentially unchanged (~49.94%). The new field therefore became more
confident without becoming more correct.

## Efficiency

- Median end-to-end native inference: corrected baseline 9.91 s CAMRI and
  8.60 s Mouse; level0 10.32 s CAMRI and 9.31 s Mouse (CPU).
- Median level0 training epoch: 249.5 s; total recorded training wall time was
  12,413 s. Several host scheduling stalls make the 620.6 s arithmetic mean
  unsuitable for architecture comparison.
- New-run peak process memory: 8,036 MiB. The architecture trace measured about
  4,104 MiB peak for its second real-volume trace. A matching historical peak
  was not recorded for the corrected baseline, so an exact memory delta cannot
  be claimed.

## Direct answers

### A. Improved by full-resolution level0 decoding

- Native learned-logit resolution doubled per axis at the decoder output.
- CAMRI precision increased and its distance distribution improved modestly.
- Mean direction-change ratio and maximum flat-run ratio moved slightly toward
  the expert in both domains.
- CAMRI factor-5 alignment decreased.

### B. Unchanged

- CAMRI HD95.
- The visible staircase/lattice signature remained, especially in Mouse.
- Probability isocontour subvoxel behavior remained essentially unchanged.

### C. Worsened / tradeoffs

- Dice, ASSD, total error, and overall distance tail.
- Mouse HD95, FP count, high-curvature errors, and confident incorrect FP.
- Parameters, inference time, and memory demand increased.

The hypothesis is **not supported**: discarding level0 spatial information was
not the primary cause of the previous boundary limitation. Native full-grid
logits produced small contour-statistic changes but no meaningful medical
segmentation benefit and materially harmed Mouse generalization. Per the
controlled stopping rule, do not add more decoder complexity on this path.

## Next diagnostic experiment

Without changing the model, evaluate whether the factor-5 pattern originates
in the encoder/preprocessing coordinate transform by measuring boundary-event
alignment directly in native level0 activations and in the corresponding
preprocessed image gradients, stratified by Mouse acquisition group. This can
separate inherited encoder-grid/domain effects from decoder capacity without
training or threshold selection.

## Validation

- All 86 locked subjects were evaluated: 6 CAMRI and 80 Mouse.
- MRI, expert, raw prediction, filtered prediction, and probability shapes
  matched for every subject; filtered masks were subsets of raw masks.
- Probability maps were accepted only when unchanged 0.5 thresholding exactly
  reproduced their canonical raw prediction after a unique axis alignment.
- Fifteen focused tests passed via direct invocation. `pytest` itself was not
  installed in the `rs2` environment.

