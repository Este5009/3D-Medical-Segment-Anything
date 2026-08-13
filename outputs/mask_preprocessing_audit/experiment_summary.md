# Mask preprocessing audit

This diagnostic analyzed all 141 locked mixed-domain subjects (40 CAMRI, 101 Mouse); no model inference or training was run.

## Findings

The current train/validation target is the original mask after transpose, image-nonzero crop, **image-mode order-1 resampling (`is_seg=False`)**, int8 truncation, `>0`, and center tiling. Validation uses the same cached representation. Canonical native test metrics reload the untouched expert NIfTI and therefore are not scored against this processed target.

| Domain | Current-vs-NN mean Dice | Mean volume change | Shrunk subjects | Lost/added ratio | Native round-trip Dice (current) | Native round-trip Dice (NN) |
|---|---:|---:|---:|---:|---:|---:|
| CAMRI | 0.985067 | -2.939% | 40/40 | inf | 0.984560 | 0.992641 |
| Mouse | 0.979678 | -3.983% | 101/101 | inf | 0.975802 | 0.981712 |

The distortion is classified as **meaningful** under the predeclared descriptive rule (absolute mean volume change at least 0.1% or mean processed Dice below 0.999). A pooled lost/added ratio above 1 and predominantly negative volume changes constitute evidence of inward/erosive bias. This makes preprocessing a plausible contributor to learned FN bias, but does not establish causation: the canonical test labels are untouched and model optimization/domain shift can independently produce FN errors.

## Decision

Correct label-aware preprocessing in a new, controlled cache and retrain/evaluate the unchanged decoder **before** interpreting a full-resolution decoder experiment. Preserve the current checkpoint/results as the comparator; do not retroactively alter reported native test metrics. The next experiment should change only segmentation interpolation and compare validation/native-test FN balance.
