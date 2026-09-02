# Does deepening the level0 refinement stage target the right problem?

**Method.** Fast, read-only, no-training diagnostic (~2 minutes actual
runtime). No model forward pass, no weight changes. Reuses only two
already-computed artifacts from `outputs/higher_resolution_true_level0/`:
`quantization_comparison.csv` (per-axis grid-lock statistics, already
computed by `compare_higher_resolution_true_level0_results.py`) and
`per_subject_comparison.csv` (paths to already-saved predictions for the
full 86-subject test cohort). Question: is the CURRENT best model's
(192x128x160 TRUE level0) residual pixelation better explained by (a)
leftover grid quantization on the native-X axis, which was never refined and
would need a resolution fix, not a decoder fix, or (b) small-scale,
locally-inconsistent per-voxel decision noise, which added level0 refinement
depth could plausibly reduce?

## 1. Per-axis grid-lock excess (existing data, re-aggregated)

`prediction_minus_expert_alignment`: how much more periodically the
prediction's boundary aligns to a candidate grid spacing than the expert's
own boundary does, at the SAME candidate spacing (near 0 = prediction looks
as naturally-quantized as the expert; higher = the prediction shows an
artificial stair-step signature the expert doesn't have).

| Domain | Axis | Touched by 128->192 refinement? | baseline_128 excess | higher_res_192 excess |
|---|---|---|---:|---:|
| CAMRI | native-X | No (still 0.16mm model spacing) | -0.009 | -0.011 |
| CAMRI | native-Y | Yes (0.25mm -> 0.167mm) | 0.036 | 0.015 |
| Mouse | native-X | No | 0.004 | 0.002 |
| Mouse | native-Y | Yes | 0.070 | 0.036 |

**Finding, and it goes against the axis-quantization hypothesis:** the
untouched native-X axis shows near-zero grid-lock excess in *both*
conditions -- it was never the dominant source of this specific artifact.
The refined native-Y axis had the excess (correctly) roughly halved by the
192 change, but a clear, non-zero residual (0.015-0.036) remains even after
making that axis' grid finer. If the remaining pixelation were purely "not
enough voxels on the untouched axis," native-X would show the elevated
signature -- it doesn't. The signature that *does* remain sits on the axis
that already got a finer grid, which points toward something in how the
decoder assembles/upsamples that axis' information, not a pure input-density
shortfall.

## 2. Residual disagreement is dominated by small, isolated clusters

Connected-component sizes of leftover FP/FN voxels (current best model, full
86-subject test cohort, per-slice 2D, native voxels):

| Domain | clusters | median size | p75 | p95 | % <=3 voxels | % >=8 voxels |
|---|---:|---:|---:|---:|---:|---:|
| CAMRI | 7,187 | 1 | 3 | 11 | 80.8% | 7.9% |
| Mouse | 89,312 | 2 | 4 | 15 | 74.2% | 11.5% |

**~75-81% of all remaining disagreement clusters are 3 voxels or smaller.**
That is the signature of scattered, locally-independent per-voxel decision
noise, not long axis-aligned staircase runs (which would show up as a heavy
tail of large, elongated clusters -- the histogram instead falls off sharply
after ~4-5 voxels). See `level0_depth_diagnostic.png`, right panel, for a
representative example: a thin, speckled disagreement ring around the
boundary, not solid blocks.

## 3. Does that scale match what added depth could plausibly fix?

Current `level0_refinement` is one depthwise-3x3x3 + pointwise-1x1x1 pass:
effective receptive field ~3 model-grid voxels. That lines up almost exactly
with the median/p75 of the measured residual-cluster scale (1-4 voxels) --
i.e. the current stage has *just barely* enough receptive field to see one
of these clusters, with essentially no margin to reconcile a decision with
its neighbors. Stacking 2-3 such blocks (~5-7 voxel effective receptive
field, still cheap: level0 is the largest grid in the decoder but each added
block only adds one more linear-cost pass over it) would put the receptive
field comfortably past the p75-p95 range of the actually-measured cluster
sizes for both domains.

## Verdict

**A) The evidence favors trying the level0-refinement-depth experiment.**
Two independent, already-available signals agree: (1) the untouched
native-X axis is *not* where the remaining quantization artifact lives, so
this is not simply "refine the other axis too"; (2) residual error is
overwhelmingly small, scattered, single-to-few-voxel clusters, at a scale
current single-pass refinement can barely span. Both are consistent with
your original intuition -- the remaining gap looks like a decoder-side local
decision-consistency issue, not a leftover resolution shortfall on an
unfixed axis.

Caveat, stated plainly: this diagnostic shows the residual error's *shape*
is consistent with the depth hypothesis being worth testing -- it does not
prove added depth will fix it, since no new weights were run (per the
no-training constraint). The next step to actually test it is the small,
controlled retraining experiment described earlier (add one residual
depthwise+pointwise block, plus a residual skip, to `level0_refinement`;
same 192x128x160 input, same hyperparameters, same eval), not a further
diagnostic.
