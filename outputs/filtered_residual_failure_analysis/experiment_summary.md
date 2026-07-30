# Filtered residual failure analysis

## Controlled method

The original 86 untouched test predictions and their already-saved deterministic
filtered counterparts were analyzed in native geometry. The only changed input
to the original residual-analysis implementation was `prediction_path`.

Filtering was produced with `scipy.ndimage.label`, a full 3×3×3 structuring
element (26-connectivity), and retention of the largest foreground component.
It is full-volume 3D filtering. It uses no morphology, size threshold, image,
expert label, subject identity, or tuned parameter.

Boundary errors retain the original 0.5 mm physical band. Attribution priority
remains detached FP → terminal → leakage → detached FN → localization →
remaining boundary/local contour. Unclassified voxels were explicitly checked.

## Absolute error comparison

| Category | Baseline voxels | Filtered voxels | Difference | Relative change | Subjects before | Subjects after |
|---|---:|---:|---:|---:|---:|---:|
| Boundary Error | 673,653 | 673,784 | +131 | 0.02% | 86 | 86 |
| Detached False Positive Island | 17,961 | 0 | -17,961 | -100.00% | 32 | 0 |
| Detached False Negative Region | 0 | 0 | +0 | 0.00% | 0 | 0 |
| Terminal Slice Failure | 2,064 | 1,933 | -131 | -6.35% | 16 | 14 |
| Leakage | 1,356 | 1,356 | +0 | 0.00% | 57 | 57 |
| Major Localization Failure | 0 | 0 | +0 | 0.00% | 0 | 0 |

Total residual error changed from 695,034 to 677,073 voxels:
17,961 residual voxels disappeared
(2.584%).

Exclusive boundary attribution changed by
+131 voxels, from
673,653 to 673,784. Its share
rose from 96.924% to
99.514% because detached errors
were removed from the denominator. The direct, non-exclusive physical
0.5-mm boundary mask changed from
673,333 to
673,281
(-52). The exclusive +131
shift is reclassification from the higher-priority terminal category, not newly
created prediction error.

Terminal attribution changed from 2,064 to
1,933; leakage changed from
1,356 to 1,356. Changes are
explained by exclusive attribution: voxels removed as detached components can
overlap the non-exclusive terminal mask, while connected leakage in the retained
primary component is unaffected.

## Segmentation metrics

| Domain | Condition | Mean Dice | Mean IoU | Mean precision | Mean recall | Mean HD95 mm |
|---|---|---:|---:|---:|---:|---:|
| CAMRI | baseline | 0.982518 | 0.965656 | 0.994785 | 0.970597 | 0.1333 |
| CAMRI | filtered | 0.982626 | 0.965862 | 0.995008 | 0.970597 | 0.1333 |
| Mouse | baseline | 0.964762 | 0.931977 | 0.981696 | 0.948492 | 0.3435 |
| Mouse | filtered | 0.965685 | 0.933669 | 0.983609 | 0.948492 | 0.2163 |
| Combined | baseline | 0.966001 | 0.934327 | 0.982609 | 0.950034 | 0.3289 |
| Combined | filtered | 0.966867 | 0.935915 | 0.984404 | 0.950034 | 0.2105 |

Full mean, median, standard deviation, best, and worst summaries are stored in
`summary.json`; subject-level values are in `per_subject_metrics.csv`.

## Detached-component safety audit

Filtered predictions contain no detached false-positive island. Baseline had
17,961 exclusively
attributed detached-FP voxels across
32
subjects. Removed components overlapped 0 expert voxels.
Therefore no true expert anatomy was removed.

The audit uses full-volume connectivity, so structures that appear disconnected
within a single 2D slice but join through adjacent slices remain one component.
There are no morphological operations that could erode thin structures.

## Qualitative findings

The six fixed figures show best, median, and worst filtered Dice separately for
CAMRI and Mouse. Their three displayed slices are selected only from the
baseline prediction extent, never from expert error. The regenerated atlas
contains mild, representative, and severe examples where a filtered category
exists. The detached-FP atlas is intentionally empty and states:

> No detached false-positive islands remained after deterministic full-volume
> connected-component filtering.

## Answers

1. Detached FP islands were eliminated: yes, from 57 components/17,961 raw
   component voxels to zero.
2. 17,961 exclusive residual voxels disappeared.
3. No expert anatomy was removed; removed-component expert overlap was zero.
4. The direct boundary mask changed by
   -52 voxels; exclusive
   boundary attribution increased by
   +131 through category
   reclassification.
5. Its attribution percentage rose primarily because the total-error
   denominator decreased, with a small terminal-to-boundary attribution shift.
6. Terminal exclusive attribution changed by
   -131 voxels because
   detached terminal-position FP voxels were removed first in the attribution.
7. Connected leakage changed by +0
   voxels; the primary connected component itself was unchanged.
8. Boundary/local contour error now dominates at
   99.514%.
9. The filter should become the default binary inference cleanup for this
   single-coherent-object task, with the unfiltered probability/logit map retained.
10. The next architectural target should be boundary refinement, not grouping.

## Conclusion

**A. Deterministic filtering should become the default inference pipeline and
future work should focus on boundary refinement.**

The conclusion follows the absolute result: detached islands were fully removed,
no expert voxels were lost, Dice/precision and Mouse HD95 improved, recall was
unchanged, and boundary disagreement is now the overwhelmingly dominant error.
