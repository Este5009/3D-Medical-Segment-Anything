# Residual failure analysis

## Evaluation contract

All 86 untouched test predictions (6 CAMRI, 80 Mouse) were analyzed
in native space. No inference, training, threshold change, cleanup, or
post-processing was performed. Components use 26-connectivity. Local boundary
errors are defined within 0.5 mm of the opposite mask.

## Failure statistics

| Category | Affected subjects | Mean size, voxels | Mean slices | Mean centroid distance, mm | Exclusive error attribution |
|---|---:|---:|---:|---:|---:|
| Boundary Error | 86 (100.0%) | 51.1 | 35.62 | 0.35 | 96.92% |
| Detached False Positive Island | 32 (37.2%) | 315.1 | 2.81 | 7.55 | 2.58% |
| Detached False Negative Region | 0 (0.0%) | 0.0 | 0.00 | 0.00 | 0.00% |
| Terminal Slice Failure | 16 (18.6%) | 21.5 | 2.25 | 8.46 | 0.30% |
| Leakage | 57 (66.3%) | 11.6 | 2.11 | 4.68 | 0.20% |
| Major Localization Failure | 0 (0.0%) | 0.0 | 0.00 | 0.00 | 0.00% |

Exclusive attribution uses this order: detached FP, terminal, leakage, detached
FN, major localization, then remaining boundary/local contour error. Therefore
the attribution percentages sum to 100%, although subjects may appear in more
than one category.

## Anatomical grouping evidence

There were 57 detached predicted islands with no
expert overlap, containing 17,961 voxels.
25 islands
(7,540 voxels) had mean MRI
intensity within one expert-brain standard deviation. These are the measured
cases where local appearance could support foreground while 3D disconnection
contradicts membership. At most those
25 islands are
theoretically removable by stronger global grouping from the collected image
and topology evidence.

Grouping-associated categories account for 2.78% of all
exclusive error voxels. Boundary/local contour errors account for
96.92%, and terminal endpoint errors account for 0.30%.

## Scientific interpretation

1. **Dominant remaining failure mode:** the largest exclusive attribution is
   Boundary Error
   at 96.92%.
2. **Is boundary prediction the primary limitation?** Boundary/local contour
   disagreement accounts for 96.92% of measured error voxels.
3. **Is anatomical grouping becoming the primary limitation?** Grouping-related
   errors account for 2.78%; this is compared directly with
   the boundary fraction rather than inferred from Dice.
4. **Would stronger boundary supervision remove most errors?** It could address
   the measured 96.92% boundary/local fraction, but not detached,
   terminal, leakage, or distant localization voxels.
5. **Would explicit 3D grouping remove a substantial fraction?** The evidence
   supports at most 2.78% of current error voxels, including
   25 locally brain-like detached
   islands. This is an upper attribution, not a guaranteed improvement.
6. **Estimated grouping insufficiency:** 2.78% of remaining
   errors under the declared exclusive classification.

## Recommendation

**A. Current architecture is primarily boundary-limited.**

This is the sole recommendation and follows the measured boundary versus
grouping attribution above.
