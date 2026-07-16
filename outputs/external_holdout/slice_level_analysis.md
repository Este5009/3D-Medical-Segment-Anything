# External-holdout slice-level Dice analysis

## Scope and policy

This analysis uses all 3,900 saved native slices from all 91 external-holdout subjects. Of these, 3,439 contain expert foreground and 461 have an empty expert mask. Subject geometry and saved metrics were revalidated before analysis. No inference, training, thresholding, post-processing, prediction modification, or metric modification was performed.

Empty expert slices are reported separately because treating a correctly empty slice as Dice 1.0 would inflate the apparent segmentation distribution. Cumulative Dice results below therefore use only non-empty expert slices.

## Complete range distribution

| Dice range | Slices | % of all slices | % of non-empty expert slices | Contributing subjects |
|---|---:|---:|---:|---:|
| Empty expert and empty prediction | 434 | 11.13% | 0.00% | 53 |
| Dice = 0.00 | 63 | 1.62% | 1.05% | 38 |
| 0.00 < Dice < 0.50 | 59 | 1.51% | 1.72% | 33 |
| 0.50 ≤ Dice < 0.70 | 41 | 1.05% | 1.19% | 30 |
| 0.70 ≤ Dice < 0.80 | 48 | 1.23% | 1.40% | 34 |
| 0.80 ≤ Dice < 0.90 | 169 | 4.33% | 4.91% | 53 |
| 0.90 ≤ Dice < 0.95 | 276 | 7.08% | 8.03% | 53 |
| 0.95 ≤ Dice < 0.97 | 430 | 11.03% | 12.50% | 66 |
| 0.97 ≤ Dice < 0.98 | 877 | 22.49% | 25.50% | 91 |
| 0.98 ≤ Dice < 0.99 | 1,461 | 37.46% | 42.48% | 91 |
| 0.99 ≤ Dice ≤ 1.00 | 42 | 1.08% | 1.22% | 9 |

The 63 zero-Dice slices comprise 36 non-empty expert slices and 27 empty-expert slices with false-positive prediction foreground.

## Cumulative non-empty-slice performance

| Scope | ≥0.80 | ≥0.90 | ≥0.95 | ≥0.97 | ≥0.98 | ≥0.99 |
|---|---:|---:|---:|---:|---:|---:|
| All subjects | 94.65% | 89.74% | 81.71% | 69.21% | 43.70% | 1.22% |
| 12-slice group | 100.00% | 100.00% | 100.00% | 95.27% | 74.55% | 0.90% |
| 64-slice group | 93.86% | 88.21% | 79.00% | 65.34% | 39.13% | 1.27% |
| First 20% | 92.35% | 83.36% | 70.01% | 47.38% | 17.84% | 1.05% |
| Middle 60% | 99.90% | 99.48% | 97.81% | 91.64% | 63.56% | 1.66% |
| Last 20% | 80.36% | 65.37% | 42.58% | 20.24% | 6.90% | 0.00% |

Thus, among all non-empty expert slices, 89.74% achieved Dice ≥0.90, 81.71% achieved ≥0.95, 69.21% achieved ≥0.97, 43.70% achieved ≥0.98, and 1.22% achieved ≥0.99.

## Empty and non-empty slice diagnostics

Among 461 empty-expert slices, 434 had a truly empty prediction and 27 contained some predicted foreground. The false-positive empty-slice rate was 5.86%.

Among 3,439 non-empty expert slices:

- 32 had zero predicted foreground;
- 95 had Dice below 0.50;
- 353 had Dice below 0.90;
- 2,810 had Dice strictly above 0.95;
- 1,503 had Dice strictly above 0.98.

Strictly-above counts differ slightly from the cumulative `≥` thresholds by design.

## Interpretation

### Which slice positions fail most often?

The last 20% of the brain extent is clearly the weakest region: only 65.37% of those slices reach Dice 0.90 and 42.58% reach 0.95. The first 20% is intermediate at 83.36% and 70.01%, respectively. The middle 60% is highly consistent, with 99.48% reaching 0.90 and 97.81% reaching 0.95. Failures are therefore concentrated at tapering peripheral anatomy, especially the terminal end of the volume.

### Are 64-slice acquisitions weaker at high Dice thresholds?

Yes, descriptively. For Dice ≥0.95, the 64-slice group reaches 79.00% versus 100.00% for the 12-slice group; at ≥0.98 the values are 39.13% versus 74.55%. This is an acquisition-group association, not proof that slice count itself causes the difference: spacing, anatomy coverage, cohort composition, and target complexity also differ.

### Are errors concentrated in a small minority of slices?

Major errors are concentrated in a minority: 94.65% of non-empty slices reach Dice 0.80, 89.74% reach 0.90, and only 2.76% fall below 0.50. However, the stricter boundary-quality criterion is less uniformly met—18.29% fall below 0.95—so performance should not be summarized only as rare catastrophic failures. The dominant pattern is excellent central-brain consistency plus a smaller but meaningful tail of peripheral under-segmentation.

## MedSAM comparison readiness

A fair MedSAM comparison requires slice-level outputs generated under exactly the same evaluation contract:

1. predictions for the identical 91 subject IDs and every identical native slice;
2. predictions restored to the same native array shape, affine, spacing, orientation, and slice axis;
3. the same binary expert masks and foreground convention;
4. the same empty-slice policy, separately distinguishing true-empty predictions from false-positive empty slices;
5. the same unsmoothed per-slice Dice definition, including the agreed handling of two empty masks;
6. the same prediction threshold and any connected-component or other post-processing rules, recorded explicitly;
7. per-slice TP, FP, FN, Dice, and expert/prediction foreground flags;
8. the same acquisition-group and normalized brain-position labels;
9. subject-level pairing so comparisons can use paired differences rather than unrelated cohort averages.

Equivalent MedSAM results are not currently present, so this report makes no superiority claim. The CSV and JSON outputs provide a ready schema for a controlled paired comparison once equivalent MedSAM native-space outputs exist.

## Outputs

- [Range distribution CSV](slice_dice_distribution.csv)
- [Machine-readable summary](slice_dice_distribution_summary.json)
- [Slice-distribution charts](visualizations_fixed/slice_distribution/)
- [Cleanup record](cleanup_report.md)
