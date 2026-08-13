# Corrected-label controlled retraining

## Controlled intervention

Only expert-mask interpolation changed: the former shared image-mode
`is_seg=False`, order-1 resampling plus integer truncation was replaced by
categorical `is_seg=True`, order-0 nearest-neighbour resampling. MRI
preprocessing, encoder, decoder, levels 1–4, one query, 170,401 parameters,
split, seed, loss, augmentation, optimizer, schedule, 0.5 threshold, native
export, and largest-26-connected-component filter were unchanged.

The new cache is `/tmp/rs2_corrected_label_features`; no old cache, checkpoint,
prediction, or result was overwritten.

## Pre-training verification (all 141 masks)

| Domain | Method | Native round-trip Dice | Net volume bias | Lost | Added | HD95 mm | ASSD mm |
|---|---|---:|---:|---:|---:|---:|---:|
| CAMRI | old | 0.984560 | -2.578% | 226,355 | 12,874 | 0.1375 | 0.02863 |
| CAMRI | corrected | 0.992641 | +0.121% | 46,377 | 46,006 | 0.0625 | 0.00797 |
| Mouse | old | 0.975802 | -4.042% | 519,466 | 42,639 | 0.1000 | 0.02693 |
| Mouse | corrected | 0.981712 | -0.084% | 222,292 | 212,552 | 0.0994 | 0.01925 |

Corrected masks remained exactly binary. The old one-direction shrinkage was
eliminated: native round-trip additions and losses became nearly balanced, and
mean volume bias fell below 0.13% in both domains. Remaining round-trip mismatch
is expected sampling disagreement, not systematic erosion.

## Training

Training began from the same original epoch-14 checkpoint, ran the same 20
epochs, and selected epoch 17. Balanced validation Dice was 0.976708 versus
0.976576 for the old run. The CAMRI safety stop did not trigger. The encoder
remained frozen and the architecture stayed unchanged.

Checkpoint:
`outputs/corrected_label_retraining/checkpoints/best_corrected_labels.pt`

## Untouched native test results

### Filtered predictions

| Domain | Metric | Old | Corrected labels | Change |
|---|---|---:|---:|---:|
| CAMRI | Dice | 0.982626 | 0.987029 | +0.004403 |
| CAMRI | IoU | 0.965862 | 0.974391 | +0.008528 |
| CAMRI | Precision | 0.995008 | 0.985825 | -0.009183 |
| CAMRI | Recall | 0.970597 | 0.988249 | +0.017652 |
| CAMRI | HD95 | 0.1333 mm | 0.1333 mm | unchanged |
| CAMRI | ASSD | 0.03034 mm | 0.01997 mm | -0.01037 mm |
| CAMRI | FP voxels | 4,648 | 14,607 | +9,959 |
| CAMRI | FN voxels | 33,693 | 12,791 | -20,902 |
| Mouse | Dice | 0.965685 | 0.969536 | +0.003850 |
| Mouse | IoU | 0.933669 | 0.940890 | +0.007222 |
| Mouse | Precision | 0.983609 | 0.966415 | -0.017194 |
| Mouse | Recall | 0.948492 | 0.972760 | +0.024268 |
| Mouse | HD95 | 0.2050 mm | 0.2009 mm | -0.0040 mm |
| Mouse | ASSD | 0.04605 mm | 0.04214 mm | -0.00391 mm |
| Mouse | FP voxels | 156,895 | 331,053 | +174,158 |
| Mouse | FN voxels | 481,837 | 254,256 | -227,581 |

Pooled post-filter FP/FN ratio moved from 0.138 to 1.142 for CAMRI and
0.326 to 1.302 for Mouse. Corrected labels removed the severe FN bias, but
slightly overshot toward FP, especially on Mouse. Total residual error fell
38,341→27,398 CAMRI voxels and 638,732→585,309 Mouse voxels.

Raw and filtered results, including fixed Surface Dice at 0.1/0.2/0.5/1.0 mm,
are in `aggregate_test_metrics.csv`.

## Boundary-distance and curvature results

Combined filtered error fell 677,073→612,707 voxels. However, the **remaining
error tail became farther**, not shorter:

| Distance | Old | New |
|---|---:|---:|
| <=1 voxel | 96.006% | 89.288% |
| >1–2 | 2.909% | 7.829% |
| >2–3 | 0.817% | 2.159% |
| >3–5 | 0.242% | 0.636% |
| >5 | 0.026% | 0.088% |

Physical mean distance increased 0.0397→0.0883 mm and P95 increased
0.200→0.300 mm. This happens because many zero-distance surface FN voxels were
recovered while more outward FP voxels remained; it must not be described as a
boundary-tail improvement.

High-complexity error rate decreased from 40.309 to 38.145 errors per 100
surface voxels (256,631→242,849 errors), while flat-region rate fell more,
26.294→22.386. Curvature failures improved modestly but remain preferential.

## Contour geometry

Contour articulation did not improve:

- CAMRI direction-change ratio: 0.824→0.821; P95 axis-run ratio 1.138→1.145;
  factor-5 native-y alignment excess 18.74→22.03 percentage points.
- Mouse direction-change ratio: 0.694→0.687; P95 axis-run ratio 1.177→1.175;
  maximum-run ratio 1.623→1.663; factor-5 excess 28.71→29.69 points.

Visible staircase/pixelation and factor-5 quantization therefore remain. This
is consistent with unchanged half-resolution level1 mask logits and continued
exclusion of level0.

## Interpretation

### A. Fixed by corrected labels

- systematic training-mask erosion;
- severe native-test FN/recall bias;
- a substantial fraction of total residual boundary error;
- CAMRI and Mouse overlap and ASSD.

### B. Improved but still present

- total boundary error and high-curvature error counts;
- Mouse filtered HD95, only slightly;
- FP/FN calibration, which passed through balance and became modestly
  FP-dominated.

### C. Unchanged or worse

- factor-5 grid alignment;
- reduced contour direction changes and long axis-run tail;
- visible pixelation;
- the proportion and physical distance of the residual >1-voxel tail.

## Decision

The experiment supports a next controlled **corrected-label + full-resolution
level0 decoder** test. Corrected supervision delivered clear recall, Dice, ASSD,
and total-error gains, while the spatial-resolution signatures did not improve.
That clean separation now justifies testing level0 as the sole architectural
factor. The new experiment should retain the corrected cache and otherwise
preserve this experiment's training and evaluation contract.
