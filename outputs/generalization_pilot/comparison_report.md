# Generalization pilot report

## Experimental design

- Architecture was unchanged: frozen RS2-Net encoder, exactly one learned
  query, four-scale attention, top-down FPN, and one mask head.
- Trainable parameters: 170,401; all encoder parameters remained frozen and
  gradient-free.
- Of 131 image/mask pairs, 124 passed exact shape and affine validation. Seven
  subjects were excluded before splitting because their affines disagreed:
  051, 052, 069, 092, 105, 107, and 116.
- A deterministic 40-subject subset (30.5% of all pairs, 32.3% of valid pairs)
  was split by subject into 28 training, 6 validation, and 6 test subjects.
  There was no subject or slice leakage.
- The decoder was trained from scratch. Validation Dice alone selected the
  checkpoint and triggered early stopping. Test subjects were evaluated once
  after checkpoint selection.

## Results

| Split | N | Mean Dice | Median Dice | SD | Mean precision | Mean recall | FP voxels | FN voxels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 28 | 0.98812 | 0.98830 | 0.00122 | 0.99047 | 0.98580 | 55,181 | 76,635 |
| Validation | 6 | 0.98280 | 0.98769 | 0.00801 | 0.98215 | 0.98354 | 23,334 | 20,180 |
| Test | 6 | 0.98783 | 0.98799 | 0.00123 | 0.99000 | 0.98568 | 12,448 | 16,462 |

The best checkpoint was epoch 14. Training stopped at epoch 20 after six epochs
without a material validation improvement.

- Train-to-validation Dice gap: 0.00532.
- Train-to-test Dice gap: 0.00029.
- Best test subject: sub-099, Dice 0.98918.
- Worst test subject: sub-064, Dice 0.98602.
- Worst overall subject: validation sub-112, Dice 0.96936.

## Failure analysis

Visual review of the automatically ranked failures shows thin boundary errors,
not complete retrieval failures. The dominant pattern is false-negative
under-segmentation along the ventral/inferior brain boundary, with smaller
false-positive islands along the dorsal and lateral boundary.

The two clear validation outliers were:

- sub-112: Dice 0.96936, precision 0.97285, recall 0.96589, 6,827 FP and
  8,640 FN voxels. Its dominant error is ventral under-segmentation.
- sub-109: Dice 0.97399, precision 0.95957, recall 0.98884, 9,426 FP and
  2,524 FN voxels. Its dominant error is boundary over-segmentation.

Across all 40 subjects, Dice correlated moderately negatively with brain volume
(`r = -0.416`) and original slice count (`r = -0.457`). The largest, 64-slice
cases include the worst validation subjects. This suggests that field-of-view,
volume, or acquisition-shape variation is harder than the common 12-slice CAMRI
case. The pilot is too small to separate these correlated causes reliably.

The worst-10 figures contain MRI, prediction, ground truth, and an FP/FN overlay
(red = FP, blue = FN). Model-space NIfTI predictions accompany every ranked
best/worst case.

## Generalization assessment

### Is the decoder generalizing?

Yes, within this CAMRI pilot. Independent test Dice (0.98783) nearly matches
training Dice (0.98812), and every test subject exceeds 0.986.

### Is there evidence of overfitting?

No material evidence. The train-test gap is 0.00029 and the train-validation
gap is 0.00532. Validation variability is higher because two of six validation
subjects are harder acquisition/volume cases, not because training performance
continued rising after validation collapsed. Early stopping selected epoch 14.

### Is the frozen encoder limiting performance?

Not at this stage. A frozen encoder supports 0.98783 test Dice and balanced
precision/recall. Residual errors are narrow boundaries and acquisition-shape
outliers; these results do not justify partial unfreezing, which would add
overfitting risk before the frozen baseline has been tested on all valid data.

## Efficiency and limitations

- Mean decoder-only CPU inference: 0.302 seconds per subject.
- Peak process RSS: 9,687 MiB. This includes Python, the frozen encoder, and
  temporary tensors; it is not accelerator-only memory.
- The 40 frozen feature sets were cached temporarily as float16 files under
  `/tmp` so the unchanged CPU encoder ran once per subject.
- Predictions remain in padded/resampled model space rather than native NIfTI
  geometry.
- Seven affine-mismatched pairs must be corrected or explicitly excluded from
  any full-data protocol.

## Recommendation

Choose **C: move to the full dataset**, using all geometry-valid subjects with
a new locked subject-level split while keeping the encoder frozen. This is the
cleanest test of whether the strong pilot result persists. Do not partially
unfreeze the encoder yet, and do not modify the decoder architecture. Revisit
encoder unfreezing only if a full-data frozen-encoder experiment exposes a
consistent performance ceiling or acquisition-specific failures.
