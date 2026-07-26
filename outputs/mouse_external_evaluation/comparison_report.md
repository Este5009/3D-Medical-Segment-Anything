# Mouse external evaluation compared with CAMRI rat

## Locked evaluation contract

No training or fine-tuning was performed. The only decoder checkpoint was epoch 14 from `outputs/generalization_pilot/best_checkpoint.pt`. Strict loading verified one learned query with shape `[1, 1, 32]`, 170,401 decoder parameters, a frozen RS2-Net encoder, evaluation mode, no optimizer, probability threshold 0.5, and no connected-component post-processing.

## Dataset audit

The recursive audit found 101 images and 101 masks, paired all 101 by normalized filename, and excluded none. All pairs passed native geometry and non-empty safely binarizable mask checks without correction. There are 15 explicitly identified biological mouse IDs spanning 49 scans. The remaining 52 filenames omit a recoverable mouse ID; they remain scan-unique, so the true total number of biological mice cannot be established from the released names. This prevents invented longitudinal linkage.

## Volumetric transfer results

Mouse mean Dice was **0.8828**, median 0.8888, and worst case 0.8015. Mean precision was 0.8087, mean recall 0.9734, and mean HD95 4.387 mm. All scans reached Dice 0.80, 31.68% reached 0.90, and none reached 0.95.

Failures are dominated by false positives: the combined native slices contain 2,762,237 FP voxels versus 311,791 FN voxels, consistent with high recall but lower precision. The first 20% is weakest at the Dice 0.90 threshold (14.03%), followed by the last 20% (22.86%); the middle 60% reaches 58.96%.

## CAMRI comparison

| Metric | Mouse transfer | CAMRI rat external holdout |
|---|---:|---:|
| Mean Dice | 0.8828 | 0.9779 |
| Mean precision | 0.8087 | 0.9964 |
| Mean recall | 0.9734 | 0.9602 |
| Mean HD95 (mm) | 4.387 | 0.196 |

The large Dice gap is principally precision loss/over-segmentation rather than loss of brain retrieval. Non-empty mouse slices reach Dice >=0.90 in 43.12% of cases versus 89.74% for CAMRI; at >=0.95 the values are 15.82% and 81.71%.

## Longitudinal consistency

The 15 explicitly named mice were evaluated across their available dates. 2 adjacent predicted-volume changes exceeded the prespecified descriptive 20% flag. These flags are diagnostics, not claims that anatomy must remain unchanged. Anonymous scans are excluded from longitudinal linkage because their identity is not recoverable.

## Interpretation and next step

The query decoder never trained on these exact mouse scans, so the result supports cross-dataset, cross-animal-domain **decoder transfer**: it consistently retrieves most expert brain tissue, but boundaries are substantially over-expanded relative to the mouse labels. It does not prove fully external encoder generalization because the original RS2-Net encoder had prior mouse-domain exposure, though not necessarily these exact images.

The most scientifically informative next step is **C: evaluate another truly unseen dataset** while preserving the model. That separates encoder-domain generalization from adaptation effects. Training on mouse data or unfreezing the encoder would answer a different question and should follow only after this locked benchmark is preserved.
