# Expert-mask lifecycle trace

## Definitive paths

- **Training:** original expert NIfTI -> SimpleITK float32 `[C,Z,Y,X]` -> transpose `[1,0,2]` -> image-nonzero crop -> the shared `3d_fullres` `is_seg=False, order=1` resampler -> int8 truncation -> `>0` -> center pad/crop to `[1,1,128,128,160]` -> uint8 feature-cache target -> optional axis flips. This is a **linearly interpolated mask cast to integer**, option C in the audit question.
- **Validation:** identical preprocessing and cache target, without training augmentation. Validation Dice is prediction thresholded at 0.5 versus that processed model-space tile, not native GT. Concrete example: `/Users/estebanfelixmoran/Desktop/Purdue/Medical_Imaging/Datasets/Mask_Database/label_adult/POLYIC_20190510_mouse31__E2_P1.nii.gz`.
- **Canonical test:** preprocessing receives the GT path, but `preprocess()` discards the returned segmentation. Logits are restored to native geometry, thresholded, and compared with ``ground_truth_path`` reloaded directly by nibabel and binarized `>0`. Thus canonical Dice/IoU/precision/recall/HD95 and later ASSD/boundary diagnostics use untouched native expert labels.

## Concrete training example

`/Users/estebanfelixmoran/Desktop/Purdue/Medical_Imaging/Datasets/Mask_Database/RodentBrainMask/CAMRI Rat/CAMRI_Rat-sub-030_ses-1_acq-RARE_T2w_040.nii.gz` starts at shape `(12, 256, 256)`, spacing `(1.0, 0.10000000149011612, 0.10000000149011612)` mm and `137126` foreground voxels. Its processed target is `(102, 60, 160)` at `(0.25, 0.20000000298023224, 0.1599999964237213)` mm; after the current method it contains `167195` foreground voxels versus `171135` with nearest-neighbour. The actual cached tile is `[1,1,128,128,160]`.

## Important distinction

The faulty label interpolation directly affects train and validation supervision. It does **not** alter the expert mask used by the canonical native test metrics. The order-1 native export is applied to logits/probabilities, not expert labels.
