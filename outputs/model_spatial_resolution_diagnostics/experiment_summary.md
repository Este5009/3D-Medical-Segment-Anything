# Effective spatial-resolution diagnostics

## Controlled scope

The unchanged epoch-17 mixed-domain checkpoint was traced under inference mode
on CAMRI 064 and Mouse `POLYIC_20190510_mouse43__E9_P1`. The complete 6-CAMRI
and 80-Mouse canonical filtered cohort was used for native geometry and
quantization measurements. No training, parameter update, threshold change,
postprocessing change, or architecture change occurred.

## Direct architecture result

Actual live encoder shapes for a `[1,1,128,128,160]` tile are:

- level0 `[1,48,128,128,160]` (1×, available but unused);
- level1 `[1,48,64,64,80]` (2×; finest decoder input and mask grid);
- level2 `[1,96,32,32,40]` (4×);
- level3 `[1,192,16,16,20]` (8×);
- level4 `[1,384,8,8,10]` (16×).

All level1–4 tensors are projected to 32 channels. The top-down FPN performs
three trilinear feature interpolations (level4→3→2→1, `align_corners=False`),
with additive lateral fusion and 3×3 refinement at each level. The one query is
updated by cross-attention at all four fused scales. A 32-channel refined
level1 voxel map is dotted with the mask embedding, so segmentation logits are
**first produced at `[1,1,64,64,80]`**. There is exactly one final 2× trilinear
logit interpolation to `[1,1,128,128,160]`, `align_corners=False`; sigmoid and
thresholding occur afterward. There is no level0 skip into this decoder.

Therefore the encoder contains a full-model-grid feature, but the decoder does
not use it. The first architecture-internal loss of available encoder spatial
detail is the exclusion of level0; the first explicit mask representation is
already level1 resolution.

## Preprocessing result

The preprocessing order is transpose `[1,0,2]` → nonzero crop → z-score →
order-1 resampling without anti-aliasing to `(0.25,0.20,0.16) mm`. CAMRI 064
changes from transposed `(144,64,144)` at `(0.2,0.2,0.2) mm` to
`(115,64,180)`. The Mouse representative changes from `(180,35,180)` at
`(0.1,0.4,0.1) mm` in model axes to `(72,70,113)`. Mouse thus loses native
detail before encoding along the axes resampled from 0.1 to 0.25/0.16 mm.

The configured preprocessor also invokes the same `is_seg=False`, order-1
resampler for the expert segmentation and then casts it to `int8`. This is a
verified implementation fact and a plausible contributor to conservative/FN
supervision, but causality is not established here.

At level1 the physical grid is `(0.50,0.40,0.32) mm` in model axes. In native
NIfTI x/y/z order that is `(0.32,0.50,0.40) mm`, corresponding to Mouse
footprints of approximately 3.2×5.0×1.0 native voxels and CAMRI footprints of
1.6×2.5×2.0 voxels before the final interpolation.

## Is the prediction objectively coarser?

Mean axis-aligned contour run-length ratios (prediction/expert) are
1.000 for
CAMRI and 1.000
for Mouse. Direction-change ratios are
0.824
and 0.694,
respectively. Mean run length is therefore nearly unchanged, so a claim of
uniformly longer predicted runs is not supported. However, P95 run-length
ratios are 1.138
and 1.177,
maximum-run ratios are
1.380 and
1.623, and
direction changes per 100 steps are substantially lower. Together these show a
coarser upper tail and less articulated predicted contour, especially in Mouse,
rather than a universal increase in every run. The full per-subject evidence is
in `expert_prediction_geometry.csv`.

The strongest domain/axis modulo-alignment excesses are:

- CAMRI x: factor 3, prediction−expert +0.77 percentage points
- CAMRI y: factor 5, prediction−expert +18.74 percentage points
- Mouse x: factor 8, prediction−expert +0.49 percentage points
- Mouse y: factor 5, prediction−expert +28.71 percentage points

The strong Mouse y-axis factor of 5 exactly matches the traced level1 footprint
of 0.50 mm / 0.10 mm = 5 native pixels. CAMRI's y footprint is 2.5 pixels, so
its half-grid phase repeats every 5 native coordinates, also matching the
detected factor. The x-axis excesses are below one percentage point and are not
treated as meaningful detections.

Modulo alignment is only one test: non-integer native/model resampling factors
and crop phase can smear exact coordinate multiples. Run lengths and the traced
physical footprint are therefore the more direct coarseness evidence.

## Probability versus threshold

Across subjects, 19.86% of
native boundary-band probability samples lie strictly between 0.1 and 0.9, and
the 0.5 isocontour has subvoxel coordinates at
49.94% of coordinate
entries. Thus the mapped probability surface is not a piecewise-constant binary
block map. Thresholding converts a continuous interpolated surface to the native
voxel lattice and reveals staircase edges, but it does not create the underlying
low-resolution mask evidence: the first logits already exist only on level1.

## Relation to the prior boundary experiment

The architecture is consistent with 96.01% of errors lying within one native
voxel: trilinear and native resampling recover accurate coarse localization and
excellent Dice/ASSD. The absence of level0 in mask prediction and the anisotropic
level1 footprint predict difficulty representing rapidly changing contours,
consistent with the measured 1.53× high-complexity burden. The FN dominance is
compatible with coarse level1 representation and with linear-resampled,
integer-cast preprocessing labels, but this observational trace cannot assign
their causal shares.

## Answers

1. **Objectively coarser?** Yes in contour articulation and the long-run tail,
   especially Mouse; no for mean run length, which is essentially unchanged.
2. **First coarseness stage?** For Mouse, native-detail loss begins in
   preprocessing downsampling. Inside the architecture, level0 is discarded and
   logits first appear at level1.
3. **Native decoder-logit resolution?** `[1,1,64,64,80]` per tile.
4. **Model input?** `[1,1,128,128,160]`; native MRI shapes remain case-specific.
5. **Effective upsampling?** One explicit 2× trilinear logit interpolation, then
   case-specific order-1 native-grid resampling.
6. **High-resolution encoder feature available?** Yes, level0 at full input grid.
7. **Used?** No. The decoder uses levels1–4.
8. **Where lost?** Mouse detail is first downsampled in preprocessing; remaining
   full model-grid encoder detail is then excluded at the decoder interface.
9. **Probability coarse?** Continuous after interpolation, but derived from a
   half-resolution native logit grid.
10. **Threshold effect?** It reveals lattice staircases; it does not originate
    the half-resolution evidence.
11. **Resampling contribution?** Yes, especially Mouse native-to-model
    downsampling and case-specific model-to-native scaling.
12–13. **Coordinate quantization?** Exact modulo results are reported above and
    must be interpreted with the non-integer footprint; geometric coarseness
    corresponds more directly to the traced level1 scale.
14. **Complex-region errors?** The half-resolution mask grid and discarded
    level0 are mechanically consistent with the 1.53× effect, but do not prove
    that architecture is its sole cause.

## Limitations and next diagnostic

- Two volumes were traced because tensor topology is input-tile invariant;
  native resampling was audited separately for all 86 saved cases.
- Feature RMS images show spatial sampling, not semantic information content.
- Digital contour metrics depend on native anisotropy and acquisition grid.
- Exact lattice modulo tests lose sensitivity under non-integer resampling.
- The label-resampling observation is mechanistically important but has not been
  isolated experimentally.

The next scientifically controlled experiment should be an **offline label and
image resampling fidelity audit**: for the same locked masks, compare current
order-1/integer-cast preprocessed labels against nearest-neighbor labels and
measure boundary displacement, curvature loss, and FN-biased erosion—without
training or changing model predictions. This isolates whether supervision was
made coarse before testing any architecture modification.
