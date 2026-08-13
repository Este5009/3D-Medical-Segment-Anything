# Post-filter boundary-error diagnostics

## Scope and provenance

This diagnostic used all 86 canonical untouched test outputs: 6 CAMRI and 80
Mouse subjects. Inputs were the saved expert masks, native MRI volumes,
continuous probability maps, and deterministic largest-26-connected-component
predictions from the unchanged epoch-17 mixed-domain system. There was no
training, inference, threshold selection, model change, or postprocessing
change. The checkpoint was `outputs/true_full_resolution_level0_decoder/checkpoints/best_true_level0_decoder.pt` and the recovered
configuration was `configs/true_full_resolution_level0_decoder.yaml`.

An expert surface voxel is expert foreground removed by one 26-connected binary
erosion. Error distance is the shortest Euclidean distance from every remaining
FP or FN voxel to this fixed expert surface. Voxel distance uses array-index
coordinates; physical distance uses the real native spacing in all three axes.

## Direct answers

### 1–3. How far are the remaining errors?

Across 552,860 post-filter error voxels:

- <=1 voxel: 91.74%
- >1 to 2 voxels: 6.04%
- >2 to 3 voxels: 1.70%
- >3 to 5 voxels: 0.46%
- >5 voxels: 0.05%

Physical distances:

- CAMRI: mean 0.083 mm, median 0.000 mm, P95 0.200 mm, maximum 1.118 mm
- Mouse: mean 0.073 mm, median 0.000 mm, P95 0.283 mm, maximum 1.982 mm
- Combined: mean 0.074 mm, median 0.000 mm, P95 0.283 mm, maximum 1.982 mm

Physical distances within the combined index-space bins:

- <=1 voxel: median 0.000 mm, P95 0.200 mm, maximum 1.000 mm
- >1 to 2 voxels: median 0.200 mm, P95 0.361 mm, maximum 0.922 mm
- >2 to 3 voxels: median 0.300 mm, P95 0.498 mm, maximum 1.039 mm
- >3 to 5 voxels: median 0.412 mm, P95 1.063 mm, maximum 1.612 mm
- >5 voxels: median 0.714 mm, P95 1.790 mm, maximum 1.982 mm

The voxel bins and millimetre summaries must not be converted into one another
with a single scale because the cohort contains anisotropic acquisitions.

### 4. FP or FN?

Combined residuals contain 272,705 FP and
280,155 FN voxels (FP/FN ratio
0.973); this is **balanced**.
Domain-specific counts and FP/FN distance summaries are in `fp_fn_summary.csv`.
Detached FP islands removed by the filter are not counted.

### 5–6. Spatial uniformity and terminal regions

Errors are not evaluated by raw counts alone: each acquisition-slice region is
normalized by its number of expert-surface voxels. The combined terminal-region
error rate is 0.857× the middle-60% rate. Thus terminal regions
are not strongly disproportionate by the predeclared 1.2× descriptive criterion.
The normalized regional rates range from
26.149 to
30.727 errors per 100 expert
surface voxels, so the distribution is
approximately uniform by the same 1.2× descriptive criterion;
the middle 60% has the highest rate.
Because NIfTI orientations vary (LPS, RIA, RPI), the report
does not invent rostral/caudal, dorsal/ventral, or left/right labels. “First” and
“final” refer only to normalized array-axis-2 acquisition position within the
expert-mask extent.

### 7. Local geometric complexity

Expert-surface complexity is 1 minus the local mean resultant length of
physical signed-distance normals. Subject-wise tertiles define flat, medium,
and high-complexity regions. High-complexity regions have 1.764×
the combined errors per 100 surface voxels of relatively flat regions
(34.680 versus 19.657). This
supports preferential
failure around curved/pixel-complex expert geometry.

### 8. MRI boundary evidence

MRI intensities were robustly standardized per subject. Mean physical gradient
magnitude is 5.760 z/mm at correctly localized surface regions and
8.637 z/mm at >2-voxel disagreement regions. Mean local
inside/outside contrast is 1.493 versus 0.788 z.
These measurements do not consistently support weaker local image evidence as the sole cause of larger errors.
Local variance results are retained in `image_boundary_analysis.csv`.

### 9. Probability behavior

Within the expert-surface neighborhood, 41.21% of erroneous voxels
have probabilities in [0.25, 0.75], while 32.51% meet the fixed
descriptive “confident incorrect” definitions (FP >=0.9 or FN <=0.1).
Consequently errors are **not predominantly uncertain**; this is diagnostic only and does not optimize or recommend a threshold.

### 10. Is the problem geometrically meaningful?

91.74% of residual voxels are within one
index-space voxel of the expert surface and 8.26%
extend farther. The complete distance distribution, physical P95/maximum,
terminal normalization, and complexity dependence should be considered
together; Dice alone does not establish one-voxel precision.

## Surface metrics

Mean filtered surface performance:

- CAMRI: Dice 0.9869, HD95
  0.133 mm, ASSD
  0.020 mm.
- Mouse: Dice 0.9724, HD95
  0.161 mm, ASSD
  0.036 mm.

Surface Dice is reported at fixed 0.1, 0.2, 0.5, and 1.0 mm tolerances. These
were selected before inspecting results to span the observed native in-plane
and through-plane resolutions, not to maximize performance.

## Figures

Sixteen role-selected figures (eight per domain) are listed in
`figures/manifest.csv`. Each uses native voxels with nearest-neighbor display,
shows the MRI, expert, filtered prediction, green/cyan contours, red/yellow
FP/FN map, voxel-exact zoom, and saved probability map.

## Limitations and next diagnostic

- Voxel distance is an index-space measure and is not rotation- or
  anisotropy-equivalent to millimetres; physical conclusions should use mm.
- The digital 26-connected surface and local-normal dispersion are robust but
  resolution-dependent approximations to continuous anatomy.
- Surface locations receive their nearest error voxel; this attribution tests
  association, not causality.
- MRI gradient, contrast, and variance are observational and cannot separate
  annotation ambiguity from acquisition noise or model behavior.
- Continuous probabilities are pre-filter model outputs. The filter changes
  only the binary mask and has no continuous “filtered probability” analogue.
- The historical probability exporter stored square in-plane arrays transposed.
  Each map was aligned by the unique shape-compatible axis permutation whose
  unchanged 0.5 threshold exactly reproduces the canonical saved raw mask.
- CAMRI has only six held-out subjects, limiting domain-specific inferential
  statistics.

The next diagnostic experiment should be a blinded, multi-rater boundary review
of the largest physical-distance and high-complexity regions, measuring expert
inter-rater surface distance on the same fixed cases. That would distinguish
model error from expert-boundary ambiguity without changing the model.
