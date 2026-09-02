# Where does the Mouse boundary staircase first appear? (fast diagnostic)

**Method.** Diagnostic-only, ~73 s total compute across two runs (well under
the 10-minute budget). No training, no optimizer, no full-cohort run, no
architecture change. Three already-characterized Mouse test subjects were
each pushed through **one direct, non-tiled** encoder+decoder forward pass
(all three fit inside a single 128x128x160 tile -- verified, `fits_single_tile:
True` for all three -- so no sliding-window inference was needed). Native
probability/prediction panels were **loaded from already-saved files**
(`outputs/true_full_resolution_level0_decoder/`), not recomputed. All
quantitative helpers (`model_axis_point`, `choose_native_boundary_point`,
`crop_plane`, `digital_contour_statistics`, `effective_native_footprint`,
`quantization_rows`, `probability_detail_statistics`, `feature_image`,
`align_probability_to_raw`) are reused unmodified from
`scripts/diagnose_model_spatial_resolution.py` and
`scripts/analyze_boundary_error_diagnostics.py`, the project's own prior,
already-validated spatial-resolution diagnostic.

Subjects (reused from the visual-comparison atlas, not re-derived):
strongly pixelated (`POLYIC_20190517_mouse39__E12_P1`), median
(`POLYIC_20190510_mouse37__E2_P1`), best TRUE-level0 case
(`POLYIC_20190517_mouse37__E3_P1`). One figure per subject:
`{role}_{subject}.png`, 8 panels tracing the same anatomical region from
native MRI through to the final native prediction. Full numbers:
`stage_metrics.csv`.

**Caveat, stated up front:** the crop point in each figure is
`choose_native_boundary_point` -- the point of *maximum expert/prediction
disagreement*, reused as-is from the existing diagnostic rather than
re-derived. That is the right point to find where detail is lost, but it
sometimes lands on a genuine local prediction error rather than a
purely-correct-but-blocky boundary (most visible in the "strongly_pixelated"
figure, where the model-space probability's wedge shape does not match the
expert's oval at all -- a real local miss, not only a resolution artifact).
This is flagged per-figure below rather than smoothed over.

## Key numbers

| Subject | Native spacing (mm) | Model spacing (mm) | Downsample factor (x,y,z) |
|---|---|---|---|
| strongly_pixelated | 0.0703 x 0.0703 x 0.300 | 0.25 x 0.20 x 0.16 | **2.28, 3.56, 0.67** |
| median | 0.1000 x 0.1000 x 0.400 | 0.25 x 0.20 x 0.16 | **1.60, 2.50, 0.50** |
| best_true_level0 | 0.0703 x 0.0703 x 0.300 | 0.25 x 0.20 x 0.16 | **2.28, 3.56, 0.67** |

Every model-space voxel already covers **1.6-3.6 native voxels** in-plane
(x/y) before the network sees anything. This is a hard, fixed ceiling on
representable detail set entirely by `native->model` preprocessing.

| Subject | Native expert dir-changes/100 steps | Model expert (corrected-NN) dir-changes/100 steps | Loss from resampling alone |
|---|---:|---:|---:|
| strongly_pixelated | 43.7 | 36.5 | **-16%** |
| median | 47.6 | 35.6 | **-25%** |
| best_true_level0 | 53.6 | 36.4 | **-32%** |

Just re-expressing the **true expert mask** on the model grid with correct
nearest-neighbor resampling -- no network involved -- already discards
16-32% of its native boundary-direction-change rate. This is a clean,
network-independent measurement of preprocessing-stage detail loss.

| Subject | Model-space probability isocontour subvoxel % | Native-mapped probability isocontour subvoxel % | Native pred dir-changes/100 | Model pred dir-changes/100 |
|---|---:|---:|---:|---:|
| strongly_pixelated | 50.0% | 50.0% | 31.8 | 58.0 |
| median | 50.0% | 50.0% | 34.3 | 35.1 |
| best_true_level0 | 49.5% | 50.0% | 27.6 | 45.9 |

The continuous probability field's "subvoxel-ness" (the fraction of its 0.5
isocontour that does **not** land exactly on a grid line -- i.e. how smooth
vs. grid-locked it is) is **essentially unchanged between model space and
native space** (49.5-50.0% both places). Native resampling neither smooths
nor further blockifies the continuous field -- it faithfully carries forward
whatever smoothness/blockiness the decoder already produced. This points
*away* from native resampling (D) as a source of *additional* detail loss.
Model-space predicted-boundary complexity is often *higher* than model-space
expert complexity (58.0 vs 36.5; 45.9 vs 36.4) -- consistent with the visual
atlas's "speckle" finding, not with the decoder being overly smooth.

Factor-5 native-y grid-alignment excess (prediction minus expert, percentage
points): strongly_pixelated **+31.2 pp**, median **+8.9 pp**,
best_true_level0 **+21.6 pp** -- the final native prediction's boundary is
measurably more grid-locked to the model's own lattice than the expert's is,
in every case.

## What the figures show

- **Panels 1->2 vs 3->4 (native vs. model-space expert):** the true shape is
  visibly simplified going from native to model space in all three cases
  (a rounder, more organic native contour becomes a coarser polygon at model
  resolution) -- this is preprocessing alone, confirmed by the 16-32% direct-changes
  loss above.
- **Panel 6 vs 7 (model-space probability vs. native-mapped probability):**
  same shape, same soft/hard texture, just upsampled -- confirms native
  resampling does not add detail loss (matches the ~50%/~50% isocontour
  finding).
- **Panel 7 vs 8 (continuous probability vs. thresholded prediction):** in
  every figure, panel 7 shows a genuinely soft gradient (multiple gray
  levels spanning several native pixels); panel 8 is a hard binary
  boundary. Thresholding visibly and necessarily converts the remaining
  smooth half of the signal into a sharp step -- this is the mechanical,
  unavoidable action of `probability >= 0.5`, not a separate bug.
- **`best_true_level0` figure** is the cleanest illustration: the V-shaped
  concave notch is reasonably preserved through preprocessing (panel 4) and
  decoding (panel 6), but the final thresholded boundary (panel 8) looks
  distinctly blockier than even the model-space expert -- because the
  smooth gradient's own banding (visible in panel 7, band-widths matching
  the ~2-3.6x downsample factor) gets snapped to hard steps at that same
  coarse scale by thresholding.
- **`strongly_pixelated` figure** shows a genuine local miss at the chosen
  point (model-space probability is wedge-shaped, expert is oval) --
  evidence of a real decoder/capacity error at this specific location, not
  only a resolution effect. This is a minor, second-order contributor, not
  the dominant pattern seen across all three figures.

## Verdict

**E -- combination, with two identifiable, unequal roles:**

- **(A) native->model preprocessing is the primary, fundamental bottleneck.**
  It fixes a 1.6-3.6x per-axis downsampling ratio before the network runs,
  and measurably destroys 16-32% of the true boundary's direction-change
  detail on the expert mask alone, with no network involved. Nothing
  downstream can recover detail this step already discarded.
- **(D) final 0.5 thresholding is the proximate mechanical cause of the
  visible staircase.** The continuous probability field is only ~50%
  subvoxel-smooth even in native space (unchanged from model space), so
  thresholding is what converts that remaining smoothness into a fully
  grid-locked binary boundary. This is necessary/mechanical, not a
  resampling bug -- native resampling itself (the coordinate mapping,
  panels 6->7) adds no additional measured loss.
- **(B) encoder level0 / (C) decoder are not shown to add extra blur beyond
  what (A) already imposes** -- model-space predicted-boundary complexity
  is comparable to or higher than model-space expert complexity in all
  three cases, and the isocontour-smoothness figure is essentially flat
  from model space to native space. The decoder's measurable defect is
  localized speckle/over-complexity (matching the visual atlas's finding),
  not systematic smoothing.

**Practical implication:** further decoder capacity changes are unlikely to
meaningfully reduce the staircase pattern on their own, since the ceiling is
set upstream at preprocessing (A) and made visible by thresholding (D). If
finer native-resolution boundaries are wanted, the preprocessing target
spacing/downsampling ratio is the first-order lever to test next --
diagnostically, not by retraining here.
