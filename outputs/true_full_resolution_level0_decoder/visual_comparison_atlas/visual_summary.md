# Visual comparison atlas: corrected-label baseline vs. TRUE full-resolution level0

Read-only. Every mask/prediction shown was already saved by
`outputs/corrected_label_retraining/` (baseline) and
`outputs/true_full_resolution_level0_decoder/` (level0); nothing was
retrained or re-inferred to build this atlas. Case selection and slice
selection are both data-driven (see `manifest.csv` for the exact rule used
per case); no case or slice was chosen arbitrarily.

Colors: expert = green, baseline = red, TRUE level0 = cyan, FP = red fill,
FN = yellow fill. No mask, contour, or crop was smoothed.

Ten cases: eight Mouse (largest Dice improvement, median improvement,
smallest/no improvement, one of the two regressions, largest HD95
improvement, a high-curvature improvement, a previously pixelated case, and
the strongest automated concavity/lobe-separation case), plus one CAMRI
representative case and the CAMRI case with the largest baseline-vs-level0
difference.

Slice per case is whichever slice has the largest baseline-vs-level0 XOR
disagreement among slices with substantial expert content (>=10% of that
subject's largest expert slice, or a hard floor of 20 voxels) -- not the
center slice, and not necessarily the slice that maximizes the Dice change.
The reported Dice/HD95 numbers in every figure title are the **whole-volume**
metrics for that subject (matching `outputs/true_full_resolution_level0_decoder/per_subject_comparison.csv`),
not slice-level numbers.

## Per-case verdicts

### 1. Largest Dice improvement -- Mouse `POLYIC_20190517_mouse37__E3_P1`
Dice 0.9612 -> 0.9684 (+0.0072); HD95 0.2899 -> 0.2109 mm (-0.0790 mm).
**Clearly improves shape.** The expert mask at this slice is two separate
lobes with a real anatomical gap between them. The baseline renders them as
one solid, gap-free blob (a whole-mask convex-hull solidity of 0.945 versus
the expert's 0.833 -- it fills the concavity in). TRUE level0 renders a
visible narrowing/notch in the same place (solidity 0.887, roughly halfway
to the expert's true concavity) and its FP/FN map is visibly tighter around
both lobes than the baseline's. **Also shows visible speckle**: small
isolated holes inside the level0 mask (panel D) and small stray loops in the
zoomed contour (panel I) that the baseline does not have.

### 2. Median Dice improvement -- Mouse `POLYIC_20190510_mouse37__E2_P1`
Dice 0.9720 -> 0.9751 (+0.0031); HD95 0.2000 -> 0.1414 mm (-0.0586 mm).
**Clearly improves shape**, modestly. The baseline under-segments a concave
notch on the upper-right of the boundary (a visible yellow FN patch); level0
follows the expert's contour into that notch much more closely, cutting the
FN area substantially. No visible speckle at this slice.

### 3. Smallest/no improvement -- Mouse `POLYIC_20190510_mouse43__E2_P1_1`
Dice 0.9608 -> 0.9608 (+0.0001); HD95 0.2236 -> 0.2000 mm (-0.0236 mm).
**Similar.** At the auto-selected slice (a rostral/caudal terminal slice with
a genuine under-segmented dip along the bottom edge), both models miss the
same region by roughly the same amount -- the red and cyan contours sit
almost on top of each other in the zoomed panel. This is an honest "no
meaningful change" example, consistent with the whole-volume Dice being
flat.

### 4. Regression -- Mouse `POLYIC_20190510_mouse43__E9_P1`
Dice 0.9527 -> 0.9501 (-0.0026); HD95 0.4000 -> 0.4000 mm (unchanged).
**Worsens shape / introduces speckle.** Both models share the same
underlying failure -- a detached, whisker-like false-positive appendage to
the right of the true brain region that survives the largest-component
filter in both cases (so it must connect to the main mass elsewhere in 3D).
The core round region both models get right is comparable to the expert.
The difference is in the appendage itself: the baseline's version is one
smooth red blob, while level0's version is thinner and more fragmented
(finger-like extensions, small holes) -- a clear case of the sharper
full-resolution field turning one coherent (wrong) region into a noisier
(also wrong) one.

### 5. Largest HD95 improvement -- Mouse `POLYIC_20190629_polyic_mo__E3_P1_1`
Dice 0.9651 -> 0.9705 (+0.0055); HD95 0.3606 -> 0.2000 mm (-0.1606 mm).
**Similar, with a mild reduction of a shared artifact -- not a clean shape
win.** Both models place a large false-positive blob above the true (lower,
round) expert region -- again a shared grouping/attachment error, not a
boundary-precision difference. Level0's version of that blob is visibly
smaller/more compact than the baseline's (accounting for the whole-volume
HD95 improvement), but neither model removes it, and the true expert region
itself is matched about equally well by both. The headline HD95 number here
reflects a partial reduction of a shared over-segmentation error, not
improved contour fidelity on the correctly-detected structure.

### 6. High-curvature improvement -- Mouse `POLYIC_20190629_polyic_mo__E4_P1_4`
Dice 0.9729 -> 0.9751 (+0.0022); HD95 0.1414 -> 0.1000 mm (-0.0414 mm).
**Clearly improves shape**, modestly. This subject has the most complex
expert boundary (by direction-changes-per-100-steps) among all improving
Mouse cases. Level0's contour tracks two small concave notches (left side,
lower right) more closely than the baseline's, with visibly smaller FN
patches in those notches. No visible speckle.

### 7. Previously pixelated/staircase -- Mouse `POLYIC_20190517_mouse39__E12_P1`
Dice 0.9668 -> 0.9710 (+0.0042); HD95 0.2223 -> 0.2109 mm (-0.0114 mm).
**Clearly improves shape.** This is the Mouse subject with the highest
baseline axis-run/expert-axis-run ratio (the baseline's longest straight
contour segments are the most disproportionate to the expert's). In the
zoomed panel, the baseline's contour is visibly blockier along the lower
edge (longer axis-aligned jumps); level0's contour is smoother and hugs the
expert boundary's actual curvature more closely. No visible speckle.

### 8. Concavity / lobe separation -- Mouse `POLYIC_20190517_mouse37__E3_P1`
Same subject and slice as case 1 above (Dice 0.9612 -> 0.9684). An automated
whole-mask convex-hull-solidity scan of **all 80 Mouse subjects'** saved
masks (not a manual pick) found this the only case in the entire test set
where the expert shape is genuinely non-convex (solidity < 0.85) and level0's
solidity is measurably closer to the expert's than the baseline's is
(expert 0.833, baseline 0.945, level0 0.887). It is shown again here, not as
a distinct case, because it is unambiguously the strongest and only clear
example of this specific failure mode being corrected. **Clearly improves
shape, with visible speckle** (same caveat as case 1).

### 9. CAMRI representative -- `007`
Dice 0.9862 -> 0.9857 (-0.0005); HD95 0.1000 -> 0.1000 mm (unchanged).
**Similar.** Both contours track the expert's scalloped upper boundary
almost identically. The one visible difference is a small isolated
false-positive speckle at the bottom-right in the level0 FP/FN map that the
baseline does not have -- a minor, localized speckle artifact, not a shape
change.

### 10. CAMRI worst difference -- `099`
Dice 0.9867 -> 0.9876 (+0.0010); HD95 0.1000 -> 0.1000 mm (unchanged).
**Similar.** Both models produce a comparable small false-positive bump
above the true boundary at the same location (a shared error, not a
level0-specific one); everywhere else both track the expert's scalloped
boundary closely and about equally well.

## Overall reading

Consistent with `experiment_report.md`'s quantitative findings: on Mouse,
TRUE level0 produces real, visible shape improvements in the majority of
cases shown here (5 of 8: cases 1/8, 2, 6, 7), most dramatically the
concavity/lobe-separation case, which the half-resolution baseline cannot
represent at all. Two cases (3, and the shared-artifact case 5) are
genuinely similar rather than clear wins -- the whole-volume metric moved
mainly because of something other than boundary precision (a flat region, or
a shared grouping error shrinking rather than resolving). One case (4, the
regression) shows the clearest instance of the report's disclosed trade-off:
sharper full-resolution predictions can turn a smooth wrong region into a
speckled wrong region. On CAMRI, both representative cases are visually
near-identical to the baseline, matching the report's "neutral" verdict for
that domain; the only visible level0-specific artifact across both CAMRI
cases is one small isolated FP speckle in case 9.
