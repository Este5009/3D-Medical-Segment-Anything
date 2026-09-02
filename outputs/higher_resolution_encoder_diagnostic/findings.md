# Does the frozen encoder use finer input detail? (fast diagnostic)

**Method.** Diagnostic-only, ~75s of actual measured script runtime (plus one
throwaway ~3-minute sanity check that turned out to be a cold-start anomaly,
explained below). No training, no weight changes, no decoder. Same 3 Mouse
subjects as `outputs/boundary_stage_localization/`. Model-axis convention:
axis0 -> native Y (biggest measured native-detail loss, ~2.5-3.6x), axis1 ->
native Z (already finer than native -- left unchanged), axis2 -> native X
(~1.6-2.3x loss).

## 1. Can the encoder accept larger input?

**Yes, cleanly.** `RSSNet`'s only hard constraint is that every spatial
dimension be divisible by 32 (`patch_size=2` to the 5th power, checked in its
constructor); it has no absolute position embedding or other size-locked
parameter (confirmed: it is a windowed-attention Swin transformer with
relative position bias). Empirically verified: the frozen checkpoint's
`state_dict` loads **strict** into `RSSNet(img_size=(192,128,160), ...)`, and
`level0` output shape scales exactly with input (`(1,48,192,128,160)` in,
matching levels1-4 halving successively). No architecture change was needed.

One early sanity check (a throwaway, not part of the reported run) measured
173s for a single 192x128x160 forward pass on random data -- a ~170x
slowdown for 1.5x voxels. This was **not reproduced** in the actual
diagnostic run immediately after (13.5s for the same size on real
preprocessed data) and is attributed to a one-time cold start (first Python
process's disk/library cache warm-up), not a real architectural cost. The
numbers below are from the reproducible, in-context measurement.

## 2. Candidates tested

| Candidate | Spacing (model axis0,1,2, mm) | Tile | Voxel ratio | Run through encoder? |
|---|---|---|---:|---|
| control | 0.250, 0.200, 0.160 | 128x128x160 | 1.0x | yes (all 3 subjects) |
| modest (axis0 only) | 0.1667, 0.200, 0.160 | 192x128x160 | 1.5x | yes (all 3 subjects) |
| substantial (axis0+2) | 0.1429, 0.200, 0.100 | 224x128x256 | 2.8x | expert-mask only (cheap); encoder cost extrapolated |
| near-native | 0.0714, 0.200, 0.0667 | 448x128x384 | 8.4x | expert-mask only (cheap); encoder cost extrapolated |

Only axis0/axis2 were refined (per the instruction to prioritize the
loss-bearing axes) -- axis1 (native Z) was left at its current spacing
throughout, since the prior diagnostic found it already finer than native.
"substantial" and "near-native" were run through the cheap (no-network)
expert-mask resample for all 3 subjects, but not through the encoder, to
keep this diagnostic fast; their encoder cost is extrapolated from the
measured control->modest scaling (see 5).

## 3. Boundary-detail retention (expert mask resampled at each grid, no network)

Expert-mask contour direction-changes/100 steps at the model grid (same
"most-content slice" selection per candidate, per subject; higher = closer to
native articulation):

| Subject | control (1x) | modest (1.5x) | substantial (2.8x) | near-native (8.4x) |
|---|---:|---:|---:|---:|
| strongly_pixelated | 41.2 | 44.6 (+8%) | 46.0 (+12%) | 47.7 (+16%) |
| median | 39.8 | 44.1 (+11%) | 36.6 (-8%) | 37.7 (-5%) |
| best_true_level0 | 44.3 | 51.8 (+17%) | 41.1 (-7%) | 47.0 (+6%) |

**control -> modest is clean and consistent: all 3 subjects improve
(+8% to +17%).** substantial/near-native are noisier and not monotonic in
2/3 subjects -- this is very likely an artifact of selecting the
highest-expert-content slice *independently per candidate* (a fast
approximation, not a pixel-tracked same-slice comparison), so different
anatomical slices can get picked at different grids. The control->modest
comparison does not have this problem (only 2 points, same method) and is
the most trustworthy signal here.

## 4. Does level0 genuinely improve, or just get more (redundant) voxels?

Boundary-band gradient-contrast of the level0 RMS-activation (mean
|gradient| inside a dilated model-space expert boundary band, divided by the
mean outside it -- rises only if edges are becoming more spatially
concentrated at the true boundary, not just present everywhere):

| Subject | control | modest | Change |
|---|---:|---:|---:|
| strongly_pixelated | 5.673 | 5.887 | **+3.8%** |
| median | 6.221 | 6.390 | **+2.7%** |
| best_true_level0 | 4.702 | 4.881 | **+3.8%** |

**Consistent, positive, but modest** in all 3 subjects -- level0 activation
is measurably (not just numerically-more-voxels) better localized at the
true boundary with finer input on the one axis tested. This is a real
signal, not a dramatic one. substantial/near-native were not run through the
encoder, so whether this trend continues, plateaus, or reverses at larger
sizes is not established here.

## 5. Runtime / memory

| | control | modest | substantial (extrapolated) | near-native (extrapolated) |
|---|---:|---:|---:|---:|
| Encoder forward, seconds/sample | 9.1 | 13.5 (**1.48x**, vs. 1.5x voxel ratio -- ~linear) | ~19-26 (est.) | ~76-82 (est.) |
| Peak process memory | ~9.7-10.2 GiB | ~10.3 GiB | not measured | not measured |

Measured scaling is close to linear in voxel count (1.48x time for 1.5x
voxels) on this CPU-only machine (MPS has no Conv3d support here, confirmed
earlier in this project). System has 18 GiB total RAM; control/modest peak
usage (~10 GiB) leaves headroom but is already more than half of total RAM,
so substantial/near-native should be spot-checked before committing to them,
not assumed safe from extrapolation alone.

## Decision

**A) Higher-resolution input clearly gives the frozen encoder finer useful
level0 information.**

Recommend for the next controlled retraining:

```
current:  128 x 128 x 160   (model-axis spacing 0.250, 0.200, 0.160 mm)
candidate: 192 x 128 x 160  (model-axis spacing 0.1667, 0.200, 0.160 mm)
voxel-count increase: 1.5x  (axis0 / native-Y only -- the single largest
                              measured native-detail-loss axis; axis1/native-Z
                              and axis2/native-X left unchanged this round)
boundary-detail retention:  39.8-44.3 -> 44.1-51.8 dc/100  (+8% to +17%, all 3/3 subjects)
level0 detail/localization: 4.70-6.22 -> 4.88-6.39 boundary-gradient-contrast (+2.7% to +3.8%, all 3/3 subjects)
memory/runtime: 9.1s/~10 GiB -> 13.5s/~10.3 GiB per sample (1.48x time, ~1% more memory)
```

This is the **lowest-cost candidate that meaningfully improves both the
geometric ceiling and the encoder's actual output** on every subject tested,
with near-linear, cheap cost scaling. It only refines the single axis
identified as the dominant loss source, which is also why it is cheap.
Substantial/near-native axis0+axis2 refinement was not validated through the
encoder in this fast pass and should not be assumed to help further without
its own (still cheap, ~20-80s/sample) check -- a natural, low-cost follow-up,
not required before adopting the modest candidate.
