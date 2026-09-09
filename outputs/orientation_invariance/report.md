# Space-invariance: rotation - combined summary

Both task queries of the shared dual-query model, 15 held-out subjects each, rotations of 5deg, 10deg, 20deg, 30deg about every axis.

## End-to-end: does the prediction move?

| query | mean self-consistency Dice | mean accuracy delta vs expert | resampling floor |
|---|---|---|---|
| brain | 0.9764 | -0.0119 | 1.0000 |
| lesion | 0.7992 | -0.0852 | 1.0000 |

## Where in the architecture the sensitivity enters

Mean relative L2 (baseline vs transformed-then-restored) per stage, averaged over every rotation tested. The resampling floor is ~1e-4 everywhere, so these are model effects.

| stage | brain | lesion |
|---|---|---|
| `input` | 0.153 | 0.121 |
| `encoder_level0` | 0.427 | 0.442 |
| `encoder_level1` | 0.294 | 0.298 |
| `encoder_level2` | 0.338 | 0.344 |
| `encoder_level3` | 0.228 | 0.246 |
| `encoder_level4` | 0.292 | 0.270 |
| `canvas_level4` | 0.292 | 0.270 |
| `canvas_level3` | 0.378 | 0.365 |
| `canvas_level2` | 0.403 | 0.411 |
| `canvas_level1` | 0.345 | 0.318 |
| `canvas_level0` | 0.323 | 0.285 |
| `output_logits` | 0.169 | 0.151 |
| `query_mask_embedding` (cosine) | 0.99986 | 0.99933 |

## What this says

1. **The frozen RS2-Net encoder is the dominant source of rotation sensitivity.** The input drifts ~0.15 in relative L2 when rotated and restored (pure interpolation cost), but `encoder_level0` drifts ~0.43 - the encoder *amplifies* pose differences rather than absorbing them. It is equivariant to neither rotation nor translation by construction (strided downsampling, padding), and the numbers confirm it.
2. **The trainable decoder trunk inherits that drift and neither removes nor compounds it.** `canvas_level0` sits at ~0.32, in the same band as the encoder levels it is built from. The trunk is not introducing a learned absolute-position dependence.
3. **The query-conditioning vector is essentially space-invariant** (cosine to baseline ~0.9996 across every axis and magnitude). The cross-attention pooling that produces `query_mask_embedding` reads a global summary that does not depend on pose - a real property of the mechanism, and the same for both the brain and lesion queries.
4. **The final logits still drift (~0.17)** because the decision is a dot product of the (stable) query embedding with the (drifted) `canvas_level0`; a stable query cannot un-drift the spatial map it multiplies.
5. **The end-to-end effect is governed by target size.** A large target (brain) stays near-invariant because the drifted logit field still crosses zero in almost the same place; a small target (lesion) visibly moves because a larger fraction of its voxels sit within one logit-noise width of the threshold.

_See each task's `report.md` and the CSVs for the per-axis, per-magnitude, per-subject numbers._
