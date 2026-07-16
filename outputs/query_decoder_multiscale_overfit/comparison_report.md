# Frozen-encoder one-query decoder comparison

## Decision

The multi-scale FPN decoder is the simplest tested architecture that nearly
perfectly fits both CAMRI subjects. It reached mean training Dice **0.98924**
with one query and a completely frozen RS2-Net encoder. Training peaked at
epoch 40 and then plateaued/degraded, so no additional capacity was added.

## Architecture

```text
Frozen RS2-Net encoder (no gradients)
        |
        +-- level4 [384,  8,  8, 10] -- 1x1 projection --+
        |                                                |
        +-- level3 [192, 16, 16, 20] -- 1x1 projection --+--> top-down FPN
        |                                                |       |
        +-- level2 [ 96, 32, 32, 40] -- 1x1 projection --+       v
        |                                                |   fused level1
        +-- level1 [ 48, 64, 64, 80] -- 1x1 projection --+       |
                                                                  |
one learned query --> cross-attend L4 --> L3 --> L2 --> L1       |
        |                                                         |
        +---------------- mask embedding dot product -------------+
                                  |
                         one 3D mask-logit volume
```

All projected and fused features use 32 channels. The decoder contains exactly
one learned query. Only the query, projections, attention/FFN blocks, FPN
refinements, and mask head are trainable.

## Controlled results

| Decoder | Trainable parameters | Mean Dice | Change from previous |
|---|---:|---:|---:|
| Minimal: level4 attention + level1 mask | 24,577 | 0.90308 | - |
| Four-scale attention, no FPN | 59,489 | 0.95344 | +0.05036 |
| Four-scale attention + top-down FPN | 170,401 | **0.98924** | +0.03580 |

The final decoder improved by **0.08616 absolute Dice** over the original
proof of concept. Subject Dice was 0.98861 for sub-001 and 0.98988 for sub-002.

## Why the changes helped

1. **Four-scale query attention** raised Dice from 0.90308 to 0.95344. The
   query no longer depended only on the coarsest semantic grid: level3 and
   level2 supplied intermediate structure, while level1 supplied local spatial
   evidence. This reduced gross retrieval errors. It was not sufficient for
   precise boundaries because its mask grid was still an independently
   projected level1 tensor with no semantic top-down fusion. Even after 100
   epochs it retained 5,876-10,172 false-negative/false-positive voxels per
   subject.

2. **Top-down FPN fusion and 3x3 refinement** raised Dice from 0.95344 to
   0.98924. Coarse semantic context was added into every finer grid before the
   mask dot product, allowing the high-resolution level1 representation to
   retain edges while disambiguating brain from nearby image structures. The
   qualitative comparisons show that the speckled probability boundary from
   the minimal decoder becomes compact and smooth. Errors fell to 1,598-1,935
   false negatives and 1,634-1,849 false positives per subject.

3. **More epochs did not justify more complexity.** The best checkpoint was at
   epoch 40 (0.98924). Dice reached 0.98971 transiently without satisfying the
   configured material-improvement rule, then declined; plateau stopping ended
   training at epoch 52. This is evidence to stop architecture expansion now.

These explanations are supported by the controlled attention-only ablation.
They remain tiny-set observations, not evidence of generalization.

## Efficiency

CPU decoder-only timing over 20 runs:

- Minimal decoder: 0.0105 seconds per volume.
- Multi-scale FPN decoder: 0.2807 seconds per volume.
- Multi-scale end-to-end estimate: 10.914 seconds per volume, of which 10.613
  seconds is the frozen encoder.

Peak process resident memory was 8,685.6 MiB for the multi-scale run versus
9,666.3 MiB reported by the previous run. These macOS RSS measurements include
the Python runtime and frozen encoder and should not be interpreted as precise
tensor-only memory. The current adapter temporarily constructs all encoder
features, including unused level0, which dominates memory.

## Figures and artifacts

- `learning_curves.svg`: minimal versus final multi-scale training Dice.
- `sub-001_previous_vs_multiscale.png`: previous result on top, final result below.
- `sub-002_previous_vs_multiscale.png`: previous result on top, final result below.
- The attention-only output directory contains its separate curve and
  qualitative ablation artifacts.
- Final binary predictions are saved as model-space NIfTI files.

## Failure cases and limitations

- Residual errors are thin false-positive and false-negative boundary bands;
  subject 001 remains slightly worse than subject 002.
- Predictions are still exported in padded/resampled model space, not restored
  to native NIfTI geometry.
- The experiment proves capacity on two memorized subjects only.
- A direct same-tile timing/Dice run of the original RS2 U-Net-style decoder was
  impractically slow on this CPU and was stopped; therefore this report does not
  invent a numerical claim that the query decoder matches the original decoder.

## Recommendation

The architecture is mature enough to freeze as the candidate decoder for the
next controlled stage. It has demonstrated sufficient capacity (0.98924 Dice),
uses every requested encoder scale, retains exactly one query, and stopped
benefiting from additional epochs. The next experiment may begin a true
subject-level train/validation/test split, but should first make feature
extraction memory-efficient and restore predictions to native geometry. No
additional decoder complexity is justified by this overfit study.
