# RS2-Net's original training technique — exact numbers

The frozen encoder this project builds on was trained once, by the RS2-Net paper
authors (Lin et al., NeuroImage 2024), and never re-trained here. This project's
own decoder is trained separately, with its own (much lighter) augmentation. This
note records the ORIGINAL encoder-side recipe precisely, so any comparison between
"their training" and "our training" is grounded in real numbers, not
recollection of the paper's prose.

**Source:** `RS2/utilities/dataloader.py` in the cloned `RS2-Net-Reproduction`
checkout. This is confirmed to be their actual training configuration, not a
generic template — its `initial_patch_size = (179, 217, 245)` and
`patch_size = [128, 128, 160]` match the paper's stated numbers exactly, and the
file has their own lab server path (`/icislab/volume1/lyk/Datasets/...`) hard-coded
into it. The full training loop itself (optimizer step, epoch count) lives in a
separate repo (`RSS-Train`, linked from the paper) not cloned locally; those
numbers below are taken from the paper's own text instead and are marked as such.

## Spatial augmentation (the part this project's space-invariance tests are about)

| Parameter | Value | Meaning |
|---|---|---|
| Rotation range, x/y/z | `(-0.5235987755982988, 0.5235987755982988)` rad = **±30°**, independently per axis | Same order of magnitude as this project's own rotation-invariance test's largest angle (30°) — our test's hardest case sits right at the edge of their training envelope, not beyond it |
| Probability rotation applied per sample | `p_rot_per_sample = 0.2` | Only **20%** of training patches are ever rotated — most training examples are seen unrotated |
| Scaling range | `(0.7, 1.4)`, same factor on all 3 axes (`independent_scale_for_each_axis=False`) | ±30-40% zoom |
| Probability scaling applied per sample | `p_scale_per_sample = 0.2` | 20% of samples |
| Elastic deformation | `do_elastic_deform=False`, `p_el_per_sample=0` | Off entirely |
| Mirror axes | `(0, 1, 2)` — all three | More aggressive than this project's own decoder training, which only ever flips one axis |
| Out-of-bounds fill | `border_cval_data=0` (zero padding) | Same convention this project's own `spatial_transform_utils.py` uses |

## Intensity / quality augmentation (not spatial, listed for completeness)

| Technique | Parameters |
|---|---|
| Gaussian noise | `p=0.1` |
| Gaussian blur | σ 0.5–1.0, `p=0.2` |
| Brightness multiply | ×0.75–1.25, `p=0.15` |
| Contrast | `p=0.15` |
| Simulated low-resolution | zoom 0.5–1.0×, `p=0.25` |
| Gamma correction | range 0.7–1.5, applied twice (once with image inverted, once not), `p=0.1` and `p=0.3` |

## Surrounding setup (from the paper's text — training-loop code not present locally)

- AdamW, learning rate `1e-4`, weight decay `1e-5`, batch size `2`
- Loss: Squared Dice Loss, `smooth_dr=1e-6`
- LR schedule: 50-epoch linear warmup, then cosine annealing
- 5-fold cross-validation on 80% of the data; remaining 20% held out as the final test set
- `oversample_foreground_percent = 0.33` — a third of sampled patches forced to center on brain tissue
- Label set at training time: `{background: 0, brain: 1}` — binary, brain only. The
  lesion task does not exist anywhere in the encoder's training history.

## The comparison this project actually needed

| | Rotation range | Fraction of samples rotated | Tasks covered |
|---|---|---|---|
| **Encoder** (RS2-Net paper, frozen, never touched by this project) | ±30°, independent per axis | 20% | brain only |
| **This project's own decoder** (`train_two_task_joint.py`, `augment_lesion_image`) | none | 0% | brain + lesion |

This explains the brain-vs-lesion asymmetry found in the space-invariance tests
(see `orientation_invariance/report.md`, `translation_invariance/report.md`) as
more than just "lesions are small targets": the brain query also inherits
partial, imperfect rotation-tolerance from the encoder's own original training,
while the lesion query — a task added entirely by this project, through a decoder
trained with zero rotation augmentation — has never had any exposure to pose
variation at any stage, frozen or trainable.

**Open follow-up this motivates directly:** run the same orientation/translation
invariance battery against the *original* RS2-Net model (its own decoder, not
this project's) on CAMRI + mouse brain data, to see whether their rotation-
augmented training measurably narrows the gap versus this project's own
un-augmented decoder on the same brain task. See `rs2net_baseline_invariance/`
in this folder once that experiment is run.
