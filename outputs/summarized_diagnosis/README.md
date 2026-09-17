# Summarized diagnosis — model robustness investigation

Running index of every experiment, inspection, and diagnostic report produced
while investigating the dual-query model's space-invariance ("is it robust to
rotation/translation?") and, going forward, its robustness to image noise/blur
and how it compares to the original RS2-Net model's own training. Each entry
below is a real, measured result with its own figures/tables/report; this file
just points to them so the whole investigation reads as one story instead of
scattered commits.

## 1. Architecture (background)

`TwoTaskLevel0OneQueryMaskDecoder`: a frozen RS2-Net Swin-based encoder (5
feature levels, `level0` full-res 48ch → `level4` 1/16-res 384ch) feeding a
trainable decoder "canvas" trunk that upsamples coarse-to-fine, cross-attended
by one of two independent query vectors (`query_brain` / `query_lesion`)
sharing every other weight. See the project README for the full picture.

## 2. Orientation invariance — does rotating the scan change the prediction?

**Result:** brain query robust (0.976 mean self-consistency Dice across
5–30°), lesion query sensitive (0.799 mean, down to 0.587 at 30°). Drift
enters almost entirely in the frozen encoder and is then compounded (not just
carried) by the decoder's skip-connection fusion; the query vector itself
stays ≥0.999 cosine-stable throughout.

- Full report: `outputs/orientation_invariance/report.md` (+ per-query
  `brain/report.md`, `lesion/report.md`)
- Figures: `figures/fig1_self_consistency_rotation.png`,
  `figures/fig3_stage_drift_bars.png`, `figures/fig4_stage_drift_table.png`
- Tables: `tables/orientation_{brain,lesion}_by_{transform,stage}.csv`
- Commits: `2994c15` (results), `4694f0e` (compounding-drift correction)

## 3. Translation invariance — does shifting the scan change the prediction?

**Result:** brain query robust (0.988 mean), lesion query moderately
sensitive (0.894 mean, non-monotonic with shift size — the signature of
strided-downsampling phase/aliasing effects, not smooth blur).

- Full report: `outputs/translation_invariance/report.md` (+ per-query reports)
- Figures: `figures/fig2_self_consistency_translation.png`
- Tables: `tables/translation_{brain,lesion}_by_{transform,stage}.csv`
- Commits: `bb52b64` (results), `177e0e2` (compounding-drift correction)

## 4. Query-embedding stability

**Result:** the one part of the architecture this project actually designed
(the cross-attention query-conditioning mechanism) is nearly pose-invariant
by itself — cosine similarity to its own baseline stays ≥0.9993 (rotation) /
≥0.9999 (translation) for both queries, every axis, every magnitude tested.
The weak points are the frozen off-the-shelf encoder and the decoder's naive
feature fusion, not the query mechanism.

- Figure: `figures/fig6_query_stability.png`

## 5. Dataset pose-variance diagnostic — is the rotation test realistic?

**Result:** yes, at least for the mouse/POLYIC brain data — real subjects
vary in pose by up to ~77° within the dataset (sd ≈10°), a well-conditioned
measurement (whole-brain shape, healthy eigenvalue separation). CAMRI's
equivalent number is inconclusive with this method (rat brain shape is too
close to rotationally symmetric for single-axis PCA to be stable — confirmed
via elongation-ratio check, median 1.6 vs mouse's 1.9). The stroke lesion
number reflects lesion-shape irregularity, not head pose, and should not be
read as a pose measurement. Translation offsets are real but small and
consistent across all three datasets (2–4mm typical).

- Figure: `figures/fig7_dataset_pose_variance.png`
- Tables: `tables/dataset_pose_variance_summary.csv`,
  `tables/dataset_pose_variance_per_subject.csv`
- Script: `scripts/diagnose_dataset_pose_variance.py` (runs on CPU, no pod needed)

## 6. How the current model is actually trained

**Result:** the frozen encoder was never trained by this project — it's
RS2-Net's own pretrained weights. This project's decoder training
(`scripts/train_two_task_joint.py`) uses only a 50% left-right mirror flip
and light intensity jitter; **no rotation or translation augmentation at
all**, for either the brain or lesion query.

- Report: (this file's companion) `rs2net_original_training_technique.md`

## 7. RS2-Net's own original training technique — exact numbers

**Result:** the encoder WAS trained with rotation augmentation — but only
±30° per axis, applied to just 20% of samples, and only ever on the brain
task (never lesions). Our rotation test's hardest case (30°) sits right at
the edge of that trained envelope, not beyond it. This explains the brain
query's partial rotation-tolerance as inherited from the encoder's own
history, not just "brains are a bigger target" — and explains why the
lesion query has no such advantage at any stage of training.

- Full report: `rs2net_original_training_technique.md` (this folder)

## 8. In progress

- **RS2-Net baseline invariance** — same orientation/translation battery,
  run against the *original* RS2-Net model (its own decoder, CAMRI + mouse
  brain only, no lesion) to see whether their rotation-augmented training
  measurably narrows the gap. Needs the GPU pod. → `rs2net_baseline_invariance/`
  once run.
- **Noise/blur/intensity robustness** — synthetic image corruption (random
  Gaussian noise, blur, intensity/contrast shifts) applied to the current
  two-task model's test inputs, to establish its operating limits under
  degraded image quality (advisor-requested). Needs the GPU pod. →
  `noise_robustness/` once run.

---
*Every number above is measured and committed; nothing here is projected or
estimated. Raw per-subject data for every experiment lives in the CSVs next
to each report, and every figure is regenerable from
`scripts/test_orientation_invariance.py`, `scripts/test_translation_invariance.py`,
`scripts/diagnose_dataset_pose_variance.py`, and (once added)
`scripts/test_rs2net_baseline_invariance.py` / `scripts/test_noise_robustness.py`.*
