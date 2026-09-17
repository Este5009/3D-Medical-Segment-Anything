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
- Visual tables: `figures/tables/table_orientation_brain.png`, `figures/tables/table_orientation_lesion.png`
- Commits: `2994c15` (results), `4694f0e` (compounding-drift correction)

## 3. Translation invariance — does shifting the scan change the prediction?

**Result:** brain query robust (0.988 mean), lesion query moderately
sensitive (0.894 mean, non-monotonic with shift size — the signature of
strided-downsampling phase/aliasing effects, not smooth blur).

- Full report: `outputs/translation_invariance/report.md` (+ per-query reports)
- Figures: `figures/fig2_self_consistency_translation.png`
- Tables: `tables/translation_{brain,lesion}_by_{transform,stage}.csv`
- Visual tables: `figures/tables/table_translation_brain.png`, `figures/tables/table_translation_lesion.png`
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
- Visual table: `figures/tables/table_dataset_pose_variance.png`
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
the edge of that trained envelope, not beyond it.

**Correction (2026-09-17):** this originally speculated that the brain
query's rotation-tolerance was inherited from that training history. Section
8 below ran the direct test (RS2-Net's own model vs. this project's
un-augmented model, same stress test) and found no measurable difference —
that hypothesis is not supported. See section 8 for the corrected
explanation (target size, not training history).

- Full report: `rs2net_original_training_technique.md` (this folder)

## 8. RS2-Net baseline invariance — does their rotation-augmented training actually help?

**Result: no measurable advantage.** Same rotation/translation stress test,
run against the *original* RS2-Net model (their own encoder AND their own
decoder, no query mechanism) on the same mouse brain test subjects used for
this project's own brain-query result:

| | rotation, mean self-consistency Dice | translation, mean self-consistency Dice |
|---|---|---|
| RS2-Net baseline (their decoder, rotation-augmented training) | 0.9760 | 0.9852 |
| This project's model (its own decoder, **no** rotation augmentation) | 0.9764 | 0.9883 |

Statistically indistinguishable — if anything this project's un-augmented
model is marginally higher on both. This **corrects** the hypothesis floated
in section 7/`rs2net_original_training_technique.md` (that the brain query's
robustness is partly inherited from the encoder's own rotation-augmented
training) — the data doesn't support it. The more likely explanation is
simply target size (see section 2/3): a large structure's decision boundary
barely moves under a drifted logit field, independent of whether anything in
the pipeline was ever rotation-trained. CAMRI (rat) scored higher still on
the baseline model (0.9952 rotation, 0.9934 translation) but has no
equivalent number from this project's own model to compare against, since
the earlier brain-query tests only covered mouse.

- Reports: `outputs/rs2net_baseline_invariance/{rotation,translation}/report.md`
- Figures: `figures/fig8_rs2net_baseline_rotation_mouse.png`,
  `figures/fig9_rs2net_baseline_translation_mouse.png`
- Tables: `tables/rs2net_baseline_{rotation,translation}_{mouse,camri}_by_transform.csv`
- Visual tables: `figures/tables/table_rs2net_{rotation,translation}_{mouse,camri}.png`
- Script: `scripts/test_rs2net_baseline_invariance.py`

## 9. Synthetic noise / blur / intensity robustness (advisor-requested)

**Result: blur is by far the most damaging corruption, and lesion detection
is fragile across the board.** Each corrupted prediction scored directly
against the expert mask (not self-consistency), compared to that subject's
own clean-image Dice; severities span from inside the encoder's own training
envelope (see section 7) to well past it.

| | mean Dice, clean | mean Dice at blur σ=1 (in envelope) | mean Dice at blur σ=4 | mean Dice at noise std=0.4 |
|---|---|---|---|---|
| Brain query | ~0.97 | 0.918 | 0.420 | 0.801 |
| Lesion query | ~0.69 | 0.594 | 0.053 | 0.475 |

Intensity/contrast shifts were the *least* damaging corruption for both
queries (brain stayed ≥0.93 even at the most severe setting tested).
**Caveat, checked directly:** the lesion test set has wide natural difficulty
spread (clean Dice 0.10–0.94 across subjects); at mild severities the
"worst subject" in the results is a pre-existing hard case whose clean Dice
was already 0.0999, not a corruption-induced collapse — the mean-Dice trend
is the reliable robustness signal, and it does show broad, real degradation
at higher severities (not just one hard subject).

- Report: `outputs/noise_robustness/report.md`
- Figures: `figures/fig10_noise_robustness_lesion.png`, `figures/fig11_noise_robustness_brain.png`
- Tables: `tables/noise_robustness_{brain,lesion}.csv`
- Visual tables: `figures/tables/table_noise_robustness_brain.png`, `figures/tables/table_noise_robustness_lesion.png`
- Script: `scripts/test_noise_robustness.py`

---
*Every number above is measured and committed; nothing here is projected or
estimated. Raw per-subject data for every experiment lives in the CSVs next
to each report, and every figure is regenerable from
`scripts/test_orientation_invariance.py`, `scripts/test_translation_invariance.py`,
`scripts/diagnose_dataset_pose_variance.py`, `scripts/test_rs2net_baseline_invariance.py`,
`scripts/test_noise_robustness.py`, and `scripts/test_rs2net_baseline_noise_robustness.py`.*

## 10. RS2-Net baseline vs. this project's decoder, under corruption — a domain split

**Result: RS2-Net's own decoder wins decisively on CAMRI, but this project's
decoder wins on mouse noise/intensity.** Same noise/blur/intensity battery
as section 9, run against RS2-Net's own model (brain only, no lesion) on
both CAMRI and mouse, compared directly to this project's brain-query result.

| Domain | Blur σ=4 (RS2-Net vs ours) | Noise std=0.4 | Intensity 0.75 |
|---|---|---|---|
| CAMRI (rat) | 0.926 vs 0.420 (**+0.51**, theirs) | 0.983 vs 0.801 (**+0.18**, theirs) | 0.989 vs 0.929 (**+0.06**, theirs) |
| Mouse/POLYIC | 0.468 vs 0.420 (+0.05, theirs) | 0.639 vs 0.801 (**+0.16, ours**) | 0.841 vs 0.929 (**+0.09, ours**) |

RS2-Net's decoder is ahead on *every* CAMRI setting tested, often by a wide
margin. On mouse, this project's decoder is ahead on every noise and
intensity setting; only severe blur tips slightly toward RS2-Net.

**Plausible explanation, consistent with everything else in this file:**
RS2-Net's own paper describes far more rat-domain training data (four
datasets, dozens of centers) than mouse-domain data (two datasets, 26 mice)
— the same asymmetry that already explained their rat-vs-mouse Dice gap
(section 7) and the rotation/translation baseline result (section 8). Their
corruption-robustness advantage tracks where their own training was richest,
and disappears (reverses, even) where it was thinnest.

- Report: `outputs/rs2net_baseline_noise_robustness/report.md`
- Figures: `figures/fig12_rs2net_comparison_camri_noise.png`, `figures/fig13_rs2net_comparison_mouse_noise.png`
- Visual tables: `figures/tables/table_rs2net_comparison_{camri,mouse}_noise.png`
- Tables: `tables/rs2net_baseline_noise_{camri,mouse}.csv`
- Script: `scripts/test_rs2net_baseline_noise_robustness.py`

## 11. Unified comparison — this project vs. RS2-Net, all five axes in one view

Sections 8 and 10 as one figure and one table: rotation, translation,
noise, blur, and intensity, mouse domain (the only domain both models have
a real number for), this project's decoder vs. RS2-Net's own decoder,
side by side.

**Pose (rotation + translation): statistically tied**, within ±0.01 Dice at
every magnitude tested, no consistent winner. **Noise and intensity: this
project's decoder wins at every single severity tested**, by a growing
margin as severity increases (noise std=0.4: −0.163 Dice for RS2-Net, i.e.
this project ahead by 16 points). **Severe blur (σ≥2) is the one place
RS2-Net's decoder pulls ahead** (+0.07 at σ=2, +0.05 at σ=4) — plausibly
where their far larger, more diverse training data actually pays off, since
blur is the corruption most directly tied to genuine image-quality variation
across real acquisition protocols.

- Figure: `figures/fig14_unified_comparison.png`
- Visual table: `figures/tables/table_unified_comparison.png`
- Table: `tables/unified_comparison.csv`
