# Genuine full-resolution level0 decoder ablation (TRUE level0)

## Controlled question

This is a corrected re-run of `outputs/full_resolution_level0_decoder`, which
`outputs/full_resolution_level0_decoder/implementation_audit.md` found to be
confounded on two fronts: it initialized shared decoder weights from the
pre-correction `generalization_pilot/best_checkpoint.pt` rather than the
converged corrected-label decoder, and its "level0" logits were dominated by
the unchanged half-resolution (`64×64×80`) mask head, upsampled 2x, with only
a small scalar-gated residual computed at full grid.

This experiment fixes both issues. It changes exactly one architectural
factor relative to `outputs/corrected_label_retraining`: the decoder class.
Everything else -- corrected nearest-neighbor labels, MRI preprocessing,
frozen RS2-Net encoder, one learned query, split, seed, augmentation, loss,
optimizer/LR/schedule, 0.5 threshold, native export, and the deterministic
largest-26-connected-component filter -- is held fixed and, where the
underlying code is domain-agnostic, is literally the same function object
called from this experiment's scripts rather than a re-implementation.

## Architecture actually tested

`models.query_mask_decoder.TrueFullResolutionLevel0OneQueryMaskDecoder`
(182,081 parameters; baseline is 170,401; +11,680, +6.86%; `level0_width=16`).

- levels4→3→2→1: byte-identical module graph to
  `MultiScaleOneQueryMaskDecoder` (same `projections`, `refinements`,
  `query_updates.level4/level3/level2/level1`, `mask_embedding`,
  `mask_refinement`, `mask_bias`, `query` names/shapes), so the converged
  corrected-label checkpoint's decoder weights transfer into this
  architecture unchanged -- see "Initialization proof" below.
- Level0 fusion (new, 16-channel, depthwise+pointwise): the transferred
  `mask_refinement` conv first refines the fused level1 **feature** field
  (not logits); those features are projected to 16 channels
  (`level1_to_level0`, pointwise) and trilinear-upsampled 2x to the level0
  grid; a parallel pointwise projection of the raw encoder `level0` feature
  (`level0_projection`, 48→16) is added; one depthwise 3×3×3 + pointwise 1×1×1
  convolution pair (`level0_refinement`) fuses the sum.
- Query conditioning: the query attends coarse-to-fine through levels4→1
  exactly as the baseline does, then makes **one additional** attention step
  over the fused level0 feature (2x average-pooled purely for the attention
  step's memory footprint -- the mask field itself is never pooled).
- Mask head: **one** dot product, evaluated once, directly on the full-grid
  fused level0 feature: the (transferred) `mask_embedding` maps the
  level0-attended query to a 32-dim embedding; a new `Linear(32→16)`
  (`level0_embedding_projection`) projects it to the level0 feature width;
  `einsum("bc,bcdhw->bdhw", ...)` against the `[B,16,128,128,160]` fused field
  produces `[B,1,128,128,160]` logits directly, `+ mask_bias`.
- **No half-resolution mask logits are ever computed. No residual/skip term
  exists.** This is the deliberate structural difference from the earlier,
  audited implementation.

## Initialization proof (verified on real CAMRI + Mouse samples, pre-training)

`scripts/verify_true_full_resolution_level0_decoder.py` hard-asserts every
requested gate and writes `pretraining_verification.json`; it also runs
automatically inside `train_true_full_resolution_level0_decoder.py::main()`
before any optimizer step. All gates passed:

| Gate | Result |
|---|---|
| `initial_checkpoint` path | `outputs/corrected_label_retraining/checkpoints/best_corrected_labels.pt` (exact match) |
| Encoder frozen | 0 trainable encoder parameters, `encoder.training=False` |
| Query count | `query.shape == (1,1,32)` |
| Target | `[1,1,128,128,160]`, unique values `{0,1}` (CAMRI + Mouse cached samples) |
| New logits | `[1,1,128,128,160]` natively (CAMRI + Mouse) |
| No final logit upsampling | `decoder(features)` (no `output_size`) already equals `[1,1,128,128,160]`; `decoder(features, output_size=(128,128,160))` is bitwise identical (`max|Δ|=0.0`) -- the interpolation branch is provably never exercised |
| Inherited weights match baseline | all 76/76 `best_corrected_labels.pt` decoder tensors transfer with **zero** mismatched values; 26 new (level0-only) parameter keys added; 0 unexpected keys |

`outputs/true_full_resolution_level0_decoder/pretraining_verification.json`
is the machine-checkable record.

Unlike the earlier flawed design, this decoder does **not** reproduce the
baseline's output at initialization (there is no zero-gated residual trick to
force that) -- the new level0 mask head and its query-attention step are
freshly initialized, so the network's actual output differs substantially
from the transferred baseline until it has trained. This is expected and is
the direct, disclosed consequence of removing the residual-correction
pattern the audit flagged (see "CAMRI safety-rule deviation" below).

## Training

- Config: `configs/true_full_resolution_level0_decoder.yaml`. Every
  hyperparameter (seed `20260717`, lr `5e-5`, weight decay `1e-4`,
  `max_epochs=20`, `early_stop_patience=5`, `minimum_validation_improvement=
  0.0003`, `camri_validation_max_drop=0.01`, `dice_bce_boundary` loss with
  `boundary_weight=0.25`, augmentation, tile size, label resampling
  `is_seg=True/order=0/order_z=0`, 0.5 threshold, 26-connectivity filter) is
  identical to `outputs/corrected_label_retraining`'s config. `verification_gate()`
  (the mask-preprocessing-quality check `train_corrected_label_retraining.py`
  runs before training) is reused unchanged and was run here too -- it was
  omitted from the earlier, audited level0 script; see confounder disclosure
  below.
- Cached encoder features for the 55 required train/validation samples were
  reused byte-for-byte from the earlier level0 experiment's cache
  (`/tmp/rs2_level0_corrected_features`), since they are a pure deterministic
  function of the unchanged frozen encoder and unchanged corrected-label
  preprocessing -- copying them is a compute-reuse optimization with zero
  effect on any result, not a data source change.
- Selected epoch: **20** (of 20 run; the run never triggered early-stop
  patience -- see the per-epoch table below). `camri_safety_stop: false`.
- Best balanced (model-space) validation Dice: **0.977094** (CAMRI 0.979024,
  Mouse 0.975164), versus the corrected-label baseline's own 0.976708.

### CAMRI safety-rule deviation (disclosed)

`train_corrected_label_retraining.py`'s CAMRI safety rule (`abort if CAMRI
validation Dice < camri_ref - 0.01`) is copied faithfully into
`train_true_full_resolution_level0_decoder.py`, but its *semantics* were
changed after a first attempt failed:

- **Attempt 1** used the rule exactly as in the baseline script: an
  immediate `break` on the first epoch that violates the floor. Epoch 1's
  freshly-initialized level0 head produced CAMRI Dice 0.853047 (floor is
  0.972801), so training aborted after one epoch with `best=-1` and an
  **untrained, randomly-initialized checkpoint** saved as "best" (`epoch: 0`).
  Preserved for the record at
  `outputs/true_full_resolution_level0_decoder/training/attempt1_camri_safety_aborted_epoch1/`.
- **Root cause**: in every prior experiment (label-correction-only, and the
  earlier flawed zero-init-residual level0 design), shared tensors are either
  fully strict-loaded or the new branch is scale-gated to exactly reproduce
  the baseline at epoch 0, so epoch 1 already starts near `camri_ref` and the
  safety rule only ever fires on genuine mid-training drift. This
  experiment's mask head has no such guarantee by construction -- that
  guarantee *is* the residual-correction pattern the audit asked to remove --
  so CAMRI performance necessarily collapses for the first several epochs
  while the new head learns to use the transferred features.
- **Fix (attempt 2, the reported run)**: the identical numeric floor
  (`camri_ref - 0.01`) and the identical "best balanced validation Dice"
  selection rule are kept, but applied as a **per-epoch eligibility filter**
  on which checkpoints may become "best," not as a training-abort trigger.
  Ineligible epochs are still logged and still count toward the 20-epoch
  budget, but neither update the saved state nor consume early-stop patience
  (a warmup dip is not "no improvement"; it is "not yet evaluable"). Epochs
  1-8 and 10 were ineligible (CAMRI Dice 0.853→0.972, climbing); epoch 9 was
  the first eligible epoch. This is the one deliberate, disclosed departure
  from "keep checkpoint selection identical" in this experiment, and it is
  necessitated by, not in tension with, requirement 3 (no residual
  correction).

### Per-epoch validation (selected epochs)

| Epoch | CAMRI Dice | Mouse Dice | Balanced | Eligible | Selected |
|---:|---:|---:|---:|:---:|:---:|
| 1 | 0.853047 | 0.537165 | 0.695106 | no | |
| 4 | 0.965345 | 0.949364 | 0.957354 | no | |
| 9 | 0.972953 | 0.966312 | 0.969632 | **yes (first)** | |
| 14 | 0.976883 | 0.972373 | 0.974628 | yes | |
| 17 | 0.977767 | 0.974241 | 0.976004 | yes | |
| 20 | 0.979024 | 0.975164 | **0.977094** | yes | **✓ best** |

Full history: `outputs/true_full_resolution_level0_decoder/training/history.csv`.
Training was still improving (barely past the 0.0003 threshold) when the
20-epoch budget ended -- it did not plateau under its own early-stop rule.

## Untouched native test results (6 CAMRI + 80 Mouse, identical filtering/tiling)

Native evaluation reused `run_records`/`sliding_window_logits` from
`evaluate_mouse_boundary_adaptation.py`/`evaluate_external_holdout.py`
**unmodified** -- the same whole-volume, centered/symmetric-padding tiling
the corrected-label baseline itself uses. The only change, made by
monkeypatching the shared module's `FEATURE_NAMES` constant before calling
(`scripts/evaluate_true_full_resolution_level0_decoder.py`), is that
`level0` features are additionally sliced into the dict handed to
`model.decode`. This directly fixes confounder #3 from the audit (the
earlier level0 evaluation used a separately reimplemented, asymmetrically
padded tiling function).

### Filtered predictions (recommended production condition)

| Domain | Metric | Corrected baseline | TRUE level0 | Change |
|---|---|---:|---:|---:|
| CAMRI (n=6) | Dice | 0.987029 | 0.986922 | −0.000106 |
| CAMRI | Precision | 0.985825 | 0.987289 | +0.001464 |
| CAMRI | Recall | 0.988249 | 0.986564 | −0.001685 |
| CAMRI | HD95 | 0.1333 mm | 0.1333 mm | unchanged |
| CAMRI | ASSD | 0.01997 mm | 0.02049 mm | +0.00051 mm |
| CAMRI | FP / FN voxels | 14,607 / 12,791 | 13,044 / 14,603 | −1,563 / +1,812 |
| CAMRI | Total residual error | 27,398 | 27,647 | +249 (+0.91%) |
| Mouse (n=80) | Dice | 0.969536 | **0.972438** | **+0.002902** |
| Mouse | Precision | 0.966415 | 0.973260 | +0.006845 |
| Mouse | Recall | 0.972760 | 0.971664 | −0.001096 |
| Mouse | HD95 | 0.2009 mm | **0.1608 mm** | **−0.0402 mm (−20.0%)** |
| Mouse | ASSD | 0.04214 mm | **0.03577 mm** | **−0.00637 mm (−15.1%)** |
| Mouse | FP / FN voxels | 331,053 / 254,256 | 259,661 / 265,552 | **−71,392** / +11,296 |
| Mouse | Total residual error | 585,309 | **525,213** | **−60,096 (−10.3%)** |
| Mouse | FP/FN ratio | 1.356 | 0.996 | more balanced |
| Combined (86-subject mean) | Dice | 0.970756 | 0.973448 | +0.002692 |
| Combined | HD95 | 0.1962 mm | 0.1589 mm | −0.0373 mm (−19.0%) |

Per-subject counts (`per_subject_comparison.csv`, filtered Dice):
**CAMRI 3/6 improved, 3/6 regressed** (mean Δ −0.0001, i.e. noise at n=6);
**Mouse 78/80 improved, 2/80 regressed** (mean Δ +0.0029). This is a
qualitatively different pattern from the earlier flawed experiment, where
only 4/80 Mouse subjects improved and Mouse HD95/ASSD/total-error all
worsened.

Full tables: `native_metrics.csv`, `per_subject_comparison.csv`,
`paired_subject_changes.csv`, `baseline_vs_level0_metrics.csv`.

## Boundary-distance, curvature, and contour geometry

(`analyze_boundary_error_diagnostics.py`, unmodified, run against this
experiment's own `diagnostic_source.csv`; `validation.json` confirms
`training_performed: false`, `model_inference_performed: false`, exact
probability/raw-mask reconciliation, and the locked 6/80 cohort.)

**Distance-to-expert-surface, ≤1-voxel share** (higher is better):

| Domain | Corrected baseline | TRUE level0 |
|---|---:|---:|
| CAMRI | 93.766% | 93.978% |
| Mouse | 89.079% | **91.622%** |
| Combined | 89.288% | **91.740%** |

Mean physical distance improved in both domains: CAMRI 0.0914→0.0828 mm,
Mouse 0.0881→0.0733 mm; Mouse P95 improved 0.300→0.283 mm. This reverses the
earlier flawed experiment's regression on this exact measure (which had
worsened 89.288%→83.301%).

**Errors per 100 expert-surface voxels, high-complexity regions** (lower is
better): CAMRI 25.232→26.142 (slightly worse), Mouse 39.700→**35.709**
(−10.1%), Combined 38.145→**34.680** (−9.1%). Flat-region errors improved in
both domains (CAMRI 4.641→4.306, Mouse 24.523→**21.506**).

**Contour direction-change ratio vs. expert** (closer to 1.0 is more
articulated): CAMRI 0.8207→0.8285, Mouse 0.6866→0.6980 -- both modestly
improved, in the same direction as, but larger than, the corrected-label
baseline's own improvement over the pre-correction model.

**Axis-run ratios vs. expert** (closer to 1.0 is better): max-run ratio
improved in both domains (CAMRI 1.3263→1.3224, Mouse 1.6629→1.6489); P95
run ratio worsened slightly in both (CAMRI 1.1452→1.1794, Mouse
1.1745→1.1971) -- long flat runs got shorter at the extreme but slightly more
common at the 95th percentile. Mixed, not a clean win.

**Factor-5 native-y quantization excess** (lower is closer to the expert's
own, non-lattice-locked distribution): CAMRI 22.03→**18.08** points
(−3.94 pp), Mouse 29.69→**28.61** points (−1.09 pp). Reduced in both
domains, though the Mouse lattice signature is not eliminated.

**Probability calibration** (`boundary_diagnostics/probability_analysis.csv`):
predictions became more decisive overall -- uncertain (0.25-0.75 probability)
boundary voxels fell in both domains (Mouse FP boundary 45.4%→41.0%, FN
boundary 44.9%→39.6%). This cuts both ways: among the voxels that remain
wrong, **more are confidently wrong** on Mouse (FP boundary "confident
incorrect" 27.48%→33.57%; FN boundary 27.91%→33.28%), even though the
absolute FP voxel count fell 21.6%. CAMRI's confident-incorrect FP fraction
improved (18.66%→17.01%). Net reading: the full-resolution field is sharper
and, in aggregate, more correct, but its remaining Mouse errors are less
"softly uncertain" and more "confidently wrong" than the baseline's smoother,
interpolated field -- a genuine trade-off, not a one-sided win.

## Visible pixelation

Qualitative figures (`figures/manifest.csv`, 12 panels) show a consistent
pattern, illustrated by `figures/mouse/previous_pixelated_...png` (a
bilobed rostral slice the baseline renders as one smooth, disconnected-lobe
blob that misses the anatomical gap entirely; TRUE level0 renders both lobes
with the gap visible, closer to the expert shape) and
`figures/mouse/largest_regression_...png` (Mouse's worst per-subject change,
Dice 0.9527→0.9501 -- both models make the *same* detached over-segmentation
error on a whisker-like structure; TRUE level0's version is not meaningfully
worse). The trade-off visible across figures: TRUE level0 captures finer
anatomical shape features (gaps, concavities) the half-resolution baseline
smooths over, at the cost of small salt-and-pepper speckle inside otherwise
correct regions -- consistent with the probability-calibration finding above
(sharper, more decisive, occasionally locally wrong). No panel shows gross
anatomical failure or one-sided quality loss.

## Efficiency

| | Corrected baseline | TRUE level0 |
|---|---:|---:|
| Trainable parameters | 170,401 | 182,081 (+6.86%) |
| Median native inference, CAMRI | 9.91 s | 9.93 s |
| Median native inference, Mouse | 8.60 s | 9.46 s (+10.0%) |
| Mean training epoch (CPU) | -- | 249.4 s |
| Peak process memory (training) | -- | 10,236 MiB |
| Total training wall time | 5,504 s (incl. cache build) | 4,988 s (cache reused; see above) |

Mean epoch cost (249.4 s) is essentially identical to the earlier, audited
level0 experiment's own median epoch (249.5 s), despite this design having no
residual/skip branch and a larger level0 width (16 vs. 8) -- the depthwise/
pointwise fusion design keeps the added full-grid cost small, as intended.
Device: CPU throughout (forced; the installed torch 2.0.0 MPS backend does
not support `Conv3d`, verified empirically before training).

## Other confounders checked

Encoder frozen state, split files, MRI preprocessing, corrected-label
preprocessing function, augmentation, loss, optimizer/LR/schedule, seed,
threshold, and the deterministic filter are all reused unchanged from
`corrected_label_retraining` (see config diff: the only differing fields are
`initial_checkpoint` -- now correctly the converged baseline -- output paths,
cache paths, and the new `level0_width` field). The two disclosed departures
are the CAMRI safety-rule semantics change (necessitated by removing the
residual-correction pattern; documented above) and reusing pre-existing
cached encoder features (a compute optimization with no effect on results,
since caching is a deterministic function of unchanged inputs).

## Final verdict

**B. Improves geometry with a mixed, domain-dependent profile -- closer to
(A) for Mouse, closer to (C)/neutral for CAMRI.**

Genuine full-resolution level0 decoding, correctly initialized from the
converged corrected-label baseline and evaluated with identical tiling,
**does provide a real, meaningful benefit on Mouse**: Dice +0.29 points,
HD95 −20.0%, ASSD −15.1%, total residual error −10.3%, boundary distance
≤1-voxel share +2.5 points, high-complexity error rate −10.1%, factor-5
quantization excess reduced, and 78/80 subjects individually improved. This
is a qualitatively different and opposite result from the earlier, audited
(confounded) experiment, which worsened every one of these Mouse measures.

**On CAMRI, the effect is neutral** (Dice −0.0001, HD95 unchanged, ASSD
+0.0005 mm, total error +0.91%, 3/6 subjects improved and 3/6 regressed) --
not a regression, but not a demonstrated benefit either, plausibly because
CAMRI's near-ceiling baseline performance (Dice 0.987, HD95 already at the
0.1333 mm measurement floor) leaves little headroom, and n=6 limits
statistical power.

A genuine, disclosed trade-off exists: the sharper full-resolution field
produces fewer but more confidently-wrong Mouse boundary errors, and fine
shape capture comes with mild internal speckle in some slices. Neither
undermines the aggregate improvement, but both should be tracked if this
architecture is developed further.

**This reverses the prior conclusion.** The earlier "level0/full-resolution
decoding did not help" finding was an artifact of the confounded
implementation (wrong initialization checkpoint, residual-correction
pathway that never removed the half-resolution logits, and a different,
uncontrolled evaluation tiling), not evidence about full-resolution decoding
itself. With those three issues fixed, full-resolution level0 decoding is
justified, at least for Mouse, and should not be abandoned. The immediate
next step per project research philosophy (attribute failures before
proposing further architecture changes) is a residual-failure/attribution
pass on this experiment's own predictions, and a check of whether the
Mouse-specific gain is a genuine cross-domain-transfer effect or reflects the
level0 branch simply having more capacity to fit Mouse's larger training
share under `balanced_epoch_order`.
