# Implementation audit: full-resolution level0 decoder ablation

**Type:** read-only inspection. No training, inference, checkpoint modification,
or prediction regeneration was performed. All numbers below were produced by
loading existing checkpoints/caches and running frozen `nn.Module.forward`
passes under `torch.inference_mode()`, or by reading existing files.

**Scope:** `outputs/full_resolution_level0_decoder/` versus
`outputs/corrected_label_retraining/`, per the eight-part audit request.

---

## 1. Loss/target resolution

Traced `scripts/train_full_resolution_level0_decoder.py` and
`scripts/corrected_label_preprocessing.py` directly.

- `preprocess_image_and_corrected_target()` calls `corrected_target()`, which
  resamples the expert mask with `resample_data_or_seg_to_shape(..., is_seg=True,
  order=0, order_z=0)` (categorical nearest-neighbor) and then
  `_pad_and_center_crop(tensor, tile_size)` to the fixed tile `(128,128,160)`.
- The cache written by `prepare_cache()` stores this tensor verbatim as
  `target` (`uint8`), plus a `label_interpolation: "nearest/order0/is_seg=True"`
  marker.
- Directly inspected two real cached samples used by this experiment:
  - CAMRI (`camri_validation_035.pt`): target shape `(1,1,128,128,160)`,
    unique values `{0,1}`.
  - Mouse (`mouse_validation_POLYIC_20190510_mouse31__E2_P1.pt`): target shape
    `(1,1,128,128,160)`, unique values `{0,1}`.
- Training loop: `logits = decoder(features, output_size=target.shape[-3:])`,
  then `loss, parts = training_loss(logits, target, config)`
  (`scripts/train_mixed_domain_decoder.py:training_loss`, unchanged). No
  downsampling of `target` occurs anywhere in the loop, and
  `FullResolutionLevel0OneQueryMaskDecoder.forward` always internally produces
  logits already at `level0.shape[-3:] == (128,128,160)`, so the
  `output_size` interpolation branch at the end of `forward` (`if output_size
  is not None and shape mismatch: F.interpolate(...)`) is never triggered —
  shapes already match.

**Answer:** target was `[1,1,128,128,160]`, never reduced to `64×64×80` for
loss, and no target interpolation occurred in the training loop. Loss was
genuinely computed between full-resolution logits and a full-resolution
corrected categorical target.

---

## 2. Successful-run initialization — CONFOUND CONFIRMED

`configs/full_resolution_level0_decoder.yaml` contains **two** checkpoint
fields:

```
"initial_checkpoint": "outputs/generalization_pilot/best_checkpoint.pt",
"corrected_baseline_checkpoint": "outputs/corrected_label_retraining/checkpoints/best_corrected_labels.pt",
```

`grep -rn "corrected_baseline_checkpoint"` across `scripts/`, `configs/`, and
`models/` returns **only the config file itself** — this key is never read by
any script. `scripts/train_full_resolution_level0_decoder.py` calls
`initialize(decoder, ROOT/config["initial_checkpoint"])` — i.e., it loads
`outputs/generalization_pilot/best_checkpoint.pt`, not the corrected-label
checkpoint.

This is confirmed independently by the checkpoint metadata itself
(`torch.load` inspection, no modification):

| Checkpoint | `epoch` | `balanced_validation_dice` | `initial_checkpoint` field |
|---|---:|---:|---|
| `generalization_pilot/best_checkpoint.pt` | 14 | n/a (pre-dates this field) | — |
| `mixed_domain_anatomical_training/checkpoints/best_mixed_domain.pt` | 17 | 0.976576 | `generalization_pilot/best_checkpoint.pt` |
| `corrected_label_retraining/checkpoints/best_corrected_labels.pt` | 17 | 0.976708 | `generalization_pilot/best_checkpoint.pt` |
| `full_resolution_level0_decoder/checkpoints/best_level0_decoder.pt` | 20 | 0.978866 | `generalization_pilot/best_checkpoint.pt` |

**ACTUAL SUCCESSFUL-RUN INITIALIZATION:**
`outputs/generalization_pilot/best_checkpoint.pt` (epoch 14, the original
CAMRI-only "generalization pilot" decoder — predates both mixed-domain
training and the label correction).

This is not an isolated diagnostic/debug artifact: it is baked into the
saved `best_level0_decoder.pt`'s own `initial_checkpoint` field, so it is the
checkpoint that was genuinely used for the reported, selected run.

**Is this "wrong," though?** Partially, and the nuance matters (see §3): the
*same* `generalization_pilot/best_checkpoint.pt` is also the documented
`initial_checkpoint` for `corrected_label_retraining` itself and for
`mixed_domain_anatomical_training`. All three experiments are independent
fine-tunes from the same epoch-14 root, not a chain. So the level0 run's
init checkpoint is *identical* to the corrected-label run's init checkpoint —
it is **not** initialized from mismatched or unrelated weights. What it is
**not** is initialized from `best_corrected_labels.pt` (the converged,
already label-corrected decoder), despite `config.json` carrying a
`corrected_baseline_checkpoint` field that implies that was the intent, and
despite the report's own text ("the corrected-label one-query decoder was
extended... to level0") describing continuation from the corrected-label
decoder specifically. That implied experimental design was not implemented.

`tests/test_full_resolution_level0_decoder.py::test_checkpoint_and_corrected_label_configuration`
only asserts `label_resampling` config and `epoch==20`; it never asserts
anything about which checkpoint initialized the run, so this gap was not
caught by the test suite despite its name.

---

## 3. Initial baseline equivalence — numerically verified

No training. Reconstructed, without modification:

- (a) `MultiScaleOneQueryMaskDecoder` loaded strict from
  `generalization_pilot/best_checkpoint.pt`.
- (b) `MultiScaleOneQueryMaskDecoder` loaded strict from
  `corrected_label_retraining/checkpoints/best_corrected_labels.pt` (the
  "intended" baseline per the audit brief).
- (c) `FullResolutionLevel0OneQueryMaskDecoder`, initialized exactly as
  `scripts/train_full_resolution_level0_decoder.py::initialize()` does: all
  76 shape-matching tensors transferred (non-strict) from
  `generalization_pilot/best_checkpoint.pt`; the 27 new level0-only
  parameters (incl. `level0_residual_scale`) at their constructor defaults.

Ran all three on the same two real cached CAMRI/Mouse validation feature
sets used by the actual experiment:

| Comparison | CAMRI max\|Δ\| (logits) | CAMRI mean\|Δ\| | Mouse max\|Δ\| | Mouse mean\|Δ\| |
|---|---:|---:|---:|---:|
| level0-at-init vs. **generalization_pilot** decoder | **0.0** | **0.0** | **0.0** | **0.0** |
| level0-at-init vs. **corrected-label baseline** | 48.08 | 7.27 | 59.20 | 8.83 |
| generalization_pilot vs. corrected-label baseline | 48.08 | 7.27 | 59.20 | 8.83 |

(Values are in pre-sigmoid logit units; differences of this size correspond
to probability flips from ≈0 to ≈1 over large parts of the volume.)

**Answer:** the new model starts **exactly** bit-identical
(max|Δ| = 0.0) to `generalization_pilot/best_checkpoint.pt` interpolated the
old way — this is mechanically guaranteed by `level0_residual_scale = 0` at
construction (`nn.Parameter(torch.zeros(1))`), which zeroes the entire level0
residual term, and by every shared tensor transferring exactly. This is
exactly what `tests/test_full_resolution_level0_decoder.py::
test_zero_residual_initialization_exactly_preserves_baseline` checks — and it
checks it correctly, but only against a *freshly constructed*
`MultiScaleOneQueryMaskDecoder`, not against the corrected-label checkpoint.

The new model does **not** start functionally equivalent to the
corrected-label baseline. It starts equivalent to the pre-correction,
pre-mixed-domain epoch-14 decoder. The level0 training run therefore had to
relearn the label-correction adaptation that `best_corrected_labels.pt`
already had, *concurrently* with learning the new level0 residual branch,
within the same 20-epoch/lr=5e-5 budget that `corrected_label_retraining`
used only for the former.

---

## 4. Corrected labels — genuinely used

- Cache path: `/tmp/rs2_level0_corrected_features/*.pt` (config key
  `corrected_cache`), separate from `corrected_label_retraining`'s own
  `/tmp/rs2_corrected_label_features/*.pt` cache (separate because the level0
  cache additionally stores `level0` features).
- Target shape: `(1,1,128,128,160)`, dtype `uint8`.
- Target unique values: `{0, 1}` (categorical, verified above).
- Interpolation method used to generate targets: nearest-neighbor
  (`is_seg=True, order=0, order_z=0`) via
  `RS2.preprocessing.resampling.default_resampling.resample_data_or_seg_to_shape`,
  identical function/parameters to `corrected_label_retraining`.
- The old order-1/linear-interpolated, int8-truncated masks do **not** appear
  anywhere in this experiment's data path — `corrected_target()` is the only
  label-generation function called, and it raises `ValueError` if any value
  outside `{0,1}` appears.

**Answer:** corrected nearest-neighbor labels were genuinely and exclusively
used for this experiment's training and validation targets.

---

## 5. Actual level0 architecture

Read `models/query_mask_decoder.py::FullResolutionLevel0OneQueryMaskDecoder`
directly (180,466 parameters) and compared against
`MultiScaleOneQueryMaskDecoder` (170,401 parameters, the "level1 decoder
stage").

**It is not "the same decoder extended to level0."** It is the unchanged
levels4→1 decoder (bit-identical weights at init, per §3) plus a
**separate, much lower-capacity, zero-gated additive residual branch**:

- **Level0 encoder input:** `features["level0"]`, `[1,48,128,128,160]`
  (full model grid, 48 channels — same RS2 stem width as level1).
- **Projection:** `level0_projection = Conv3d(48→8, kernel=1)` — note the
  target width is **8 channels**, not the `embedding_dim=32` used by every
  other stage of this decoder (levels1–4 all project to 32).
- **Top-down/lateral fusion:** `level1_to_level0 = Conv3d(32→8, kernel=1)`
  applied to the already-fused 32-channel `level1` feature, trilinear
  2×-upsampled (`align_corners=False`) to the level0 grid, added to the
  8-channel level0 projection.
- **Depthwise convolution:** yes —
  `level0_refinement[0] = Conv3d(8→8, kernel=3, groups=8)` (depthwise).
- **Pointwise convolution:** yes — `level0_refinement[1] = Conv3d(8→8,
  kernel=1)` immediately after (plus `level0_projection` and
  `level1_to_level0`, both 1×1×1).
- **Query conditioning:** the query first passes through the **unchanged**
  levels4→3→2→1 `QueryUpdateBlock` chain (identical to the baseline decoder,
  same transferred weights) producing `baseline_query`. A **second**,
  level0-only `QueryUpdateBlock` then updates that query further using
  `level0_query_projection(avg_pool3d(fused_level0, kernel=2, stride=2))` —
  i.e., the level0 feature is average-pooled back down to `64×64×80` *only*
  for this global-query attention step (memory control); the mask branch
  itself is not pooled.
- **Residual mechanism/scale:** `level0_residual_scale =
  nn.Parameter(torch.zeros(1))`, a **single learned scalar**, initialized to
  exactly 0. Final logits = `baseline_logits + level0_residual_scale *
  residual`.
- **Mask head:** two separate dot products, not one:
  1. `baseline_logits`: `baseline_embedding (32-dim, from the query *before*
     level0 attention) · mask_refinement(fused_level1) (32-dim, 64×64×80)`
     → `[1,1,64,64,80]`, **+ `mask_bias`**, then explicitly
     `F.interpolate(..., mode="trilinear", align_corners=False)` up to
     `128×128×160`. **This is the literal, unconditional `64×64×80 →
     interpolate 2× → 128×128×160` pathway from the old decoder, executed
     unchanged inside every forward pass, not removed.**
  2. `residual`: an 8-channel full-grid voxel feature
     (`level0_refinement(level1_to_level0(baseline_voxels)↑2× + level0
     projection)`) dotted with an 8-dim query embedding
     (`level0_mask_embedding: Linear(32→8)` applied to the level0-attended
     query) → native `[1,1,128,128,160]`, no interpolation.
- **Final learned-logit shape:** `[1,1,128,128,160]`, computed as
  `interpolated_64x64x80_baseline_term + (learned scalar) * native_8-channel_128x128x160_residual_term`.

**Scientific hypothesis actually tested:** not "does the decoder predict
natively at full model resolution instead of interpolating," but "does
adding a small (8-channel, 1/4 the width of the rest of the decoder),
additive, scalar-gated full-grid residual correction on top of an otherwise
unchanged, still-interpolated coarse decoder improve boundaries." These are
materially different hypotheses; the report's own architecture-trace section
is honest about the mechanism (it calls the coarse path a "transferred
baseline logit skip... not the sole final output"), but the higher-level
framing ("extended... to level0", "native full-grid logits") reads as
testing the former when the code implements the latter.

---

## 6. Full-resolution logits — confirmed, with an important qualification

`architecture_trace.json` (`stage: "native_learned_logits"`, `shape:
[1,1,128,128,160]`, `final_interpolation: false`) and the checkpoint's own
`training/summary.json` (`native_logits_shape: [1,1,128,128,160],
final_logit_interpolation: false`) are both literally accurate: the tensor
`forward()` returns already has shape `[1,1,128,128,160]`, and no
interpolation node runs *after* that tensor is formed (the `output_size`
check in `forward` is a no-op here because `target.shape[-3:]` already equals
`(128,128,160)`).

**However**, per §5, this framing is easy to misread. There **is** a literal
`[1,1,64,64,80] → F.interpolate (2×, trilinear) → [1,1,128,128,160]` step
inside every forward pass — it produces `baseline_logits`, the dominant term
in the sum (the residual is scaled by a parameter that starts at exactly 0
and, per training history, likely remains small after only 20 epochs — see
§7). The distinction the report draws ("native learned residual, not a
resampling for native-geometry export") is technically correct but should
not be read as "the interpolation pathway was removed." It was not; a
correction was added alongside it.

This is separate from, and should not be confused with, the *second*,
unrelated resampling step that maps model-space logits back to native MRI
geometry for evaluation (`configuration.resampling_fn` in
`export_native`/`export_probability`) — that step is unchanged and applies
identically to both experiments' outputs.

---

## 7. Other confounders found

Diffed `outputs/corrected_label_retraining/config.json` against
`outputs/full_resolution_level0_decoder/config.json`, and the training/eval
scripts line by line.

**Identical (verified, not just documented):** encoder config/frozen state
(`RS2NetEncoderAdapter`, `requires_grad_(False)` on every encoder parameter
in both `prepare_cache` functions), CAMRI split, Mouse split, seed
(`20260717`), learning rate (5e-5), weight decay (1e-4), `max_epochs` (20),
`early_stop_patience` (5), `minimum_validation_improvement` (0.0003),
`camri_validation_max_drop` (0.01), loss (`dice_bce_boundary`,
`boundary_weight=0.25`, `boundary_width=1`), augmentation
(flip/scale/noise), tile size, label resampling parameters, inference
threshold (0.5), filter connectivity (26), query count (1), checkpoint
selection criterion (balanced validation Dice, same improvement/patience
logic in both `main()` implementations), and — per §2/§3 — the
`initial_checkpoint` value itself.

**Genuine differences found, beyond the intended level0 architecture:**

1. **§2's initialization gap**: shared/overlapping decoder weights are
   sourced from the pre-correction `generalization_pilot` checkpoint rather
   than from the converged corrected-label checkpoint. (Primary confound —
   see §2/§3/§8.)
2. **`verification_gate()` (mask-preprocessing correctness check) is called
   in `train_corrected_label_retraining.py::main()` before training, and is
   absent from `train_full_resolution_level0_decoder.py::main()`.** The
   level0 run relies on the same underlying `corrected_target()` function
   (§4 confirms its output is correct), so this is a process/rigor gap
   rather than a demonstrated data error, but it means the level0 run did
   not re-verify its own label cache the way the baseline run did.
3. **Native-geometry sliding-window inference uses two different tiling
   implementations.** `evaluate_corrected_label_retraining.py` uses the
   shared `sliding_window_logits` (`scripts/evaluate_external_holdout.py`),
   which pads the *entire* volume symmetrically (half before/half after) to
   at least tile size before tiling. `evaluate_full_resolution_level0_decoder.py`
   defines its own local `sliding_window_level0_logits` (justified in its
   docstring only as "changes... feature selection," i.e., adding level0),
   which instead pads each undersized *tile* with **zeros appended only on
   the high-index side** (`F.pad(tile, (0, padding[2], 0, padding[1], 0,
   padding[0]))`), not centered. `outputs/model_spatial_resolution_diagnostics`
   already establishes that several CAMRI/Mouse subjects have at least one
   preprocessed axis smaller than the `128,128,160` tile (e.g., CAMRI height
   ≈64, Mouse ≈70–72 on two axes), so this asymmetric-vs-symmetric padding
   difference is actually exercised at test time, not a theoretical edge
   case. It does not discard or duplicate any real voxel (the final crop
   recovers exactly the real spatial extent either way), but it changes the
   zero-padding context the convolutions see near volume edges for a
   nontrivial fraction of subjects, and it is a genuine, uncontrolled
   implementation difference between the two evaluation pipelines that
   was not called out in the report.
4. **Test coverage gap**: `tests/test_full_resolution_level0_decoder.py`
   verifies parameter count, native output shape, zero-residual equivalence
   against a *fresh* baseline module, and `label_resampling`/`epoch==20` —
   it does not assert anything about the initialization checkpoint's
   provenance, which is how #1 went uncaught.

---

## 8. Final verdict

### **B. MOSTLY VALID WITH QUALIFICATIONS**

**What was actually trained:** `FullResolutionLevel0OneQueryMaskDecoder`
— the unchanged levels4→1 decoder (weights transferred bit-exact from
`generalization_pilot/best_checkpoint.pt`, epoch 14) plus a new, narrow
(8-channel) additive residual branch on the level0 grid, gated by a scalar
initialized to 0 — trained end-to-end for 20 epochs on corrected
nearest-neighbor labels, with identical seed/lr/loss/optimizer/schedule/split
to the `corrected_label_retraining` run.

**Resolution actually learned against:** full-resolution, categorical,
nearest-neighbor corrected targets at `[1,1,128,128,160]` (§1, §4 — this part
of the report is accurate and verified). But the network's own dominant
output term is still produced at `[1,1,64,64,80]` and 2× trilinear-
interpolated before the (small, scalar-gated) full-grid residual is added
(§5, §6) — so supervision was full-resolution, but the *architecture* is not
a clean "native full-resolution decoder," it is "half-resolution decoder +
low-capacity full-resolution residual correction."

**Was this a fair comparison against the corrected-label baseline?**
Partially. Every hyperparameter, all data, the loss, the optimizer, the
split, the checkpoint-selection rule, and — critically — the *initialization
checkpoint value itself* (`generalization_pilot/best_checkpoint.pt`) were
identical between the `corrected_label_retraining` and
`full_resolution_level0_decoder` training runs (§7), so this is not the
"wrong, unrelated checkpoint by accident" scenario the audit brief worried
about, and it is a legitimate, controlled comparison of *"does adding this
residual branch help, when both models are fine-tuned from the same
epoch-14 root under identical conditions."*

It is **not**, however, a fair test of the framing stated in the report and
in `MEMORY.md` — "the corrected-label decoder was extended... to level0,"
implying continuation from the *converged*, already label-corrected decoder
(`best_corrected_labels.pt`). That did not happen (§2/§3: max logit
difference from that checkpoint at initialization was 48–59, not ≈0). The
level0 run had to simultaneously relearn the label-correction adaptation
*and* learn the new residual branch, inside the same budget the baseline run
used only for the former. Both runs used the full 20-epoch budget without
early-stop patience triggering (§ training histories), and the level0 run's
own model-space balanced validation Dice was still rising at epoch 20
(+0.000264 over epoch 19, just over the 0.0003 improvement threshold) —
i.e., it had not clearly plateaued when the budget ran out, unlike the
baseline, whose peak (epoch 17) was not beaten again in its own remaining 3
epochs.

**Can we trust "using level0/full-resolution decoding did not help"?**
Not yet, as a general conclusion about full-resolution decoding — because
what was actually tested is a narrower, weaker variant (a small residual
correction bolted onto an unchanged, still-interpolated coarse decoder,
initialized from an older root checkpoint, given one fixed epoch budget that
had to cover two learning problems at once, evaluated with a second,
uncontrolled inference-tiling difference for undersized volumes). The
negative native-test result (Mouse Dice 0.969536→0.964860, only 4/80
subjects improved) is real and should not be discarded, but it supports a
narrower claim: *this specific low-capacity, residual-gated, generalization-pilot-initialized
implementation of level0 use did not help within one 20-epoch budget.* It
does not yet cleanly rule out a genuinely native, non-residual full-resolution
decoder head trained from the actual corrected-label baseline.

## Recommendation

Repeat the ablation with two corrections before trusting the "level0 does
not help" conclusion as a general architectural finding:

1. Initialize the shared levels1–4 tensors from
   `outputs/corrected_label_retraining/checkpoints/best_corrected_labels.pt`
   (wire up the already-present but unused `corrected_baseline_checkpoint`
   config key), so the level0 branch is added on top of the actually-converged
   corrected-label decoder rather than the epoch-14 pre-correction root.
2. Use the shared `sliding_window_logits` symmetric-padding tiling for
   native evaluation (or otherwise make the level0 evaluation path bit-for-bit
   identical to the baseline's outside of the added `level0` feature), so the
   test-time comparison has one, not two, uncontrolled differences.

Whether to also revisit the residual's low channel width (8, versus the
decoder's 32 elsewhere) and whether it had enough epoch budget to grow past
a small scale before the run ended are secondary questions worth watching in
the repeat, but are not, on their own, reasons to distrust this audit's
primary finding.
