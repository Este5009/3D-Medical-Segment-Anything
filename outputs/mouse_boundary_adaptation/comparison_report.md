# Mouse boundary-head adaptation

The validation-selected epoch-16 boundary-head checkpoint trained 29,793 parameters: `mask_embedding`, `mask_refinement`, and `mask_bias`. The encoder, query, attention blocks, projections, and FPN remained frozen. The loss was `0.50 Dice + 0.25 BCE + 0.25 Tversky(alpha_FP=0.70, beta_FN=0.30)`.

The deterministic split contains 11 scans/3 identified mice for training, 10 scans/3 different identified mice for validation, and 80 untouched test scans (9 identified mice plus 52 identity-unknown scans). All known longitudinal scans remain together. Unknown filenames remain a documented residual linkage limitation.

## Untouched test result

Mean Dice changed from 0.8819 to 0.9347 (paired change +0.0528; Wilcoxon p=7.85e-15). Precision changed 0.8076->0.9633; recall 0.9727->0.9081; volume ratio 1.208->0.943; FP voxels 27,685->4,141. 80 scans improved and 0 degraded.

Surface Dice improved 0.5304->0.6813, and boundary precision improved 0.4635->0.6797. Rostral FP fell 1,522,818.0->219,750.0 (85.6%); dorsal FP fell 1,262,558.0->179,678.0 (85.8%). First-20% slice Dice improved 0.7481->0.8953, and middle-60% improved 0.9061->0.9423, but last-20% worsened 0.7851->0.7171. The recall and terminal-slice losses show that the FP-aware calibration slightly over-corrects.

Boundary-head-only adaptation satisfied all validation gates, so full-decoder Experiment C was not triggered. CAMRI retention mean Dice changed from 0.9762 to 0.9133 (-0.0628), with worst individual change -0.0941. This is substantial domain specialization, although not total collapse.

## Decision

Boundary-head-only adaptation suffices to correct the primary mouse FP leakage, but the selected checkpoint is **mouse-specialized rather than broadly transferable** because it reduces recall, worsens the last brain region, and harms CAMRI retention. The next controlled experiment should add a fixed CAMRI rehearsal/retention constraint during the same boundary-head-only training; the encoder, query, attention, and FPN should remain frozen. That directly tests whether boundary calibration can improve mouse performance without forgetting CAMRI.
