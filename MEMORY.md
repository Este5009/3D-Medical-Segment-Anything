# MEMORY.md

# Long-Term Vision

Build a universal query-conditioned volumetric segmentation framework that
learns anatomically meaningful grouping.

The long-term objective is not brain segmentation, rodent segmentation, or
maximizing Dice on one dataset. The model should identify complete anatomical
entities from image evidence, reason with global, regional, and local 3D
context, and generalize across species, scanners, institutions, protocols,
contrast, image quality, artifacts, pathology, and anatomical variation.

The model should ultimately propose multiple coherent anatomical structures.
A clinician may select a proposal or optionally refine it.

---

# Core Architecture Vision

3D Volume

↓

Transformer Encoder

↓

Rich Volumetric Feature Pyramid

↓

Learned Object Queries

↓

Transformer Query Decoder

↓

Automatic 3D Mask Proposals

↓

Optional Prompt Refinement

---

# Current Roadmap

Phase 1

Reproduce state-of-the-art baselines.

Completed:

- MedSAM
- MedSAM2
- RS2-Net

Phase 2

Separate encoder from decoder.

Freeze encoder initially.

Train only a learned query decoder.

Compare against the original RS2-Net decoder.

Current Phase 2 evidence:

- the verified RS2-Net encoder is safely exposed without its decoder;
- one learned query plus four-scale attention and a top-down FPN has adequate
  tiny-set capacity;
- the frozen architecture generalizes strongly within CAMRI;
- naive Mouse transfer retrieves the brain but over-segments it;
- Mouse-only boundary adaptation improves Mouse but causes CAMRI forgetting;
- balanced mixed-domain full-decoder training preserves CAMRI and substantially
  improves Mouse.

Current controlled question:

Can the identical frozen-encoder, one-query architecture learn better
cross-domain anatomical boundaries through supervision, optimization, sampling,
and augmentation alone?

## Completed Query-Conditioned 3D Grouping Experiment

A controlled post-decoder grouping experiment used the unchanged epoch-17
mixed-domain checkpoint and the identical locked CAMRI/Mouse splits. The
RS2-Net encoder and 170,401-parameter one-query decoder remained frozen.

The tested module projected frozen level-2 features, concatenated downsampled
initial logits/probability/uncertainty and optional normalized coordinates,
applied two query-FiLM depthwise-separable 3D residual blocks plus a global
context gate, and added a bounded correction to the frozen logits. It contained
8,537 parameters with coordinates. The preserved checkpoint is:

`outputs/query_conditioned_3d_grouping/checkpoints/best_grouping_module.pt`

Validation selected the coordinate/no-query ablation at epoch 1:

- baseline balanced validation Dice: 0.976576;
- selected learned Dice: 0.976714 (+0.000138);
- coordinate+query Dice: 0.976706;
- no-coordinate/query Dice: 0.976683.

Query conditioning therefore made no measurable contribution. On the complete
untouched tests, learned grouping regressed CAMRI Dice from 0.982518 to 0.981967
and Mouse Dice from 0.964762 to 0.963906. It reduced FP voxels but also removed
correct anatomy: recall and volume ratio fell in both domains, Mouse HD95
slightly worsened, and connected-component counts were not improved.

The deterministic inference-only largest-26-connected-component filter was
stronger: CAMRI Dice 0.982626 and Mouse Dice 0.965685, with unchanged recall and
Mouse HD95 improved from 0.3435 to 0.2163 mm. Exact three-slice, eight-panel
good/median/worst figures for each domain and the detached-FP atlas are under
`outputs/query_conditioned_3d_grouping/`.

Scientific conclusion: deterministic 3D filtering captures the available
grouping benefit; the learned grouping decoder is not justified on the current
benchmarks and should not be integrated or made more complex.

## Completed Filtered Residual-Failure Analysis

The complete original residual-failure analysis was repeated on the 86
already-saved deterministic-filter test predictions. No inference, retraining,
threshold selection, preprocessing change, or split change occurred. The sole
changed variable was the prediction path.

The filter uses `scipy.ndimage.label` with a full 3×3×3 structuring element
(26-connectivity) and retains the largest foreground component. It is applied
to the complete 3D binary mask. It has no morphology, component-size threshold,
image input, expert-label input, or dataset-specific parameter.

Absolute exclusive residual attribution changed as follows:

- total error: 695,034 → 677,073 voxels (-17,961; -2.584%);
- detached FP: 17,961 → 0 voxels; 32 → 0 affected subjects;
- terminal: 2,064 → 1,933 voxels (-131); 16 → 14 subjects;
- connected leakage: 1,356 → 1,356 voxels; 57 → 57 subjects;
- detached FN and localization: zero before and after;
- final exclusive boundary bucket: 673,653 → 673,784 voxels (+131).

The exclusive boundary increase is attribution reallocation, not newly created
error. The direct non-exclusive physical 0.5-mm boundary mask decreased by 52
voxels (673,333 → 673,281). Its exclusive percentage rose from 96.924% to
99.514% primarily because detached FP voxels disappeared from the denominator.
All residual voxels remained classified; unclassified count was zero.

Safety audit:

- 57 detached components totaling 17,961 voxels were removed;
- removed-component overlap with expert anatomy was exactly zero;
- recall was identical before and after in both domains;
- full-volume connectivity preserves structures joined across adjacent slices;
- no erosion, smoothing, or thresholding was used.

Filtered test performance reproduced the prior result exactly:

- CAMRI Dice 0.982518 → 0.982626; HD95 unchanged at 0.1333 mm;
- Mouse Dice 0.964762 → 0.965685; HD95 0.3435 → 0.2163 mm;
- combined Dice 0.966001 → 0.966867; HD95 0.3289 → 0.2105 mm.

The regenerated atlas contains mild/representative/severe examples for every
remaining category. The detached-FP directory is explicitly empty. Six
prediction-extent-selected comparison figures cover best/median/worst CAMRI and
Mouse cases.

Scientific conclusion:

**Deterministic largest-26-connected-component filtering should be the default
binary inference cleanup for the current single-coherent-object task, while raw
logits/probabilities must remain available. Future architecture work should
target boundary refinement.**

Primary artifacts:

`outputs/filtered_residual_failure_analysis/`

Phase 3

Fine-tune encoder and query decoder jointly.

Phase 4

Train with multiple anatomical structures.

Phase 5

Add interactive prompt refinement.

---

# Research Principles

The encoder should represent anatomy.

Queries should retrieve objects.

The decoder should delineate precise voxel boundaries.

Avoid organ-specific assumptions whenever possible.

Train for anatomical objectness rather than fixed semantic classes.

Dice is evidence, not the project objective. Prefer anatomically supported
segmentations and robust generalization over benchmark-specific gains. Do not
use post-processing, threshold tuning, anatomical shrinking, or dataset-specific
heuristics as substitutes for learned anatomical grouping.

Change one scientific factor at a time. Do not redesign the architecture until
error analysis provides evidence of a representational or capacity limitation.

---

# Current Data

Controlled benchmarks:

- CAMRI Rat MRI skull stripping.
- External Mouse MRI skull stripping.

Ground truth:

Whole brain masks.

Future datasets will include multiple anatomical structures.

All experiments must use deterministic subject-level splits. Known longitudinal
Mouse scans must remain in one split. Identity-unknown Mouse scans are a
documented residual leakage limitation and must never be assigned invented
biological identities.

---

# Important Rule

Do not modify reproduction repositories.

They are permanent baselines.

Research code belongs only inside this repository.

---

# Evaluation Contract

Report CAMRI and Mouse independently. Include volumetric, surface, per-subject,
per-slice, terminal-slice, topology, volume-error, and inference-time evidence.
Automatically rank worst Dice, HD95, FP, FN, terminal, disconnected-component,
and leakage-region failures before drawing architectural conclusions.

---

# Completed Residual-Failure Analysis

The validation-selected mixed-domain anatomical checkpoint (epoch 17) was
analyzed without retraining or rerunning inference. The cohort contained every
untouched test prediction: 6 CAMRI and 80 Mouse subjects. Analysis used native
geometry, 26-connected components, and a 0.5 mm physical boundary band.

Exclusive error attribution across 695,034 erroneous voxels:

- Boundary/local contour error: 96.92%.
- Detached false-positive islands: 2.58%.
- Terminal endpoint failures: 0.30%.
- Connected leakage: 0.20%.
- Detached false-negative regions: 0%.
- Major localization failures: 0%.

All 86 subjects had some boundary disagreement. Detached FP islands affected
32 subjects (37.2%), terminal endpoint mismatch affected 16 (18.6%), and small
connected leakage affected 57 (66.3%). Leakage was frequent but tiny: 0.20% of
all error voxels and 0.11 mm3 mean category-specific FP volume per affected
subject. No subject met the declared major-localization criterion.

The predictions contained 57 zero-expert-overlap detached FP islands totaling
17,961 voxels. Twenty-five islands totaling 7,540 voxels had mean MRI intensity
within one expert-brain standard deviation. These are the measured cases where
local appearance supports foreground but 3D disconnection contradicts object
membership. Grouping-associated categories collectively account for only 2.78%
of all remaining error voxels.

Representative Failure Atlas examples:

- Boundary representative: Mouse
  `POLYIC_20190524_polyic___E3_P1_1`, 7,610 boundary-error voxels.
- Detached-FP severe: Mouse
  `POLYIC_20190531_polyic_m__E3_P1_3`, a 25.40 mm3 island spanning 11
  slices, 10.11 mm centroid distance from the main brain, zero expert overlap.
- Terminal severe: Mouse `POLYIC_20190614_polyic___E3_P1_5`, prediction
  extends one superior slice beyond the expert endpoint.
- Leakage severe: Mouse `POLYIC_20190510_mouse43__E9_P1`, 606 connected FP
  voxels farther than 0.5 mm from expert anatomy.

The primary reference artifacts are under
`outputs/mixed_domain_anatomical_training/failure_analysis/`, especially
`failure_statistics.csv`, `connected_component_statistics.csv`,
`failure_summary.md`, and `Failure_Atlas/`.

Research direction from this completed analysis:

**A. The current architecture is primarily boundary-limited.**

Anatomical grouping errors are real and visually clear in a minority of cases,
but their 2.78% exclusive attribution is not the dominant current limitation.
Future work should treat boundary accuracy as the next controlled research
target; explicit grouping changes are not quantitatively justified as the
primary next direction by this experiment.
