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

## Completed Post-Filter Boundary-Error Diagnostics

The geometry of all 677,073 residual error voxels was measured on the complete
locked filtered test cohort (6 CAMRI, 80 Mouse) without training, inference,
threshold selection, model changes, or postprocessing changes. Distances were
measured from every FP/FN voxel to the 26-connected expert inner surface in both
array-index voxel units and native physical millimetres.

Combined index-space distance distribution:

- <=1 voxel: 96.006%;
- >1 to 2 voxels: 2.909%;
- >2 to 3 voxels: 0.817%;
- >3 to 5 voxels: 0.242%;
- >5 voxels: 0.026%.

Combined physical distance was mean 0.040 mm, median 0 mm, P95 0.200 mm, and
maximum 1.625 mm. CAMRI maximum was 1.000 mm and Mouse maximum was 1.625 mm.
The median is zero because incorrectly excluded expert-surface voxels are FN
errors located on the reference surface itself.

Errors are strongly under-segmentation dominated: 161,543 FP versus 515,530 FN
voxels, FP/FN ratio 0.313. The pattern holds in CAMRI (ratio 0.138) and Mouse
(0.326). Detached FP islands cannot enter these counts because only the saved
largest-component predictions were analyzed.

Normalized acquisition-slice position is non-uniform after normalization by
expert-surface opportunities, but terminal regions are not disproportionately
difficult: their combined rate is 0.795 times the middle-60% rate. The middle
60% has the highest normalized burden. NIfTI orientations vary (LPS, RIA, RPI),
so no rostral/caudal, dorsal/ventral, or left/right identities were invented.

Local expert-surface complexity used subject-wise tertiles of physical
signed-distance normal dispersion. High-complexity regions have 40.309 errors
per 100 surface voxels versus 26.294 in relatively flat regions, a 1.533-fold
increase. Thus visually curved/pixel-complex geometry is preferentially
affected, although the measure remains resolution-dependent.

MRI evidence does not support simple low-gradient ambiguity as the sole cause.
Regions with >2-voxel disagreement have higher mean gradient (8.294 versus
5.889 z/mm for correctly localized surface) but lower local inside/outside
contrast (0.738 versus 1.487 z). Assigned error distance has weak Spearman
association with gradient (rho 0.052), modest negative association with local
contrast (rho -0.147), and negligible association with variance (rho 0.008).

Probability errors are heterogeneous rather than predominantly uncertain:
41.72% of erroneous voxels lie in [0.25, 0.75], while 31.13% are confidently
incorrect by fixed descriptive definitions (FP >=0.9 or FN <=0.1). Combined FP
median probability is 0.739 and FN median is 0.181. These observations are not a
held-out threshold optimization.

Mean post-filter surface metrics are CAMRI Dice 0.982626, HD95 0.133 mm, ASSD
0.030 mm and Mouse Dice 0.965685, HD95 0.205 mm, ASSD 0.046 mm. Fixed Surface
Dice tolerances of 0.1/0.2/0.5/1.0 mm were declared before inspecting results.

The historical probability exporter stored equal-sized in-plane arrays in a
transposed order. Every probability map was aligned by the unique
shape-compatible axis permutation whose unchanged 0.5 threshold exactly
reproduced the canonical saved raw prediction. No probability value changed.

Scientific conclusion: most residual disagreement is already at approximately
one-index-voxel surface precision, but a small physically meaningful tail and a
clear complexity-associated failure pattern remain. The evidence does not by
itself justify a loss or architectural change. The next diagnostic should be a
blinded multi-rater review of the largest physical-distance and high-complexity
regions to quantify expert-boundary ambiguity.

Primary artifacts:

`outputs/boundary_error_diagnostics/`

## Completed Model Spatial-Resolution Diagnostics

The unchanged epoch-17 mixed-domain model was traced under inference mode on
CAMRI 064 and Mouse `POLYIC_20190510_mouse43__E9_P1`. The stepwise diagnostic
decoder agreed exactly with the ordinary forward pass (maximum absolute
difference 0), and native export reproduced both canonical saved raw masks
exactly. Geometry and quantization were then measured on all 86 filtered test
predictions. There was no training, threshold change, model change, or
postprocessing change.

Actual tensor hierarchy for a `[1,1,128,128,160]` tile:

- level0 `[1,48,128,128,160]`, full grid, available but unused by the decoder;
- level1 `[1,48,64,64,80]`, 2× downsampled and the finest decoder input;
- level2 `[1,96,32,32,40]`, 4×;
- level3 `[1,192,16,16,20]`, 8×;
- level4 `[1,384,8,8,10]`, 16×.

The decoder projects levels1–4 to 32 channels, performs three coarse-to-fine
trilinear FPN feature interpolations with additive lateral fusion and 3×3
refinement, and updates the single query by cross-attention at every fused
scale. The query-derived mask embedding is dotted with refined fused level1
features. Segmentation logits therefore first exist at `[1,1,64,64,80]`.
Exactly one 2× trilinear interpolation (`align_corners=False`) expands logits
to `[1,1,128,128,160]`; sigmoid and thresholding follow. There is no level0
skip or mask prediction at full model-input resolution.

Preprocessing is transpose `[1,0,2]`, nonzero crop, z-score normalization, and
order-1 resampling without anti-aliasing to model-axis spacing
`(0.25,0.20,0.16) mm`. CAMRI 064 changes from transposed `(144,64,144)` at
isotropic 0.2 mm to `(115,64,180)`. The Mouse representative changes from
`(180,35,180)` at model-axis spacing `(0.1,0.4,0.1) mm` to `(72,70,113)`, so
native Mouse detail is downsampled before encoding on two axes. The configured
preprocessor also calls its `is_seg=False`, order-1 resampler for segmentation
and then casts to `int8`; this is a verified implementation fact and a plausible
source of conservative supervision, not a demonstrated causal effect.

Level1 physical spacing is `(0.50,0.40,0.32) mm` in model axes, corresponding
to native NIfTI `(x,y,z)` spacing `(0.32,0.50,0.40) mm`. Its native footprint is
about `3.2×5.0×1.0` Mouse voxels and `1.6×2.5×2.0` CAMRI voxels. Mouse boundary
coordinates show a strong factor-5 alignment excess on native y (+28.71
percentage points versus expert), exactly matching its five-pixel level1
footprint. CAMRI native y also shows factor 5 (+18.74 points), consistent with
the repeated phase of its 2.5-pixel footprint. Native x-axis excesses are below
one point and are not meaningful detections.

Predicted contours are objectively less articulated, but not by every metric:
mean axis-run ratios are approximately 1.000 in both domains. P95 run ratios are
1.138 CAMRI and 1.177 Mouse; maximum-run ratios are 1.380 and 1.623. Direction
changes per 100 contour steps are only 0.824× expert in CAMRI and 0.694× expert
in Mouse. Thus the long-run tail is coarser and direction changes are reduced,
especially in Mouse, even though average run length is unchanged.

The mapped native probability field is continuous: 19.86% of boundary-band
samples are between 0.1 and 0.9, and 49.94% of 0.5-isocontour coordinate entries
are subvoxel. Thresholding reveals the native lattice staircase but does not
create the underlying coarse evidence; logits already originate at level1.

Scientific conclusion: the hypothesis is supported with qualifications. The
model localizes the brain very accurately, but fine boundary fidelity is
limited by two sequential effects: Mouse detail is first reduced during
native-to-model preprocessing, and the decoder then discards the available
full-grid level0 feature and forms its first mask at half resolution. This
mechanically explains why interpolation preserves excellent Dice/ASSD and
one-voxel localization while complex curves have 1.53× more error. It does not
prove that the decoder alone causes the error or explain all FN bias.

The next diagnostic should be an offline preprocessing-fidelity audit comparing
the current order-1/integer-cast preprocessed labels with nearest-neighbor
labels, measuring boundary displacement, curvature loss, and directional
erosion without training or modifying predictions.

Primary artifacts:

`outputs/model_spatial_resolution_diagnostics/`

## Completed TCIA Mouse-Astrocytoma Annotation Audit

The local `Mouse-Astrocytoma-doiJNLP` download was exhaustively inspected
without inference, conversion, or dataset modification. The audit covered
36,973 files in 382 directories (2.03 GiB):

- 36,966 readable DICOM instances;
- 283 locally present DICOM series;
- 48 subjects;
- one CSV download manifest;
- six `.DS_Store` files;
- no archive, unusual binary, extensionless DICOM, or non-DICOM medical image.

Every locally present DICOM instance is standard MR Image Storage
(`1.2.840.10008.5.1.4.1.1.4`), and every series has Modality `MR`. No DICOM
SEG, RTSTRUCT, surface segmentation, parametric map, SR, PR, encapsulated
document, ROI/contour sequence, DICOM overlay, or private annotation-like object
was found. No NIfTI, NRRD, MHA/MHD, ROI, label map, MAT, HDF5, NumPy, or VTK
annotation exists locally.

The 218 screened candidates consist of 217 ordinary MR series whose descriptions
contain disease/anatomy terms (`intracranial` or `brain tumor`) plus the manifest
CSV. All are rejected as annotations. Confirmed whole-brain masks, tumor masks,
annotation objects, and annotated subjects are all zero. The local metadata
contains no annotation reference. Quantitative Dice evaluation is therefore not
currently possible.

Manifest reconciliation identified a separate download-completeness issue:
286 successful manifest rows map to 286 UIDs, but only 283 UIDs remain in local
headers. Three pairs use the same destination folders, so one same-sized
subtraction image series in each pair appears to have overwritten the other.
This is not evidence of annotation data, but it prevents claiming that every
manifest image UID is simultaneously present.

The paper's statement that manual whole-brain and tumor masks were created does
not prove public release. Highest-priority follow-up: externally verify the TCIA
collection's Analysis Results/related DOI packages, then request both manual
mask sets from the study authors if no separate label package exists.

Primary artifacts:

`outputs/mouse_astrocytoma_annotation_audit/`

## Completed Zero-Shot Mouse Astrocytoma Pathology Pilot

The unchanged epoch-17 mixed-domain one-query model was run qualitatively on
four preselected TCIA primary-anatomical series: IC1 TSE, IC2 post-contrast T1,
U87 TSE, and U87 pre-contrast T1. The RS2-Net encoder and 170,401-parameter
decoder were frozen. There was no training, adaptation, architecture,
preprocessing, or threshold change. Raw masks and the existing
largest-26-connected-component filtered masks were saved independently.

The complete pre-inference QC reconstructed all 283 series from 48 subjects:
140 primary anatomical, 47 secondary anatomical, 50 dynamic, and 46
subtraction. Every series has a contact sheet and metadata record. Every pilot
has dense native and preprocessed contact sheets, raw/filtered/probability
review pages, a full-slice MP4, orthogonal views, and raw/filtered 3D surfaces.

Probability export required an in-plane axis-order reconciliation between
SimpleITK and nibabel. The accepted map was required to reproduce the raw mask
exactly at probability 0.5; all four maps have zero mismatched voxels.

Confirmed computational observations:

- IC1 TSE: 12 raw components, 11,874 voxels removed, 38 cavity voxels;
- IC2 post-contrast T1: 2 raw components, 4 removed voxels, 248 cavity voxels;
- U87 TSE: 10 raw components, 11,888 removed voxels, 14 cavity voxels;
- U87 pre-contrast T1: 4 raw components, 29 removed voxels, 21 cavity voxels.

Visual review suggests broadly continuous bilateral envelope retrieval in the
two T1 pilots, but strong partial-exclusion/hemisphere-failure candidates in
both TSE pilots. This is not a quantitative accuracy claim: no expert
whole-brain or tumor masks exist locally. Cavity and indentation products are
analysis-only and never modify predictions. The largest-component filter
removes detached islands but cannot restore omitted connected anatomy or fill
cavities.

Scientific conclusion: the learned query does not yet have defensible evidence
of complete-envelope robustness under unseen pathology. Sequence/contrast
domain shift and pathology cannot be disentangled in this four-case,
unmatched-control pilot. Obtain expert whole-brain and tumor masks for these
same series before changing the architecture or drawing pathology-specific
conclusions.

Primary artifacts:

`outputs/mouse_astrocytoma_zero_shot/`

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
## Expert-mask preprocessing lifecycle audit (2026-08-12)

- A full diagnostic audit of the locked mixed-domain cohort (40 CAMRI and 101 Mouse scans) is in `outputs/mask_preprocessing_audit/`; it did not train a model or regenerate predictions.
- The current train/validation target path is: original expert NIfTI -> SimpleITK float32 `[C,Z,Y,X]` -> RS2 `transpose_forward=[1,0,2]` -> image-nonzero crop -> the shared `3d_fullres` resampler configured with `is_seg=False, order=1, order_z=0` -> final int8 truncation -> `>0` -> center pad/crop to `[1,1,128,128,160]` -> uint8 cached target. Training augmentation may flip this target; validation does not augment it.
- `DefaultPreprocessor.run_case` uses the exact same `configuration_manager.resampling_fn` for image and segmentation. The project then thresholds the int8 result. Thus training and validation labels are linearly interpolated and integer-truncated, rather than label-aware nearest-neighbor masks.
- Mixed-domain training validation Dice/precision/recall are computed in model space against this cached processed target. Canonical CAMRI/Mouse native evaluation instead discards the preprocessed segmentation, exports logits to native geometry, reloads the original `mask_path`/`ground_truth_path` directly with nibabel, and binarizes it. Current native Dice, IoU, precision, recall, HD95, ASSD, filtered-error, and boundary-diagnostic values therefore use untouched native expert masks. Order-1 interpolation in native export applies to logits/probabilities, not the expert label.
- Relative to an otherwise identical label-aware nearest-neighbor forward resample, current processed masks were smaller in all 141/141 subjects and never contained added foreground voxels. CAMRI lost 249,659 pooled processed voxels, mean volume change was -2.939%, mean current-vs-NN Dice 0.985067, HD95 0.1880 mm, ASSD 0.03884 mm, and mean inward shift 0.1839 mm. Mouse lost 230,563 pooled processed voxels, mean volume change was -3.983%, mean Dice 0.979678, HD95 0.1925 mm, ASSD 0.03247 mm, and mean inward shift 0.1840 mm.
- After identical nearest-neighbor restoration to the native grid, mean Dice against the original expert was 0.984560 (current) versus 0.992641 (NN reference) for CAMRI and 0.975802 (current) versus 0.981712 (NN reference) for Mouse.
- This is verified systematic erosion and a plausible contributor to learned FN bias, but not proof of causation because final test metrics use untouched native labels and optimization/domain shift remain independent causes. The controlled next step is to build a separate label-aware cache and retrain the unchanged decoder, changing only segmentation interpolation, before interpreting a full-resolution decoder experiment. Preserve all current checkpoints/results as comparators.

## Corrected-label controlled retraining (2026-08-12)

- The segmentation-only preprocessing correction was tested in `outputs/corrected_label_retraining/`. MRI preprocessing and the frozen RS2 encoder were unchanged. The decoder remained the same one-query, levels1–4, 170,401-parameter architecture; split, seed, initialization, augmentations, Dice+BCE+boundary loss, optimizer, 20-epoch schedule, 0.5 threshold, native export, and largest-26-connected-component filter were preserved.
- Full 141-mask verification confirmed categorical `{0,1}` nearest-neighbor targets and removal of systematic shrinkage. Native round-trip mean volume bias changed from -2.578% to +0.121% CAMRI and -4.042% to -0.084% Mouse. Round-trip Dice improved 0.984560→0.992641 CAMRI and 0.975802→0.981712 Mouse.
- Training started from the same epoch-14 checkpoint and again selected epoch 17. Balanced validation Dice was 0.976708 versus 0.976576 old; the CAMRI safety stop did not trigger. The checkpoint is `outputs/corrected_label_retraining/checkpoints/best_corrected_labels.pt`.
- On the same untouched native test cohort after the same deterministic filter, CAMRI Dice improved 0.982626→0.987029, recall 0.970597→0.988249, ASSD 0.03034→0.01997 mm, and total error 38,341→27,398; HD95 remained 0.1333 mm. Mouse Dice improved 0.965685→0.969536, recall 0.948492→0.972760, ASSD 0.04605→0.04214 mm, HD95 0.2050→0.2009 mm, and total error 638,732→585,309.
- FN voxels fell 33,693→12,791 CAMRI and 481,837→254,256 Mouse, confirming the damaged labels materially contributed to FN bias. FP increased 4,648→14,607 and 156,895→331,053. Pooled FP/FN moved from 0.138→1.142 CAMRI and 0.326→1.302 Mouse: severe under-segmentation was corrected but calibration modestly overshot toward FP.
- The remaining distance tail did not improve. Combined <=1-voxel share fell 96.006%→89.288%, >1–2 rose 2.909%→7.829%, >2–3 rose 0.817%→2.159%, >3–5 rose 0.242%→0.636%, and >5 rose 0.026%→0.088%; physical mean/P95 rose 0.0397/0.200 mm→0.0883/0.300 mm. Many zero-distance surface FN voxels were recovered while farther outward FP remained.
- High-complexity errors decreased modestly (256,631→242,849; 40.309→38.145 per 100 surface voxels), but flat errors improved more. Contour articulation and pixelation were unchanged: direction-change ratios stayed about 0.82 CAMRI and 0.69 Mouse; factor-5 native-y alignment excess remained and slightly increased (18.74→22.03 points CAMRI, 28.71→29.69 Mouse).
- Scientific attribution: erroneous label interpolation caused systematic eroded supervision, severe FN/recall bias, and part of the overlap/ASSD error. It did not cause the factor-5 quantization, reduced contour articulation, or visible staircase pattern. The next justified controlled experiment is corrected labels plus a full-resolution level0 decoder, changing level0 use as the sole architectural factor.

## Full-resolution level0 decoder ablation (2026-08-12)

- The controlled corrected-label level0 ablation is complete in `outputs/full_resolution_level0_decoder/`. The frozen encoder, one learned query, split, seed, initialization, loss, threshold, native export, and deterministic 26-connected-component filter were preserved. Only a lightweight full-grid level0 fusion/query/mask-residual stage was added.
- The decoder produced genuine learned `[1,1,128,128,160]` logits without final logit interpolation, versus `[1,1,64,64,80]` plus 2x interpolation previously. Trainable parameters increased 170,401→180,466 (+5.91%). Epoch 20 was selected; the encoder remained frozen.
- On the same filtered untouched tests, CAMRI Dice changed 0.987029→0.986841, HD95 stayed 0.1333 mm, ASSD worsened 0.01997→0.02081 mm, and total error rose 27,398→28,112. Mouse Dice fell 0.969536→0.964860, HD95 worsened 0.2009→0.2649 mm, ASSD 0.04214→0.05524 mm, and total error rose 585,309→680,375. Only 4/80 Mouse subjects improved.
- Combined <=1-voxel errors fell 89.288%→83.301%; >1–2 rose 7.829%→11.437%, >2–3 2.159%→3.742%, >3–5 0.636%→1.272%, and >5 0.088%→0.248%. Mean/P95 physical distance worsened 0.0883/0.300→0.1315/0.400 mm. CAMRI distance concentration improved modestly, but Mouse dominated the regression.
- High-complexity error rates worsened 25.232→26.101 per 100 surface voxels CAMRI and 39.700→46.721 Mouse. Direction-change/expert ratios improved only slightly (0.8207→0.8243 and 0.6866→0.6976). Factor-5 alignment excess decreased 22.03→15.94 points CAMRI but remained 29.69→29.30 Mouse; visible lattice/pixelation did not meaningfully disappear.
- Mouse confident incorrect boundary FP increased 27.48%→38.30%, while boundary gradients and intermediate probabilities did not show useful finer localization. The full-grid field was real but more confident without being more correct.
- Scientific attribution: the hypothesis that the boundary limitation was primarily caused by discarding level0 spatial information is not supported. Added level0 capacity produced minor contour-statistic changes but worsened cross-domain segmentation. Stop this architecture path. The next diagnostic is a no-training measurement of factor-5 alignment in native level0 activations and preprocessed image gradients, stratified by Mouse acquisition group, to localize the grid/domain source.
