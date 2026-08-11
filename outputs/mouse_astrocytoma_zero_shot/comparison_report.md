# Zero-shot Mouse Astrocytoma pathology pilot

## Scope and safeguards

This was a **qualitative-only** zero-shot experiment. No whole-brain or tumor
ground truth is available locally, so no Dice, sensitivity, specificity, or
pathology accuracy is reported. The epoch-17 mixed-domain checkpoint was loaded
strictly; its RS2-Net encoder was frozen; its one learned query, multi-scale
attention, FPN, and mask head were unchanged. There was no training, adaptation,
threshold selection, preprocessing change, morphology, hole filling, or manual
mask editing. The only binary cleanup was the established largest
26-connected-component filter. Raw predictions remain saved separately.

The complete audit covered 48 subjects and 283 reconstructed MR series:
140 primary anatomical, 47 secondary anatomical, 50 dynamic, and 46
subtraction series. Four distinct primary-anatomical pilots were selected
before inference: IC1 TSE, IC2 post-contrast T1, U87 TSE, and U87 pre-contrast
T1. Every original and preprocessed slice is available in dense contact sheets,
and every native slice is available in MP4 review files.

## Confirmed computational facts

| Pilot | Sequence | Raw components | Filter-removed voxels | Analysis-only cavity voxels |
|---|---|---:|---:|---:|
| IC1 100110 | TSE | 12 | 11874 | 38 |
| IC2 021811 | post-contrast T1 | 2 | 4 | 248 |
| U87 021412 | TSE | 10 | 11888 | 14 |
| U87 021512 | pre-contrast T1 | 4 | 29 | 21 |

Probability-map geometry was accepted only after its 0.5 threshold matched the
raw mask at every voxel (zero mismatches for all four pilots). Cavity masks are
defined as `binary_fill_holes(filtered) - filtered`; they are analysis-only and
were never applied to predictions.

## Visual observations

- The two T1 pilots produce one dominant, broadly continuous intracranial mask.
  Their contours visually track much of the apparent outer brain envelope.
- Both TSE pilots produce a dominant mask concentrated on the conspicuously
  bright, expanded side of the intracranial volume. Large darker contralateral
  regions that appear brain-like are not included. This is a strong
  **partial-exclusion / hemisphere-failure candidate**, not a confirmed
  anatomical error without expert masks.
- Raw TSE predictions contain 10–12 components, and the fixed filter removes
  about 11.9k voxels in each. The removed objects are spatially detached; their
  anatomical status cannot be established without labels.
- Every pilot contains at least one mathematically enclosed background cavity.
  The post-contrast IC2 series has the largest cavity burden (248 voxels across
  six components). Some coincide visually with dark internal structures;
  whether these should belong to a whole-brain envelope requires expert review.
- The fixed indentation screen flags many slices, especially in the TSE and
  post-contrast series. Because acquisition axes and terminal anatomy can create
  abrupt area changes naturally, these are review candidates rather than
  failures.

## Hypotheses

The contrast-dependent difference—dominant-side retrieval in TSE versus broad
bilateral inclusion in T1—suggests that tumor-associated intensity and mass
effect may compete with the learned whole-brain grouping cue. A second plausible
cause is acquisition-domain shift: these Philips pathology sequences differ
substantially from the healthy CAMRI/Mouse training distributions. The current
evidence cannot separate pathology shift from sequence/contrast shift.

Healthy-control cavity/asymmetry analysis provides context but not a matched
causal control: the healthy images come from different datasets and protocols.
It therefore cannot prove that the observed cavities or asymmetry are caused by
tumor.

## Answers to the scientific questions

1. **Does the learned query always segment the complete brain envelope?** No
   defensible universal claim can be made. The T1 pilots appear broadly complete,
   while both TSE pilots show visually compelling partial-exclusion candidates.
2. **Is there catastrophic failure?** No empty mask or total retrieval failure
   occurred. The TSE behavior is nevertheless a major candidate failure.
3. **Are masks fragmented?** Raw masks are multi-component in all four cases;
   the established filter yields one component by definition.
4. **Are there internal cavities?** Yes computationally, in every filtered mask.
   Their anatomical correctness is unverified.
5. **Are there inward indentations?** Automated candidates exist, but none is
   confirmed without expert annotation.
6. **Is hemisphere failure present?** It is a strong visual candidate in both
   TSE pilots.
7. **Does pathology appear excluded?** This cannot be determined: no tumor mask
   exists and image appearance alone is insufficient for tumor delineation.
8. **Does the largest-component filter solve the pathology problem?** No. It
   removes detached islands but cannot restore omitted connected anatomy or fill
   cavities.
9. **What should happen next?** Obtain expert whole-brain and tumor masks for
   these same series, or have a qualified reader annotate the saved review
   panels. Only then quantify envelope inclusion and localize errors relative to
   tumor. No model modification is justified by this qualitative pilot alone.

## Artifact guide

- `series_inventory.csv`, `dataset_inventory.csv`: complete audit.
- `series_qc/`, `qc_overviews/`: pre-inference review of every series.
- `pilots/<subject>/contact_sheets/`: every native/preprocessed/prediction slice.
- `pilots/<subject>/full_slice_review.mp4`: complete native-volume review.
- `pilots/<subject>/surfaces/`, `cavity_analysis/`, and
  `boundary_candidate_zooms/`: 3D and focused analysis.
- `healthy_controls/`, `healthy_control_comparison.csv`, and
  `pathology_vs_healthy_controls.png`: descriptive control comparison.
- `failure_atlas/`: nonexclusive visual candidate categories.
