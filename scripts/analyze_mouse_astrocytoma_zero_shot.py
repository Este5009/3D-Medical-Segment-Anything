#!/usr/bin/env python3
"""Build the qualitative comparison report and failure atlas.

All pathology labels in this script are *visual candidate categories*.  They
are not diagnoses and are not accuracy measurements because no expert masks
are available for the TCIA pathology collection.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from models.query_conditioned_grouping import largest_component_3d
from run_mouse_astrocytoma_zero_shot import (
    align_probability_to_raw, cavity_analysis, indentation_analysis, orthogonal_figure,
    read_csv, robust_normalize, save_json, surface_figure, write_csv,
)


def asymmetry(mask: np.ndarray) -> float:
    coords = np.argwhere(mask)
    if not len(coords):
        return 1.0
    midpoint = (coords[:, 0].min() + coords[:, 0].max()) / 2
    left = int((coords[:, 0] <= midpoint).sum())
    right = int((coords[:, 0] > midpoint).sum())
    return abs(left - right) / max(left + right, 1)


def healthy_controls(output: Path) -> list[dict]:
    """Apply the same prediction-only analyses to four fixed healthy controls."""
    roots = {
        "CAMRI-007": ROOT / "outputs/mixed_domain_anatomical_training/native_predictions/camri_mixed/007_metrics.json",
        "CAMRI-099": ROOT / "outputs/mixed_domain_anatomical_training/native_predictions/camri_mixed/099_metrics.json",
    }
    mouse_json = sorted((ROOT / "outputs/mixed_domain_anatomical_training/native_predictions/mouse_mixed").glob("*_metrics.json"))
    roots.update({f"Mouse-{index + 1}": path for index, path in enumerate(mouse_json[:2])})
    results = []
    for control_id, metadata_path in roots.items():
        metadata = json.loads(metadata_path.read_text())
        image_obj = nib.load(metadata["image_path"])
        image = np.asarray(image_obj.dataobj, dtype=np.float32)
        raw = np.asarray(nib.load(metadata["prediction_path"]).dataobj) > 0
        probability = np.asarray(nib.load(metadata["probability_path"]).dataobj, dtype=np.float32)
        probability, alignment = align_probability_to_raw(probability, raw)
        filtered = largest_component_3d(raw)
        cavity, cavities = cavity_analysis(filtered.astype(bool), image_obj.affine)
        candidates = indentation_analysis(filtered.astype(bool), probability)
        destination = output / "healthy_controls" / control_id
        orthogonal_figure(image, raw, filtered.astype(bool), probability,
                          destination / "orthogonal_views.png", f"{control_id} healthy control")
        surface_figure(filtered, destination / "cavity_surface.png",
                       f"{control_id}: analysis-only cavity", cavity=cavity)
        result = {
            "control_id": control_id, "domain": metadata["domain"],
            "scan_id": metadata["scan_id"], "raw_components": int(metadata["connected_components"]),
            "filtered_voxels": int(filtered.sum()), "cavity_components": len(cavities),
            "cavity_voxels": int(cavity.sum()), "whole_mask_asymmetry": asymmetry(filtered),
            "boundary_candidate_slices": len(candidates),
            "probability_threshold_alignment_mismatches":
                alignment["threshold_mismatched_voxels"],
        }
        save_json(destination / "analysis.json", result)
        results.append(result)
    write_csv(output / "healthy_control_comparison.csv", results)
    return results


def build_failure_atlas(output: Path, pathology: list[dict]) -> dict:
    """Copy complete review figures into transparent candidate categories."""
    categories = {
        "plausible_complete_inclusion": [
            "TVD_GBM_IC2_11311059_021811", "TVD_GBM_U87_1202252_021512"],
        "partial_exclusion_or_hemisphere_failure_candidate": [
            "TVD_GBM_IC1_070610_16_100110", "TVD_GBM_U87_1202141_021412"],
        "internal_cavity_candidate": [
            row["subject_id"] for row in pathology if int(row["cavity_voxels"]) > 0],
        "fragmented_raw_prediction": [
            row["subject_id"] for row in pathology if int(row["raw_components"]) > 1],
        "boundary_indentation_candidate": [
            row["subject_id"] for row in pathology if int(row["boundary_candidate_slices"]) > 0],
    }
    atlas = output / "failure_atlas"
    for category, subjects in categories.items():
        for subject in subjects:
            source = output / "pilots" / subject / "orthogonal_views.png"
            destination = atlas / category / f"{subject}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    save_json(atlas / "atlas_index.json", {
        "scope": "visual candidate categories without expert pathology masks",
        "nonexclusive_categories": categories,
    })
    return categories


def comparison_figure(pathology: list[dict], controls: list[dict], destination: Path):
    labels = [row["subject_id"].split("_")[-2] for row in pathology] + [row["control_id"] for row in controls]
    cavities = [int(row["cavity_voxels"]) for row in pathology + controls]
    asymmetries = [asymmetry(
        np.asarray(nib.load(str(ROOT / "outputs/mouse_astrocytoma_zero_shot/pilots" /
                                      row["subject_id"] / "filtered_prediction.nii.gz")).dataobj) > 0
    ) for row in pathology] + [float(row["whole_mask_asymmetry"]) for row in controls]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    colors = ["#d1495b"] * len(pathology) + ["#2a9d8f"] * len(controls)
    axes[0].bar(labels, cavities, color=colors)
    axes[0].set(title="Analysis-only enclosed cavity voxels", ylabel="Voxels")
    axes[1].bar(labels, asymmetries, color=colors)
    axes[1].set(title="Prediction-mask left/right asymmetry", ylabel="Absolute volume fraction difference")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=.25)
    figure.suptitle("Pathology pilots (red) versus healthy controls (green); descriptive only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def write_report(output: Path, pathology: list[dict], controls: list[dict], categories: dict):
    by_id = {row["subject_id"]: row for row in pathology}
    report = f"""# Zero-shot Mouse Astrocytoma pathology pilot

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
| IC1 100110 | TSE | {by_id['TVD_GBM_IC1_070610_16_100110']['raw_components']} | {by_id['TVD_GBM_IC1_070610_16_100110']['removed_voxels']} | {by_id['TVD_GBM_IC1_070610_16_100110']['cavity_voxels']} |
| IC2 021811 | post-contrast T1 | {by_id['TVD_GBM_IC2_11311059_021811']['raw_components']} | {by_id['TVD_GBM_IC2_11311059_021811']['removed_voxels']} | {by_id['TVD_GBM_IC2_11311059_021811']['cavity_voxels']} |
| U87 021412 | TSE | {by_id['TVD_GBM_U87_1202141_021412']['raw_components']} | {by_id['TVD_GBM_U87_1202141_021412']['removed_voxels']} | {by_id['TVD_GBM_U87_1202141_021412']['cavity_voxels']} |
| U87 021512 | pre-contrast T1 | {by_id['TVD_GBM_U87_1202252_021512']['raw_components']} | {by_id['TVD_GBM_U87_1202252_021512']['removed_voxels']} | {by_id['TVD_GBM_U87_1202252_021512']['cavity_voxels']} |

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
"""
    (output / "comparison_report.md").write_text(report)


def main():
    output = ROOT / "outputs/mouse_astrocytoma_zero_shot"
    pathology = read_csv(output / "pilot_summary.csv")
    controls = healthy_controls(output)
    categories = build_failure_atlas(output, pathology)
    comparison_figure(pathology, controls, output / "pathology_vs_healthy_controls.png")
    write_report(output, pathology, controls, categories)
    save_json(output / "qualitative_conclusion.json", {
        "quantitative_accuracy_claimed": False,
        "complete_envelope_in_all_pilots": False,
        "t1_visual_observation": "broadly continuous bilateral inclusion candidate",
        "tse_visual_observation": "partial exclusion / hemisphere failure candidate",
        "catastrophic_empty_failure": False,
        "expert_review_required": True,
    })


if __name__ == "__main__":
    main()
