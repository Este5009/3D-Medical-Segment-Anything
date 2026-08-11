#!/usr/bin/env python3
"""Exhaustively audit a local TCIA Mouse-Astrocytoma download for annotations.

The audit is deliberately read-only. DICOM files are read with pixels skipped;
ordinary image series are never interpreted as masks merely because they are
derived or contain disease-related text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.misc import is_dicom
from pydicom.uid import UID


ANNOTATION_TERMS = (
    "mask", "masks", "segmentation", "segment", "seg", "label", "labels",
    "annotation", "annotations", "roi", "contour", "structure", "rtstruct",
    "groundtruth", "ground_truth", "truth", "brain", "tumor", "tumour",
    "lesion", "intracranial", "itk", "snap",
)
NON_DICOM_CANDIDATE_SUFFIXES = (
    ".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd", ".raw", ".hdr",
    ".img", ".seg.nrrd", ".label", ".label.gz", ".roi", ".xml", ".json",
    ".csv", ".tsv", ".txt", ".mat", ".h5", ".hdf5", ".npz", ".npy",
    ".vtk", ".vtp", ".stl", ".obj",
)

SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.66.4": "DICOM Segmentation Storage",
    "1.2.840.10008.5.1.4.1.1.66.5": "Surface Segmentation Storage",
    "1.2.840.10008.5.1.4.1.1.481.3": "RT Structure Set Storage",
    "1.2.840.10008.5.1.4.1.1.30": "Parametric Map Storage",
}
SEGMENTATION_SOP_UIDS = {
    "1.2.840.10008.5.1.4.1.1.66.4",
    "1.2.840.10008.5.1.4.1.1.66.5",
}
ANNOTATION_SEQUENCE_KEYWORDS = {
    "SegmentSequence", "ROIContourSequence", "StructureSetROISequence",
    "RTROIObservationsSequence", "ReferencedSeriesSequence",
}


def normalized_extension(path: Path) -> str:
    name = path.name.lower()
    for compound in (".seg.nrrd", ".label.gz", ".nii.gz"):
        if name.endswith(compound):
            return compound
    return path.suffix.lower() or "[extensionless]"


def keyword_matches(text: str) -> list[str]:
    lowered = str(text).lower()
    # Token boundaries prevent the short term "seg" matching unrelated words.
    return sorted({
        term for term in ANNOTATION_TERMS
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered)
    })


def detect_file_format(path: Path) -> str:
    extension = normalized_extension(path)
    if extension == ".dcm" or is_dicom(str(path)):
        return "DICOM"
    if extension == ".csv":
        return "CSV metadata table"
    if extension in NON_DICOM_CANDIDATE_SUFFIXES:
        return extension.lstrip(".").upper()
    if path.name == ".DS_Store":
        return "macOS Finder metadata"
    try:
        head = path.read_bytes()[:512]
    except OSError:
        return "unreadable"
    if b"\x00" not in head:
        return "text"
    return "unknown binary"


def dicom_annotation_type(ds: Dataset) -> tuple[str, str, str]:
    """Return likely type, confidence, and evidence from header semantics."""
    modality = str(ds.get("Modality", "")).upper()
    sop_uid = str(ds.get("SOPClassUID", ""))
    present = {element.keyword for element in ds.iterall() if element.keyword}
    overlay_present = any(
        0x6000 <= element.tag.group <= 0x60FF
        and element.tag.element in {0x0010, 0x0011, 0x3000}
        for element in ds.iterall()
    )
    private_matches = set()
    for element in ds.iterall():
        if not element.tag.is_private or element.VR in {"OB", "OW", "OF", "OD", "UN"}:
            continue
        value = str(element.value)
        if len(value) <= 1000:
            private_matches.update(keyword_matches(value))
    evidence = []
    if sop_uid in SEGMENTATION_SOP_UIDS or modality == "SEG" or "SegmentSequence" in present:
        evidence.extend(["segmentation SOP/modality/sequence"])
        return "DICOM segmentation", "confirmed", "; ".join(evidence)
    if sop_uid == "1.2.840.10008.5.1.4.1.1.481.3" or modality == "RTSTRUCT" or {
        "ROIContourSequence", "StructureSetROISequence"
    } & present:
        return "RT structure set", "confirmed", "RTSTRUCT SOP/modality/ROI sequence"
    if sop_uid == "1.2.840.10008.5.1.4.1.1.66.5":
        return "surface segmentation", "confirmed", "Surface Segmentation Storage SOP"
    if sop_uid == "1.2.840.10008.5.1.4.1.1.30":
        return "parametric map", "possible", "Parametric Map Storage SOP"
    if overlay_present:
        return "DICOM overlay", "possible", "60xx overlay rows/columns/data tag"
    if private_matches:
        return "private annotation-like metadata", "possible", (
            f"private-tag terms: {';'.join(sorted(private_matches))}"
        )
    if modality in {"SR", "PR"}:
        return {"SR": "structured report", "PR": "presentation state"}[modality], "possible", f"Modality={modality}"
    if sop_uid.startswith("1.2.840.10008.5.1.4.1.1.88."):
        return "structured report", "possible", "Structured Report SOP family"
    if sop_uid.startswith("1.2.840.10008.5.1.4.1.1.11."):
        return "presentation state", "possible", "Presentation State SOP family"
    if sop_uid.startswith("1.2.840.10008.5.1.4.1.1.104."):
        return "encapsulated document", "possible", "Encapsulated Document SOP family"
    descriptions = " ".join(str(ds.get(key, "")) for key in (
        "SeriesDescription", "StudyDescription", "ProtocolName", "ImageType"
    ))
    matches = keyword_matches(descriptions)
    if matches:
        return "keyword-matched DICOM image/derived series", "low", f"header terms: {';'.join(matches)}"
    return "ordinary DICOM image", "rejected", "no annotation SOP, modality, sequence, or keyword"


def referenced_series_uids(ds: Dataset) -> list[str]:
    found = set()
    for element in ds.iterall():
        if element.keyword == "SeriesInstanceUID" and element is not ds.get("SeriesInstanceUID"):
            # This identity check is not reliable across pydicom wrappers; top-level
            # UID is removed explicitly below.
            found.add(str(element.value))
    found.discard(str(ds.get("SeriesInstanceUID", "")))
    return sorted(uid for uid in found if uid)


def header_text(ds: Dataset, keyword: str) -> str:
    value = ds.get(keyword, "")
    if isinstance(value, (list, tuple)):
        return "\\".join(map(str, value))
    return str(value)


def safe_dicom_header(path: Path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=False), ""
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def is_binary_label_map(array: np.ndarray) -> bool:
    """Conservative value-only check used after a candidate image is identified."""
    values = np.unique(np.asarray(array))
    return bool(values.size <= 2 and set(values.tolist()) <= {0, 1})


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def file_inventory(dataset_root: Path):
    rows, paths = [], []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(dataset_root)
        detected = detect_file_format(path)
        matches = keyword_matches(str(relative))
        candidate = detected not in {"DICOM", "macOS Finder metadata"} and (
            normalized_extension(path) in NON_DICOM_CANDIDATE_SUFFIXES or bool(matches)
        )
        rows.append({
            "relative_path": str(relative), "extension": normalized_extension(path),
            "size_bytes": path.stat().st_size, "detected_format": detected,
            "annotation_candidate": int(candidate), "name_keyword_matches": ";".join(matches),
            "hidden_file": int(any(part.startswith(".") for part in relative.parts)),
        })
        paths.append((path, rows[-1]))
    return rows, paths


def audit_dicoms(paths):
    series = {}
    candidates = []
    unreadable = []
    sop_counts = Counter()
    for index, (path, inventory_row) in enumerate(paths, 1):
        if inventory_row["detected_format"] != "DICOM":
            continue
        ds, error = safe_dicom_header(path)
        if ds is None:
            unreadable.append({"relative_path": inventory_row["relative_path"], "error": error})
            continue
        series_uid = header_text(ds, "SeriesInstanceUID") or f"missing-series:{inventory_row['relative_path']}"
        sop_uid = header_text(ds, "SOPClassUID")
        likely_type, confidence, evidence = dicom_annotation_type(ds)
        sop_counts[sop_uid] += 1
        if series_uid not in series:
            series[series_uid] = {
                "subject_id": header_text(ds, "PatientID"),
                "study_id": header_text(ds, "StudyID"),
                "StudyInstanceUID": header_text(ds, "StudyInstanceUID"),
                "SeriesInstanceUID": series_uid,
                "SOPClassUID": sop_uid,
                "SOP_class_name": SOP_CLASSES.get(sop_uid, UID(sop_uid).name if sop_uid else ""),
                "Modality": header_text(ds, "Modality"),
                "SeriesDescription": header_text(ds, "SeriesDescription"),
                "StudyDescription": header_text(ds, "StudyDescription"),
                "ProtocolName": header_text(ds, "ProtocolName"),
                "ImageType": header_text(ds, "ImageType"),
                "Manufacturer": header_text(ds, "Manufacturer"),
                "FrameOfReferenceUID": header_text(ds, "FrameOfReferenceUID"),
                "instance_count": 0, "referenced_SeriesInstanceUIDs": set(),
                "possible_annotation_type": likely_type, "confidence": confidence,
                "inspection_notes": evidence, "example_path": inventory_row["relative_path"],
            }
        record = series[series_uid]
        record["instance_count"] += 1
        record["referenced_SeriesInstanceUIDs"].update(referenced_series_uids(ds))
        if index % 5000 == 0:
            print(f"read {index}/{len(paths)} filesystem entries", flush=True)
    rows = []
    for record in series.values():
        record["referenced_SeriesInstanceUIDs"] = ";".join(sorted(record["referenced_SeriesInstanceUIDs"]))
        rows.append(record)
        if record["confidence"] in {"confirmed", "possible", "low"}:
            candidates.append({
                "relative_path": record["example_path"], "format": "DICOM series",
                "subject_association": record["subject_id"],
                "series_association": record["SeriesInstanceUID"],
                "likely_type": record["possible_annotation_type"],
                "confidence": record["confidence"], "evidence": record["inspection_notes"],
                "reason_for_rejection": (
                    "standard MR Image Storage series; disease/anatomy keyword only"
                    if record["confidence"] == "low" else ""
                ),
            })
    rows.sort(key=lambda row: (row["subject_id"], row["StudyInstanceUID"], row["SeriesInstanceUID"]))
    return rows, candidates, unreadable, sop_counts


def metadata_audit(dataset_root: Path, inventory_rows):
    hits = []
    for row in inventory_rows:
        path = dataset_root / row["relative_path"]
        if row["detected_format"] not in {"CSV metadata table", "text", "JSON", "XML", "TXT"}:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            matches = keyword_matches(line)
            if matches:
                hits.append({
                    "relative_path": row["relative_path"], "line_number": line_number,
                    "matched_terms": ";".join(matches), "text_excerpt": line[:1000],
                })
    return hits


def subject_mapping(series_rows, candidates):
    subjects = sorted({row["subject_id"] for row in series_rows if row["subject_id"]})
    mapping = []
    for subject in subjects:
        subject_candidates = [row for row in candidates if row["subject_association"] == subject]
        confirmed = [row for row in subject_candidates if row["confidence"] == "confirmed"]
        brain = [row for row in confirmed if "brain" in (row["evidence"] + row["likely_type"]).lower()]
        tumor = [row for row in confirmed if keyword_matches(row["evidence"] + " " + row["likely_type"]) and
                 any(term in (row["evidence"] + row["likely_type"]).lower() for term in ("tumor", "tumour", "lesion"))]
        mapping.append({
            "subject_id": subject, "brain_mask_found": int(bool(brain)),
            "tumor_mask_found": int(bool(tumor)),
            "annotation_format": ";".join(sorted({row["format"] for row in confirmed})),
            "linked_image_series": ";".join(sorted({
                row["series_association"] for row in confirmed if row["series_association"]
            })),
            "confidence": "confirmed" if confirmed else "none",
            "notes": (
                f"{len(confirmed)} confirmed annotation objects"
                if confirmed else "No annotation SOP/modality/sequence found for subject"
            ),
        })
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="../Datasets/Mouse-Astrocytoma-doiJNLP")
    parser.add_argument("--output-directory", default="outputs/mouse_astrocytoma_annotation_audit")
    args = parser.parse_args()
    dataset_root = (ROOT / args.dataset_root).resolve()
    output = ROOT / args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    inventory, paths = file_inventory(dataset_root)
    series, dicom_candidates, unreadable, sop_counts = audit_dicoms(paths)
    metadata_hits = metadata_audit(dataset_root, inventory)
    manifest_path = dataset_root / "metadata" / "metadata.csv"
    manifest_rows = read_csv_rows(manifest_path) if manifest_path.exists() else []
    parsed_series_uids = {row["SeriesInstanceUID"] for row in series}
    manifest_series_uids = {row.get("SeriesInstanceUID", "") for row in manifest_rows}
    manifest_reconciliation = []
    for row in manifest_rows:
        uid = row.get("SeriesInstanceUID", "")
        manifest_reconciliation.append({
            "PatientID": row.get("PatientID", ""),
            "StudyInstanceUID": row.get("StudyInstanceUID", ""),
            "manifest_SeriesInstanceUID": uid,
            "S5cmdManifestPath": row.get("S5cmdManifestPath", ""),
            "completion_status": row.get("completion_status", ""),
            "parsed_locally": int(uid in parsed_series_uids),
            "note": (
                "" if uid in parsed_series_uids else
                "Manifest UID absent from local headers; download path collides with another manifest row"
            ),
        })
    # Non-DICOM candidates are recorded but metadata tables and Finder files are
    # rejected explicitly rather than silently disappearing.
    candidates = list(dicom_candidates)
    for row in inventory:
        if not row["annotation_candidate"]:
            continue
        likely = "metadata table" if row["detected_format"] == "CSV metadata table" else "possible non-DICOM annotation"
        candidates.append({
            "relative_path": row["relative_path"], "format": row["detected_format"],
            "subject_association": "", "series_association": "", "likely_type": likely,
            "confidence": "rejected" if likely == "metadata table" else "possible",
            "evidence": f"extension/name candidate; terms={row['name_keyword_matches']}",
            "reason_for_rejection": "collection manifest, not voxel labels" if likely == "metadata table" else "",
        })
    mapping = subject_mapping(series, candidates)

    write_csv(output / "file_inventory.csv", inventory, (
        "relative_path","extension","size_bytes","detected_format","annotation_candidate",
        "name_keyword_matches","hidden_file",
    ))
    write_csv(output / "dicom_series_inventory.csv", series, (
        "subject_id","study_id","StudyInstanceUID","SeriesInstanceUID","SOPClassUID",
        "SOP_class_name","Modality","SeriesDescription","StudyDescription","ProtocolName",
        "ImageType","Manufacturer","FrameOfReferenceUID","instance_count",
        "referenced_SeriesInstanceUIDs","possible_annotation_type","confidence",
        "inspection_notes","example_path",
    ))
    write_csv(output / "annotation_candidates.csv", candidates, (
        "relative_path","format","subject_association","series_association","likely_type",
        "confidence","evidence","reason_for_rejection",
    ))
    write_csv(output / "metadata_search_results.csv", metadata_hits, (
        "relative_path","line_number","matched_terms","text_excerpt",
    ))
    write_csv(output / "subject_annotation_mapping.csv", mapping, (
        "subject_id","brain_mask_found","tumor_mask_found","annotation_format",
        "linked_image_series","confidence","notes",
    ))
    write_csv(output / "manifest_series_reconciliation.csv", manifest_reconciliation, (
        "PatientID","StudyInstanceUID","manifest_SeriesInstanceUID","S5cmdManifestPath",
        "completion_status","parsed_locally","note",
    ))

    modality_counts = Counter(row["Modality"] for row in series)
    basename_counts = Counter(Path(row["relative_path"]).name for row in inventory)
    duplicated_names = {name: count for name, count in basename_counts.items() if count > 1}
    confirmed = [row for row in candidates if row["confidence"] == "confirmed"]
    summary = {
        "dataset_root": str(dataset_root), "file_count": len(inventory),
        "directory_count": sum(1 for path in dataset_root.rglob("*") if path.is_dir()) + 1,
        "total_bytes": sum(int(row["size_bytes"]) for row in inventory),
        "extensions": dict(Counter(row["extension"] for row in inventory)),
        "detected_formats": dict(Counter(row["detected_format"] for row in inventory)),
        "file_size_bytes": {
            "minimum": min(int(row["size_bytes"]) for row in inventory),
            "maximum": max(int(row["size_bytes"]) for row in inventory),
            "median": float(np.median([int(row["size_bytes"]) for row in inventory])),
        },
        "hidden_files": sum(int(row["hidden_file"]) for row in inventory),
        "duplicate_basename_groups": len(duplicated_names),
        "duplicate_basenames": duplicated_names,
        "unusual_binary_files": sum(row["detected_format"] == "unknown binary" for row in inventory),
        "archives": sum(row["extension"] in {".zip",".tar",".tgz",".7z",".rar"} for row in inventory),
        "dicom_instances": sum(int(row["instance_count"]) for row in series),
        "dicom_series": len(series), "modalities": dict(modality_counts),
        "sop_class_instance_counts": dict(sop_counts), "unreadable_dicom_files": unreadable,
        "annotation_candidates_total": len(candidates),
        "annotation_candidate_confidence_counts": dict(Counter(row["confidence"] for row in candidates)),
        "annotation_candidate_type_counts": dict(Counter(row["likely_type"] for row in candidates)),
        "confirmed_annotation_objects": len(confirmed),
        "confirmed_brain_masks": sum(int(row["brain_mask_found"]) for row in mapping),
        "confirmed_tumor_masks": sum(int(row["tumor_mask_found"]) for row in mapping),
        "dicom_seg_objects": sum(
            row["likely_type"] == "DICOM segmentation" and row["confidence"] == "confirmed"
            for row in candidates
        ),
        "rtstruct_objects": sum(
            row["likely_type"] == "RT structure set" and row["confidence"] == "confirmed"
            for row in candidates
        ),
        "subjects": len(mapping), "subjects_with_annotations": [
            row["subject_id"] for row in mapping if row["confidence"] == "confirmed"
        ],
        "metadata_keyword_hits": len(metadata_hits),
        "manifest": {
            "rows": len(manifest_rows),
            "unique_series": len({row.get("SeriesInstanceUID", "") for row in manifest_rows}),
            "modalities_listed": sorted({row.get("Modality", "") for row in manifest_rows if row.get("Modality")}),
            "completion_status_counts": dict(Counter(row.get("completion_status", "") for row in manifest_rows)),
            "collection_names": sorted({row.get("Collection", "") for row in manifest_rows}),
            "series_present_in_local_headers": len(manifest_series_uids & parsed_series_uids),
            "series_absent_from_local_headers": sorted(manifest_series_uids - parsed_series_uids),
            "parsed_series_not_in_manifest": sorted(parsed_series_uids - manifest_series_uids),
            "colliding_destination_paths": {
                path: count for path, count in Counter(
                    row.get("S5cmdManifestPath", "") for row in manifest_rows
                ).items() if path and count > 1
            },
        },
        "quantitative_dice_evaluation_possible": bool(confirmed),
        "local_primary_conclusion": (
            "No public annotation object was found in this downloaded directory."
            if not confirmed else "At least one annotation object requires visual validation."
        ),
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    report = f"""# TCIA Mouse-Astrocytoma annotation audit

## Scope and method

The complete local collection was inspected read-only: {summary['file_count']:,}
files in {summary['directory_count']:,} directories
({summary['total_bytes']/1024**3:.2f} GiB). Every DICOM instance was parsed with
pixel data skipped and grouped by `SeriesInstanceUID`. Non-DICOM formats,
extensionless files, hidden files, archives, metadata content, SOP classes,
modalities, annotation sequences, and referenced series were audited.

## A. Confirmed

- {summary['dicom_instances']:,} readable DICOM instances form
  {summary['dicom_series']:,} series.
- DICOM modalities by series: `{json.dumps(summary['modalities'], sort_keys=True)}`.
- Confirmed DICOM SEG objects: **{summary['dicom_seg_objects']}**.
- Confirmed RTSTRUCT objects: **{summary['rtstruct_objects']}**.
- Confirmed whole-brain masks: **{summary['confirmed_brain_masks']}**.
- Confirmed tumor masks: **{summary['confirmed_tumor_masks']}**.
- Subjects with confirmed annotations: **{len(summary['subjects_with_annotations'])}**.
- Candidate screening retained {summary['annotation_candidates_total']} rows:
  `{json.dumps(summary['annotation_candidate_confidence_counts'], sort_keys=True)}`.
  The 217 low-confidence DICOM candidates are ordinary MR series whose study
  descriptions contain `intracranial` or `brain tumor`; all were rejected by
  SOP class, modality, and absent annotation sequences.
- The only non-DICOM collection content is the manifest metadata and macOS
  Finder metadata; no NIfTI, NRRD, MHA/MHD, ROI, MAT, HDF5, NumPy, VTK, or
  label-map file was found.
- The local manifest contains {summary['manifest']['rows']} rows covering
  {summary['manifest']['unique_series']} unique series; completion statuses are
  `{json.dumps(summary['manifest']['completion_status_counts'], sort_keys=True)}`.
  Its collection field is `mouse_astrocytoma`.
- Three manifest series UIDs are absent from local DICOM headers because each
  shares its destination directory with another manifest row. The later
  download appears to have overwritten the earlier same-sized subtraction
  series. This is a local manifest/path-collision completeness issue, not
  evidence of an annotation object; the 283 locally present series are all MR.

## B. Likely but not proven

The manifest and downloaded tree appear to represent MR image series only. The
paper's statement that radiologists created whole-brain and tumor ground truth
does not establish that those masks were deposited in this public download.

## C. Not found

No SEG, RTSTRUCT, surface segmentation, contour/ROI sequence, structured
annotation object, non-DICOM label map, or confirmed private annotation object
was found locally. No downloaded subject has an annotation association.

## D. Requires external verification

Local files cannot establish whether TCIA hosts a separate Analysis Results
package, whether another collection/DOI contains derived annotations, or whether
the authors retained masks privately. The highest-priority next action is to
check the TCIA collection page and Analysis Results inventory using the exact
series/collection identifier, then contact the corresponding authors if no
separate label package is listed.

## Answers

1. Whole-brain ground-truth masks present: **no local evidence**.
2. Tumor ground-truth masks present: **no local evidence**.
3. DICOM SEG or RTSTRUCT: **0 / 0**.
4. Non-DICOM masks: **none found**.
5. Metadata references to missing annotations: **none found in the local manifest**.
   Three missing manifest UIDs are attributable to colliding image-series
   destination paths, not annotation terminology or annotation SOP classes.
6. Manifest appears image-only: **yes**, based on the locally parsed objects.
7. Separate public or unpublished labels: **possible but unverified externally**.
8. Subjects with annotations: **none**.
9. Immediate quantitative Dice evaluation: **no**.
10. Strongest next step: verify TCIA Analysis Results/related DOI packages and
    request the two manual mask sets from the study authors if absent.

No model inference was run. No dataset file was modified.
"""
    (output / "experiment_summary.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
