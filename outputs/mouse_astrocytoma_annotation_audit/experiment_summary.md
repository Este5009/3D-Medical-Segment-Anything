# TCIA Mouse-Astrocytoma annotation audit

## Scope and method

The complete local collection was inspected read-only: 36,973
files in 382 directories
(2.03 GiB). Every DICOM instance was parsed with
pixel data skipped and grouped by `SeriesInstanceUID`. Non-DICOM formats,
extensionless files, hidden files, archives, metadata content, SOP classes,
modalities, annotation sequences, and referenced series were audited.

## A. Confirmed

- 36,966 readable DICOM instances form
  283 series.
- DICOM modalities by series: `{"MR": 283}`.
- Confirmed DICOM SEG objects: **0**.
- Confirmed RTSTRUCT objects: **0**.
- Confirmed whole-brain masks: **0**.
- Confirmed tumor masks: **0**.
- Subjects with confirmed annotations: **0**.
- Candidate screening retained 218 rows:
  `{"low": 217, "rejected": 1}`.
  The 217 low-confidence DICOM candidates are ordinary MR series whose study
  descriptions contain `intracranial` or `brain tumor`; all were rejected by
  SOP class, modality, and absent annotation sequences.
- The only non-DICOM collection content is the manifest metadata and macOS
  Finder metadata; no NIfTI, NRRD, MHA/MHD, ROI, MAT, HDF5, NumPy, VTK, or
  label-map file was found.
- The local manifest contains 286 rows covering
  286 unique series; completion statuses are
  `{"success": 286}`.
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
