"""Synthetic tests for the TCIA annotation audit; no real dataset required."""
from __future__ import annotations

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

from scripts.audit_mouse_astrocytoma_annotations import (
    audit_dicoms,
    detect_file_format,
    dicom_annotation_type,
    is_binary_label_map,
    keyword_matches,
    normalized_extension,
    referenced_series_uids,
    safe_dicom_header,
    subject_mapping,
)


def dataset(sop, modality):
    ds = Dataset(); ds.SOPClassUID = sop; ds.Modality = modality
    ds.SeriesInstanceUID = "1.2.3"; ds.PatientID = "mouse-1"
    return ds


def test_compound_file_format_detection(tmp_path):
    assert normalized_extension(tmp_path / "mask.seg.nrrd") == ".seg.nrrd"
    assert normalized_extension(tmp_path / "mask.nii.gz") == ".nii.gz"


def test_dicom_seg_detection():
    result = dicom_annotation_type(dataset("1.2.840.10008.5.1.4.1.1.66.4", "SEG"))
    assert result[:2] == ("DICOM segmentation", "confirmed")


def test_rtstruct_detection():
    result = dicom_annotation_type(dataset("1.2.840.10008.5.1.4.1.1.481.3", "RTSTRUCT"))
    assert result[:2] == ("RT structure set", "confirmed")


def test_ordinary_mr_is_rejected_as_mask():
    result = dicom_annotation_type(dataset("1.2.840.10008.5.1.4.1.1.4", "MR"))
    assert result[:2] == ("ordinary DICOM image", "rejected")


def test_referenced_series_parsing():
    ds = dataset("1.2.840.10008.5.1.4.1.1.66.4", "SEG")
    item = Dataset(); item.SeriesInstanceUID = "9.8.7"
    ds.ReferencedSeriesSequence = [item]
    assert referenced_series_uids(ds) == ["9.8.7"]


def test_extensionless_dicom_detection(tmp_path):
    path = tmp_path / "instance"
    meta = FileMetaDataset(); meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"; ds.save_as(path)
    assert detect_file_format(path) == "DICOM"


def test_annotation_keyword_boundaries():
    assert keyword_matches("manual tumor ROI contour") == ["contour", "roi", "tumor"]
    assert "seg" not in keyword_matches("segmentation")


def test_subject_annotation_mapping():
    series = [{"subject_id": "mouse-1"}]
    candidates = [{
        "subject_association": "mouse-1", "confidence": "confirmed",
        "evidence": "brain tumor", "likely_type": "DICOM segmentation",
        "format": "DICOM", "series_association": "1.2.3",
    }]
    mapped = subject_mapping(series, candidates)[0]
    assert mapped["brain_mask_found"] == 1 and mapped["tumor_mask_found"] == 1


def test_unreadable_file_is_not_misclassified(tmp_path):
    path = tmp_path / "broken.bin"; path.write_bytes(b"\x00\x01garbage")
    assert detect_file_format(path) == "unknown binary"
    header, error = safe_dicom_header(path)
    assert header is None and error


def test_binary_label_map_detection_synthetic():
    array = np.asarray([0, 1, 1, 0], dtype=np.uint8)
    assert is_binary_label_map(array)
    assert not is_binary_label_map(np.asarray([0, 1, 2], dtype=np.uint8))


def test_instances_group_by_series_uid(tmp_path):
    entries = []
    for index in range(2):
        path = tmp_path / f"{index}.dcm"
        meta = FileMetaDataset(); meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"; ds.SOPInstanceUID = f"1.2.3.{index}"
        ds.SeriesInstanceUID = "1.2.3"; ds.StudyInstanceUID = "1.2"; ds.PatientID = "mouse-1"
        ds.Modality = "MR"; ds.save_as(path)
        entries.append((path, {
            "detected_format": "DICOM", "relative_path": path.name,
        }))
    series, _, unreadable, _ = audit_dicoms(entries)
    assert len(series) == 1 and series[0]["instance_count"] == 2 and not unreadable
