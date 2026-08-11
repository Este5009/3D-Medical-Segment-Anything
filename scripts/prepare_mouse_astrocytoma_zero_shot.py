#!/usr/bin/env python3
"""Reconstruct and visually inventory every TCIA Mouse-Astrocytoma MR series."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def robust_normalize(array):
    foreground = array[np.isfinite(array)]
    low, high = np.percentile(foreground, (1, 99))
    return np.clip((array - low) / max(float(high - low), 1e-8), 0, 1)


def position_key(ds):
    if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds:
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        return round(float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), normal)), 5)
    return float(ds.get("SliceLocation", ds.get("InstanceNumber", 0)))


def series_headers(paths):
    records = []
    for path in paths:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        records.append({
            "path": path, "position": position_key(ds),
            "instance": int(ds.get("InstanceNumber", 0) or 0),
            "temporal": int(ds.get("TemporalPositionIdentifier", 0) or 0),
            "acquisition": int(ds.get("AcquisitionNumber", 0) or 0),
            "rows": int(ds.get("Rows", 0) or 0), "columns": int(ds.get("Columns", 0) or 0),
            "pixel_spacing": [float(x) for x in ds.get("PixelSpacing", [])],
            "slice_thickness": float(ds.get("SliceThickness", 0) or 0),
            "spacing_between_slices": float(ds.get("SpacingBetweenSlices", 0) or 0),
            "orientation": [float(x) for x in ds.get("ImageOrientationPatient", [])],
            "image_position": [float(x) for x in ds.get("ImagePositionPatient", [])],
            "image_type": "\\".join(map(str, ds.get("ImageType", []))),
            "series_description": str(ds.get("SeriesDescription", "")),
            "protocol_name": str(ds.get("ProtocolName", "")),
            "study_description": str(ds.get("StudyDescription", "")),
            "manufacturer": str(ds.get("Manufacturer", "")),
        })
    return records


def first_spatial_volume(records):
    """Select one image per physical slice for QC; repeated positions imply dynamics."""
    by_position = defaultdict(list)
    for record in records:
        by_position[record["position"]].append(record)
    selected = []
    for position in sorted(by_position):
        group = sorted(by_position[position], key=lambda r: (r["temporal"], r["acquisition"], r["instance"]))
        selected.append(group[0])
    return selected, max(len(group) for group in by_position.values())


def load_volume(records):
    slices = []
    for record in records:
        ds = pydicom.dcmread(str(record["path"]))
        array = ds.pixel_array.astype(np.float32)
        array = array * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
        slices.append(array)
    return np.stack(slices, axis=-1)


def initial_qc_class(description, protocol, image_type, unique_slices, repetitions):
    text = f"{description} {protocol} {image_type}".lower()
    if "subtr" in text:
        return "subtraction", "subtraction series"
    if "dyn" in text or repetitions > 2:
        return "dynamic", f"{repetitions} repeated images per spatial position"
    if unique_slices < 8:
        return "incomplete", f"only {unique_slices} unique spatial slices"
    if "tse_100" in text or ("postc" in text and "cor" in text) or ("prec" in text and "cor" in text):
        return "primary anatomical", "full TSE or coronal T1 anatomical acquisition"
    if "prefa" in text:
        return "secondary anatomical", "low-flip-angle anatomical reference"
    if "survey" in text or "scout" in text or "localizer" in text:
        return "scout", "localizer/scout description"
    if "derived" in text:
        return "derived", "DICOM ImageType marks derived data"
    return "manual review", "metadata alone is insufficient"


def save_contact_sheet(volume, destination, title, maximum_panels=64):
    count = volume.shape[2]
    indices = np.linspace(0, count - 1, min(count, maximum_panels)).round().astype(int)
    columns = 8; rows = math.ceil(len(indices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 2 * rows), squeeze=False)
    normalized = robust_normalize(volume)
    for axis, index in zip(axes.flat, indices):
        axis.imshow(normalized[:, :, index].T, cmap="gray", origin="lower")
        axis.set_title(f"{index}", fontsize=7); axis.axis("off")
    for axis in axes.flat[len(indices):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=10)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120, bbox_inches="tight"); plt.close(figure)


def save_overviews(series_rows, output):
    for page_start in range(0, len(series_rows), 48):
        page = series_rows[page_start:page_start + 48]
        figure, axes = plt.subplots(6, 8, figsize=(20, 15), constrained_layout=True)
        for axis, row in zip(axes.flat, page):
            image = plt.imread(row["contact_sheet_path"])
            axis.imshow(image); axis.axis("off")
            axis.set_title(
                f"{row['subject_id'][-15:]}\n{row['SeriesDescription'][:25]}\n{row['qc_class']}",
                fontsize=5,
            )
        for axis in axes.flat[len(page):]: axis.axis("off")
        figure.savefig(output / "qc_overviews" / f"page_{page_start//48+1:02d}.png",
                       dpi=120, bbox_inches="tight")
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="../Datasets/Mouse-Astrocytoma-doiJNLP")
    parser.add_argument("--output-directory", default="outputs/mouse_astrocytoma_zero_shot")
    args = parser.parse_args()
    dataset_root = (ROOT / args.dataset_root).resolve()
    output = ROOT / args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    audit_series = read_csv(ROOT / "outputs/mouse_astrocytoma_annotation_audit/dicom_series_inventory.csv")
    all_paths = defaultdict(list)
    for path in dataset_root.rglob("*.dcm"):
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=["SeriesInstanceUID"])
        all_paths[str(ds.SeriesInstanceUID)].append(path)
    series_rows = []
    for index, row in enumerate(audit_series, 1):
        uid = row["SeriesInstanceUID"]; paths = all_paths[uid]
        headers = series_headers(paths); selected, repetitions = first_spatial_volume(headers)
        try:
            volume = load_volume(selected); reconstruction = "success"
        except Exception as error:
            volume = np.zeros((64, 64, 1), dtype=np.float32)
            reconstruction = f"failed: {type(error).__name__}: {error}"
        first = headers[0]
        qc_class, reason = initial_qc_class(
            first["series_description"], first["protocol_name"], first["image_type"],
            len(selected), repetitions,
        )
        safe = f"{index:03d}_{uid[-16:].replace('.', '_')}"
        series_dir = output / "series_qc" / row["subject_id"] / safe
        contact = series_dir / "contact_sheet.png"
        save_contact_sheet(
            volume, contact,
            f"{row['subject_id']} | {first['series_description']} | {qc_class} | "
            f"{volume.shape} | repetitions {repetitions}",
        )
        serializable_first = {
            key: str(value) if isinstance(value, Path) else value for key, value in first.items()
        }
        metadata = {
            **serializable_first, "subject_id": row["subject_id"], "StudyInstanceUID": row["StudyInstanceUID"],
            "SeriesInstanceUID": uid, "instances": len(paths), "unique_spatial_slices": len(selected),
            "maximum_position_repetitions": repetitions, "reconstructed_shape": list(volume.shape),
            "reconstruction_status": reconstruction, "qc_class": qc_class,
            "qc_reason": reason, "visual_review_status": "contact sheet generated for review",
        }
        series_dir.mkdir(parents=True, exist_ok=True)
        (series_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        series_rows.append({
            "subject_id": row["subject_id"], "StudyInstanceUID": row["StudyInstanceUID"],
            "SeriesInstanceUID": uid, "SeriesDescription": first["series_description"],
            "ProtocolName": first["protocol_name"], "ImageType": first["image_type"],
            "rows": first["rows"], "columns": first["columns"],
            "pixel_spacing": "\\".join(map(str, first["pixel_spacing"])),
            "slice_thickness": first["slice_thickness"],
            "spacing_between_slices": first["spacing_between_slices"],
            "instances": len(paths), "unique_spatial_slices": len(selected),
            "maximum_position_repetitions": repetitions,
            "reconstructed_shape": "x".join(map(str, volume.shape)),
            "qc_class": qc_class, "qc_reason": reason,
            "visual_review_status": "contact sheet generated",
            "contact_sheet_path": str(contact),
            "metadata_path": str(series_dir / "metadata.json"),
            "example_dicom_path": str(selected[0]["path"]),
        })
        print(f"QC {index}/{len(audit_series)} {row['subject_id']} {first['series_description']}", flush=True)
    write_csv(output / "series_inventory.csv", series_rows)
    subjects = []
    for subject in sorted({row["subject_id"] for row in series_rows}):
        subset = [row for row in series_rows if row["subject_id"] == subject]
        counts = Counter(row["qc_class"] for row in subset)
        subjects.append({
            "subject_id": subject, "series_count": len(subset),
            **{f"{name.replace(' ','_')}_series": counts[name] for name in (
                "primary anatomical","secondary anatomical","dynamic","subtraction","derived",
                "scout","incomplete","duplicate","unsuitable","manual review"
            )},
        })
    write_csv(output / "dataset_inventory.csv", subjects)
    (output / "qc_overviews").mkdir(exist_ok=True)
    save_overviews(series_rows, output)
    summary = {
        "subjects": len(subjects), "series": len(series_rows),
        "qc_class_counts": dict(Counter(row["qc_class"] for row in series_rows)),
        "reconstruction_failures": sum("failed" in json.loads(Path(row["metadata_path"]).read_text())["reconstruction_status"] for row in series_rows),
        "overview_pages": math.ceil(len(series_rows) / 48),
    }
    (output / "dataset_qc_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
