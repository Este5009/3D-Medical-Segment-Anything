#!/usr/bin/env python3
"""Untouched-test comparison of baseline, 3D filter, and selected grouping model."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_external_holdout import metrics as native_metrics
from evaluate_mouse_boundary_adaptation import make_comparison_figure, run_records, summarize, write_csv
from models.query_conditioned_grouping import (
    FrozenBaselineWithGrouping, QueryConditionedGroupingDecoder, largest_component_3d,
)
from models.query_mask_decoder import FrozenEncoderQueryModel, MultiScaleOneQueryMaskDecoder
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths
from train_query_decoder_overfit import choose_device, load_json


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def construct_model(config, checkpoint, paths, device):
    baseline_payload = torch.load(ROOT / config["baseline_checkpoint"], map_location="cpu", weights_only=False)
    decoder = MultiScaleOneQueryMaskDecoder(32, 4)
    decoder.load_state_dict(baseline_payload["decoder_state_dict"], strict=True)
    baseline = FrozenEncoderQueryModel(
        RS2NetEncoderAdapter(paths, image_size=tuple(config["tile_size"]),
                             in_channels=1, out_channels=1, feature_size=48),
        decoder,
    )
    grouping = QueryConditionedGroupingDecoder(
        feature_channels=config["feature_channels"], hidden_channels=config["hidden_channels"],
        use_coordinates=checkpoint["options"]["use_coordinates"],
        max_correction=config["max_correction"],
    )
    grouping.load_state_dict(checkpoint["grouping_state_dict"], strict=True)
    model = FrozenBaselineWithGrouping(
        baseline, grouping, use_query=checkpoint["options"]["use_query"]
    ).to(device).eval()
    assert not any(parameter.requires_grad for parameter in model.baseline.parameters())
    return model


def deterministic_results(baseline_rows, output):
    rows = []
    destination = output / "native_predictions" / "deterministic_component_filter"
    destination.mkdir(parents=True, exist_ok=True)
    for row in baseline_rows:
        source = nib.load(row["prediction_path"])
        baseline = np.asarray(source.dataobj) > 0
        prediction = largest_component_3d(baseline)
        path = destination / f"{row['domain']}_{row['subject']}_prediction.nii.gz"
        nib.save(nib.Nifti1Image(prediction, source.affine, source.header), path)
        target_obj = nib.load(row["ground_truth_path"])
        target = np.asarray(target_obj.dataobj) > 0
        spacing = tuple(map(float, target_obj.header.get_zooms()[:3]))
        measured = native_metrics(prediction.astype(bool), target, spacing)
        rows.append({
            **row, "condition": "deterministic_component_filter",
            "prediction_path": str(path), **measured,
            "volume_ratio": float(prediction.sum() / max(target.sum(), 1)),
            "connected_components": 1 if prediction.any() else 0,
        })
    return rows


def baseline_rows():
    rows = []
    for domain, filename in (
        ("CAMRI", "camri_subject_dice.csv"), ("Mouse", "mouse_subject_dice.csv")
    ):
        for row in read_csv(ROOT / "outputs/mixed_domain_anatomical_training/fast_evaluation" / filename):
            pred = np.asarray(nib.load(row["prediction_path"]).dataobj) > 0
            target_obj = nib.load(row["ground_truth_path"])
            target = np.asarray(target_obj.dataobj) > 0
            measured = native_metrics(pred, target, tuple(map(float, target_obj.header.get_zooms()[:3])))
            labels = __import__("scipy").ndimage.label(pred, structure=np.ones((3, 3, 3)))[1]
            rows.append({
                **row, "domain": domain, "condition": "baseline",
                **measured, "hd95_mm": measured["hd95"],
                "volume_ratio": float(pred.sum() / max(target.sum(), 1)),
                "connected_components": int(labels), "inference_seconds": float("nan"),
            })
    return rows


def selected_cases(comparison):
    selected = []
    for domain in ("CAMRI", "Mouse"):
        domain_rows = sorted((r for r in comparison if r["domain"] == domain), key=lambda r: r["learned_dice"])
        selected.extend([
            ("worst", domain_rows[0]),
            ("median", domain_rows[len(domain_rows) // 2]),
            ("good", domain_rows[-1]),
        ])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/query_conditioned_3d_grouping.yaml")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    output = ROOT / config["output_directory"]
    checkpoint = torch.load(output / "best_grouping_checkpoint.pt", map_location="cpu", weights_only=False)
    split = load_json(ROOT / config["baseline_output"] / "split.json")
    camri_source = {r["subject"]: r for r in read_csv(ROOT / config["camri_metrics"])}
    mouse_source = {r["scan_id"]: r for r in read_csv(ROOT / config["mouse_metrics"])}
    camri = []
    for subject in split["camri"]["test"]:
        row = dict(camri_source[subject]); row["ground_truth_path"] = row["mask_path"]; camri.append(row)
    mouse = [mouse_source[scan] for scan in split["mouse"]["test"]["scans"]]
    paths = RS2NetPaths.from_config(load_json(ROOT / config["encoder_config"]))
    device = choose_device()
    model = construct_model(config, checkpoint, paths, device)
    learned_camri, _ = run_records(model, camri, paths, config, output, device, "learned_camri")
    learned_mouse, _ = run_records(model, mouse, paths, config, output, device, "learned_mouse")
    learned = []
    for domain, values in (("CAMRI", learned_camri), ("Mouse", learned_mouse)):
        for row in values:
            learned.append({**row, "domain": domain, "subject": row["scan_id"], "condition": "learned"})

    baseline = baseline_rows()
    deterministic = deterministic_results(baseline, output)
    by_learned = {(r["domain"], r["subject"]): r for r in learned}
    by_filter = {(r["domain"], r["subject"]): r for r in deterministic}
    comparison = []
    for base in baseline:
        key = (base["domain"], base["subject"]); learned_row = by_learned[key]; filtered = by_filter[key]
        comparison.append({
            "domain": key[0], "subject": key[1],
            **{f"baseline_{k}": base[k] for k in ("dice","precision","recall","hd95_mm","false_positives","false_negatives","volume_ratio","connected_components")},
            **{f"filter_{k}": filtered[k] for k in ("dice","precision","recall","hd95","false_positives","false_negatives","volume_ratio","connected_components")},
            **{f"learned_{k}": learned_row[k] for k in ("dice","precision","recall","hd95_mm","false_positives","false_negatives","volume_ratio","connected_components","inference_seconds")},
            "image_path": base["image_path"], "ground_truth_path": base["ground_truth_path"],
            "baseline_prediction_path": base["prediction_path"],
            "filter_prediction_path": filtered["prediction_path"],
            "learned_prediction_path": learned_row["prediction_path"],
            "learned_probability_path": learned_row["probability_path"],
        })
    write_csv(output / "test_subject_comparison.csv", comparison)
    ranked = sorted(comparison, key=lambda r: float(r["learned_dice"]) - float(r["baseline_dice"]))
    write_csv(output / "ranked_subject_changes.csv", ranked)

    figure_manifest = []
    for category, row in selected_cases(comparison):
        figure_path = output / "qualitative_figures" / f"{row['domain']}_{category}_{row['subject']}.png"
        figure_row = {
            "subject_id": row["subject"], "baseline_dice": row["baseline_dice"],
            "adapted_dice": row["learned_dice"], "baseline_precision": row["baseline_precision"],
            "adapted_precision": row["learned_precision"], "baseline_recall": row["baseline_recall"],
            "adapted_recall": row["learned_recall"], "baseline_volume_ratio": row["baseline_volume_ratio"],
            "adapted_volume_ratio": row["learned_volume_ratio"],
            "baseline_prediction_path": row["baseline_prediction_path"],
            "adapted_prediction_path": row["learned_prediction_path"],
            "probability_path": row["learned_probability_path"], "image_path": row["image_path"],
            "ground_truth_path": row["ground_truth_path"],
        }
        make_comparison_figure(figure_row, figure_path)
        figure_manifest.append({"domain": row["domain"], "category": category, "subject": row["subject"], "path": str(figure_path)})
    write_csv(output / "qualitative_figures" / "manifest.csv", figure_manifest)

    summaries = {}
    for domain in ("CAMRI", "Mouse"):
        subset = [r for r in comparison if r["domain"] == domain]
        summaries[domain] = {}
        for condition, prefix in (("baseline","baseline"),("deterministic_filter","filter"),("learned","learned")):
            summaries[domain][condition] = {
                key: float(np.mean([float(r[f"{prefix}_{key}"]) for r in subset]))
                for key in ("dice","precision","recall","false_positives","false_negatives","volume_ratio","connected_components")
            }
            hdkey = "hd95_mm" if prefix != "filter" else "hd95"
            summaries[domain][condition]["hd95_mm"] = float(np.mean([float(r[f"{prefix}_{hdkey}"]) for r in subset]))
    summary = {
        "device": str(device), "test_labels_used_for": "metrics and figures only",
        "selected_ablation": checkpoint["ablation"], "parameter_count": checkpoint["parameter_count"],
        "test_counts": {"CAMRI": len(camri), "Mouse": len(mouse)}, "metrics": summaries,
        "absolute_worst_learned": min(comparison, key=lambda r: float(r["learned_dice"]))["subject"],
        "query_conditioning_selected": checkpoint["options"]["use_query"],
    }
    (output / "test_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
