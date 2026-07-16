#!/usr/bin/env python3
"""Smoke-audit the pretrained RS2-Net encoder on one shared rodent MRI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as stream:
        # JSON is a strict subset of YAML. Keeping this file JSON-compatible
        # avoids adding PyYAML to the verified baseline environment.
        return json.load(stream)


def choose_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def center_crop_or_pad(volume, patch_size):
    """Create the baseline inference tile after its official preprocessing."""
    import torch.nn.functional as functional

    spatial = volume.shape[-3:]
    padding = []
    for current, target in reversed(list(zip(spatial, patch_size))):
        total = max(target - current, 0)
        padding.extend((total // 2, total - total // 2))
    volume = functional.pad(volume, padding)
    starts = [(size - target) // 2 for size, target in zip(volume.shape[-3:], patch_size)]
    return volume[
        :,
        :,
        starts[0] : starts[0] + patch_size[0],
        starts[1] : starts[1] + patch_size[1],
        starts[2] : starts[2] + patch_size[2],
    ]


def preprocess_with_baseline(image_path, paths):
    """Run the exact RS2/nnU-Net reader, crop, resampling, and z-normalization."""
    import json
    import numpy as np
    import torch

    # The adapter import registers the sibling project on sys.path.
    from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from RS2.utilities.plans_handling.plans_handler import PlansManager

    json_root = paths.baseline_project / "RS2" / "jsons"
    plans = json.loads((json_root / "plans.json").read_text())
    dataset = json.loads((json_root / "dataset.json").read_text())
    manager = PlansManager(plans)
    configuration = manager.get_configuration("3d_fullres")
    data, _, properties = DefaultPreprocessor(verbose=False).run_case(
        [str(image_path)], None, manager, configuration, dataset
    )
    tensor = torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0)
    return tensor, properties


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/rs2net_encoder.yaml")
    parser.add_argument("--input", type=Path, help="Optional NIfTI override")
    args = parser.parse_args()

    import nibabel as nib
    import torch
    from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths

    config = load_config(args.config.resolve())
    paths = RS2NetPaths.from_config(config)
    model_config = config["model"]
    image_path = args.input.resolve() if args.input else paths.dataset_root / config["sample_volume"]
    device = choose_device(config["inference"]["device"])

    if config["inference"].get("deterministic", True):
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)

    original_shape = tuple(int(value) for value in nib.load(str(image_path)).shape)
    adapter = RS2NetEncoderAdapter(paths, **{
        "image_size": model_config["image_size"],
        "in_channels": model_config["in_channels"],
        "out_channels": model_config["out_channels"],
        "feature_size": model_config["feature_size"],
    }).to(device)
    preprocessed, _ = preprocess_with_baseline(image_path, paths)
    model_input = center_crop_or_pad(preprocessed, tuple(model_config["image_size"])).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        features = adapter(model_input)

    print(f"input path: {image_path}")
    print(f"original volume shape: {original_shape}")
    print(f"preprocessed shape: {tuple(preprocessed.shape)}")
    print(f"encoder tile shape: {tuple(model_input.shape)}")
    print(f"device: {device}")
    print(f"checkpoint used: {paths.checkpoint}")
    for name, feature in features.items():
        print(f"{name}: {tuple(feature.shape)}")
    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    print(f"peak memory: {peak:.1f} MiB" if peak is not None else "peak memory: unavailable")


if __name__ == "__main__":
    main()
