#!/usr/bin/env python3
"""Overfit one learned query on two geometry-verified CAMRI rat subjects."""

from __future__ import annotations

import csv
import copy
import json
import random
import re
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from models.query_mask_decoder import (
    FrozenEncoderQueryModel,
    OneQueryMaskDecoder,
    dice_bce_loss,
    volumetric_dice,
)
from models.rs2net_encoder_adapter import RS2NetEncoderAdapter, RS2NetPaths


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_device() -> torch.device:
    """Required priority: Apple MPS, then CUDA, then CPU."""
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_pairs(config: dict, dataset_root: Path):
    masks = list((dataset_root / config["mask_directory"]).glob("*.nii.gz"))
    pairs = []
    for subject in config["subjects"]:
        image = dataset_root / config["image_pattern"].format(subject=subject)
        matching_masks = [path for path in masks if re.search(rf"sub-{re.escape(subject)}(?:_|$)", path.name)]
        if not image.is_file() or len(matching_masks) != 1:
            raise FileNotFoundError(
                f"Subject {subject}: expected one image and one mask, found image={image.is_file()}, "
                f"masks={len(matching_masks)}"
            )
        pairs.append((subject, image, matching_masks[0]))
    return pairs


def verify_geometry(image_path: Path, mask_path: Path) -> dict:
    import nibabel as nib

    image = nib.load(str(image_path))
    mask = nib.load(str(mask_path))
    if image.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {image.shape} versus {mask.shape}")
    if not np.allclose(image.affine, mask.affine, atol=1e-5):
        raise ValueError(f"Affine mismatch: {image_path} versus {mask_path}")
    return {
        "original_shape": list(image.shape),
        "voxel_spacing": list(map(float, image.header.get_zooms()[:3])),
        "affines_match": True,
    }


def _pad_and_center_crop(array: torch.Tensor, size):
    padding = []
    for current, target in reversed(list(zip(array.shape[-3:], size))):
        total = max(target - current, 0)
        padding.extend((total // 2, total - total // 2))
    array = F.pad(array, padding)
    starts = [(current - target) // 2 for current, target in zip(array.shape[-3:], size)]
    return array[
        ...,
        starts[0] : starts[0] + size[0],
        starts[1] : starts[1] + size[1],
        starts[2] : starts[2] + size[2],
    ]


def preprocess_pair(image_path: Path, mask_path: Path, paths: RS2NetPaths, tile_size):
    """Use the baseline preprocessor jointly so image/mask transforms stay aligned."""
    from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from RS2.utilities.plans_handling.plans_handler import PlansManager

    json_root = paths.baseline_project / "RS2" / "jsons"
    plans = load_json(json_root / "plans.json")
    dataset = load_json(json_root / "dataset.json")
    manager = PlansManager(plans)
    configuration = manager.get_configuration("3d_fullres")
    data, segmentation, properties = DefaultPreprocessor(verbose=False).run_case(
        [str(image_path)], str(mask_path), manager, configuration, dataset
    )
    image_tensor = torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0)
    mask_tensor = torch.from_numpy((np.asarray(segmentation) > 0).astype(np.float32)).unsqueeze(0)
    return (
        _pad_and_center_crop(image_tensor, tile_size),
        _pad_and_center_crop(mask_tensor, tile_size),
        list(image_tensor.shape),
        properties,
    )


def save_prediction_and_figure(output_dir, subject, image, target, logits, spacing):
    import imageio.v3 as iio
    import nibabel as nib

    probability = logits.sigmoid().detach().cpu()[0, 0].numpy()
    prediction = (probability >= 0.5).astype(np.uint8)
    target_np = target.detach().cpu()[0, 0].numpy()
    image_np = image.detach().cpu()[0, 0].numpy()

    # Files are explicitly in the model's transposed/resampled tile space.
    affine = np.diag([*map(float, spacing), 1.0])
    nib.save(nib.Nifti1Image(prediction, affine), output_dir / f"sub-{subject}_prediction_model_space.nii.gz")

    depth_scores = target_np.sum(axis=(1, 2))
    depth = int(depth_scores.argmax())
    image_slice = image_np[depth]
    image_slice = (image_slice - image_slice.min()) / max(float(np.ptp(image_slice)), 1e-8)
    panels = [image_slice, target_np[depth], probability[depth]]
    panels = [(np.clip(panel, 0, 1) * 255).astype(np.uint8) for panel in panels]
    separator = np.full((panels[0].shape[0], 4), 255, dtype=np.uint8)
    montage = np.concatenate((panels[0], separator, panels[1], separator, panels[2]), axis=1)
    iio.imwrite(output_dir / f"sub-{subject}_qualitative.png", montage)


def evaluate(model, samples, device):
    model.eval()
    results = []
    with torch.inference_mode():
        for sample in samples:
            target = sample["mask"].to(device)
            features = {name: value.to(device) for name, value in sample["features"].items()}
            logits = model.decode(features, output_size=target.shape[-3:])
            results.append((sample, logits.cpu(), volumetric_dice(logits, target)))
    return results


def main() -> None:
    config_path = REPO_ROOT / "configs/query_decoder_overfit.yaml"
    config = load_json(config_path)
    encoder_config = load_json(REPO_ROOT / config["encoder_config"])
    paths = RS2NetPaths.from_config(encoder_config)
    paths.validate()
    output_dir = REPO_ROOT / config["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device()
    tile_size = tuple(encoder_config["model"]["image_size"])

    encoder = RS2NetEncoderAdapter(paths, **{
        "image_size": tile_size,
        "in_channels": encoder_config["model"]["in_channels"],
        "out_channels": encoder_config["model"]["out_channels"],
        "feature_size": encoder_config["model"]["feature_size"],
    })
    decoder = OneQueryMaskDecoder(config["embedding_dim"], config["num_heads"])
    model = FrozenEncoderQueryModel(encoder, decoder).to(device)

    samples = []
    geometry = {}
    for subject, image_path, mask_path in resolve_pairs(config, paths.dataset_root):
        geometry[subject] = verify_geometry(image_path, mask_path)
        image, mask, preprocessed_shape, properties = preprocess_pair(image_path, mask_path, paths, tile_size)
        geometry[subject]["preprocessed_shape"] = preprocessed_shape
        geometry[subject]["tile_shape"] = list(image.shape)
        geometry[subject]["image"] = str(image_path)
        geometry[subject]["mask"] = str(mask_path)
        samples.append({"subject": subject, "image": image, "mask": mask})

    # The set contains only two subjects. Keep the two selected frozen feature
    # maps in RAM for this process so epochs test the decoder rather than repeat
    # identical encoder work. Nothing is cached to disk or for the full dataset.
    model.eval()
    for sample in samples:
        encoded = model.encode(sample["image"].to(device))
        sample["features"] = {
            name: encoded[name].detach().cpu() for name in ("level1", "level4")
        }

    optimizer = torch.optim.AdamW(
        model.decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    initial_results = evaluate(model, samples, device)
    starting_dice = float(np.mean([item[2] for item in initial_results]))
    best_dice = starting_dice
    best_epoch = 0
    best_decoder_state = copy.deepcopy(model.decoder.state_dict())
    epochs_without_improvement = 0
    history = []
    peak_memory = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        epoch_losses = []
        for sample in samples:  # batch size is exactly one; no dataset cache is created.
            target = sample["mask"].to(device)
            features = {name: value.to(device) for name, value in sample["features"].items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model.decode(features, output_size=target.shape[-3:])
            loss, _ = dice_bce_loss(logits, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        epoch_results = evaluate(model, samples, device)
        mean_dice = float(np.mean([item[2] for item in epoch_results]))
        history.append({"epoch": epoch, "loss": float(np.mean(epoch_losses)), "mean_dice": mean_dice})
        print(f"epoch={epoch:03d} loss={history[-1]['loss']:.6f} mean_dice={mean_dice:.6f}")

        if mean_dice > best_dice + float(config["minimum_improvement"]):
            best_dice = mean_dice
            best_epoch = epoch
            best_decoder_state = copy.deepcopy(model.decoder.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if mean_dice >= float(config["overfit_dice"]):
            break
        if epochs_without_improvement >= int(config["early_stop_patience"]):
            break

    # Write once, after training, so interrupted/stale runs cannot leave a
    # checkpoint whose metadata and parameters come from different epochs.
    torch.save(
        {"decoder_state_dict": best_decoder_state, "config": config, "epoch": best_epoch, "dice": best_dice},
        output_dir / "best_checkpoint.pt",
    )
    checkpoint = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    final_results = evaluate(model, samples, device)
    final_dice = float(np.mean([item[2] for item in final_results]))

    with (output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "mean_dice"])
        writer.writeheader()
        writer.writerows(history)
    for sample, logits, dice in final_results:
        save_prediction_and_figure(
            output_dir,
            sample["subject"],
            sample["image"],
            sample["mask"],
            logits,
            spacing=(0.25, 0.20000000298, 0.15999999642),
        )
        geometry[sample["subject"]]["final_dice"] = dice

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    elif device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        peak_memory = torch.mps.driver_allocated_memory() / 1024**2
    elif sys.platform == "darwin":
        # macOS reports ru_maxrss in bytes. This is process RSS, not tensor-only.
        peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    encoder_gradients_disabled = all(
        not parameter.requires_grad and parameter.grad is None for parameter in model.encoder.parameters()
    )
    summary = {
        "architecture": "one query; level4 cross-attention; level1 dot-product mask",
        "trainable_parameters": model.trainable_parameter_count(),
        "subjects": list(config["subjects"]),
        "device": str(device),
        "encoder_gradients_disabled": encoder_gradients_disabled,
        "starting_mean_dice": starting_dice,
        "final_mean_dice": final_dice,
        "best_epoch": int(checkpoint["epoch"]),
        "overfit_threshold": config["overfit_dice"],
        "successfully_overfit": final_dice >= float(config["overfit_dice"]),
        "epochs_run": len(history),
        "peak_memory_mib": peak_memory,
        "elapsed_seconds": time.time() - start_time,
        "geometry": geometry,
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
