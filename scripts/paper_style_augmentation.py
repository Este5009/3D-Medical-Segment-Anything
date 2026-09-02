#!/usr/bin/env python3
"""Richer, paper-style training augmentation, applied to the RAW image/target
pair BEFORE the frozen encoder -- not the cached feature-space trick every
prior decoder-only experiment in this project used.

Motivation: outputs/unetr_style_level0_full_width_decoder/ improved Mouse
boundary metrics in aggregate, but visual inspection found small, locally
confident false-positive "bulbs" escaping the true contour on some subjects
-- a signature consistent with a higher-capacity decoder (444,449 params)
memorizing specific patterns from only 39 real training images rather than
learning boundary features that generalize. The RS2-Net paper's own training
recipe uses rotation, zoom, Gaussian blur/noise, brightness/contrast,
simulated low-resolution, and gamma specifically to combat this on a (much
larger, but still augmented) dataset. This module reproduces that list.

Two transforms (simulated-low-resolution, gamma) are implemented directly
with torch ops rather than monai.transforms.RandSimulateLowResolutiond /
RandGammad, because the local dev environment's MONAI (1.2.0) predates
RandSimulateLowResolutiond -- this keeps the module portable across the
local (1.2.0) and pod (1.6.0) MONAI versions without relying on a
version-specific API.

Applied to (image, target) as a matched pair: spatial transforms (rotate,
zoom, flip) use the same random parameters for both, target always
nearest-neighbor interpolated (stays exactly binary); intensity transforms
(blur, noise, contrast, gamma, simulated-low-res) apply to the image only.
"""
from __future__ import annotations
import random
import torch
import torch.nn.functional as F
from monai.transforms import RandRotated, RandZoomd, RandGaussianSmoothd, RandGaussianNoised, RandAdjustContrastd

DEFAULT_SPEC = {
    "flip_probability": 0.5,
    "rotate_probability": 0.2, "rotate_range_deg": 15,
    "zoom_probability": 0.2, "zoom_range": (0.85, 1.15),
    "blur_probability": 0.15, "blur_sigma_range": (0.5, 1.5),
    "noise_probability": 0.15, "noise_std_fraction": 0.05,
    "contrast_probability": 0.15, "contrast_range": (0.75, 1.25),
    "gamma_probability": 0.15, "gamma_range": (0.7, 1.5),
    "low_res_probability": 0.15, "low_res_scale_range": (0.5, 1.0),
}


def _simulate_low_resolution(image: torch.Tensor, scale: float) -> torch.Tensor:
    """Downsample then upsample back to the original grid -- simulates a
    coarser acquisition, matching the paper's 'simulated low-resolution'
    augmentation. image: [1,1,D,H,W]."""
    size = image.shape[-3:]
    small = tuple(max(1, round(s * scale)) for s in size)
    down = F.interpolate(image, size=small, mode="trilinear", align_corners=False)
    return F.interpolate(down, size=size, mode="trilinear", align_corners=False)


def _apply_gamma(image: torch.Tensor, gamma: float) -> torch.Tensor:
    """Power-law gamma correction on a [0,1]-rescaled copy, then rescaled
    back to the input's own min/max so z-score-normalized inputs stay in a
    comparable range."""
    lo, hi = image.min(), image.max()
    span = (hi - lo).clamp_min(1e-6)
    unit = ((image - lo) / span).clamp(0, 1)
    corrected = unit.pow(gamma)
    return corrected * span + lo


def augment_pair(image: torch.Tensor, target: torch.Tensor, rng: random.Random, spec: dict = DEFAULT_SPEC):
    """image, target: [1,1,D,H,W] tensors (target binary 0/1). Returns a new
    (image, target) pair with paper-style augmentation applied. Deterministic
    given rng's state -- callers pass a fresh random.Random(seed) per sample
    per epoch, matching this project's existing augmentation convention."""
    device = image.device

    # --- Spatial transforms: same random draw applied to image AND target. ---
    if rng.random() < spec["flip_probability"]:
        axis = rng.choice([2, 3, 4])
        image = torch.flip(image, dims=[axis])
        target = torch.flip(target, dims=[axis])

    if rng.random() < spec["rotate_probability"]:
        deg = spec["rotate_range_deg"]
        angles = [rng.uniform(-deg, deg) * 3.14159265 / 180 for _ in range(3)]
        seed = rng.randint(0, 2**31 - 1)
        rot_img = RandRotated(keys=["image"], range_x=angles[0], range_y=angles[1], range_z=angles[2],
                               prob=1.0, mode="bilinear", padding_mode="zeros")
        rot_img.set_random_state(seed=seed)
        rot_tgt = RandRotated(keys=["target"], range_x=angles[0], range_y=angles[1], range_z=angles[2],
                               prob=1.0, mode="nearest", padding_mode="zeros")
        rot_tgt.set_random_state(seed=seed)
        image = rot_img({"image": image[0]})["image"].unsqueeze(0)
        target = rot_tgt({"target": target[0]})["target"].unsqueeze(0)
        target = (target > 0.5).float()

    if rng.random() < spec["zoom_probability"]:
        lo, hi = spec["zoom_range"]
        factor = rng.uniform(lo, hi)
        seed = rng.randint(0, 2**31 - 1)
        zoom_img = RandZoomd(keys=["image"], min_zoom=factor, max_zoom=factor, prob=1.0, mode="trilinear")
        zoom_img.set_random_state(seed=seed)
        zoom_tgt = RandZoomd(keys=["target"], min_zoom=factor, max_zoom=factor, prob=1.0, mode="nearest")
        zoom_tgt.set_random_state(seed=seed)
        image = zoom_img({"image": image[0]})["image"].unsqueeze(0)
        target = zoom_tgt({"target": target[0]})["target"].unsqueeze(0)
        target = (target > 0.5).float()

    # --- Intensity transforms: image only. ---
    if rng.random() < spec["blur_probability"]:
        lo, hi = spec["blur_sigma_range"]
        sigma = rng.uniform(lo, hi)
        blur = RandGaussianSmoothd(keys=["image"], sigma_x=(sigma, sigma), sigma_y=(sigma, sigma), sigma_z=(sigma, sigma), prob=1.0)
        blur.set_random_state(seed=rng.randint(0, 2**31 - 1))
        image = blur({"image": image[0]})["image"].unsqueeze(0)

    if rng.random() < spec["noise_probability"]:
        std = image.std().item() * spec["noise_std_fraction"]
        noise = RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=max(std, 1e-6))
        noise.set_random_state(seed=rng.randint(0, 2**31 - 1))
        image = noise({"image": image[0]})["image"].unsqueeze(0)

    if rng.random() < spec["contrast_probability"]:
        lo, hi = spec["contrast_range"]
        factor = rng.uniform(lo, hi)
        contrast = RandAdjustContrastd(keys=["image"], prob=1.0, gamma=(factor, factor))
        contrast.set_random_state(seed=rng.randint(0, 2**31 - 1))
        image = contrast({"image": image[0]})["image"].unsqueeze(0)

    if rng.random() < spec["gamma_probability"]:
        lo, hi = spec["gamma_range"]
        image = _apply_gamma(image, rng.uniform(lo, hi))

    if rng.random() < spec["low_res_probability"]:
        lo, hi = spec["low_res_scale_range"]
        image = _simulate_low_resolution(image, rng.uniform(lo, hi))

    return image.to(device), target.to(device)
