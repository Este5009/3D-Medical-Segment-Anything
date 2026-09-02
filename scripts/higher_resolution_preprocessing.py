"""Higher-resolution companion to corrected_label_preprocessing.py.

Identical crop/transpose/resample pipeline (same DefaultPreprocessor, same
image interpolation method, same corrected nearest-neighbor categorical mask
path) as the valid true-level0 baseline. The ONLY difference is that the
target model-grid spacing is overridden in-memory (a dict entry on the
already-loaded plans configuration, never a file edit) before resampling, so
callers can request a finer grid on the one axis the prior diagnostic
(outputs/higher_resolution_encoder_diagnostic/) identified as the dominant
native-detail-loss axis, while leaving the other two axes' spacing unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from train_query_decoder_overfit import _pad_and_center_crop, load_json

# The "modest" candidate validated by the encoder diagnostic: only model
# axis0 (-> native Y, the dominant loss axis) is refined; axis1 (-> native Z)
# and axis2 (-> native X) keep the valid baseline's spacing.
HIGHER_RESOLUTION_SPACING = (0.16666666666666666, 0.20000000298023224, 0.1599999964237213)
HIGHER_RESOLUTION_TILE = (192, 128, 160)


def _load_manager_and_configuration(paths, spacing):
    from RS2.utilities.plans_handling.plans_handler import PlansManager

    root = paths.baseline_project / "RS2/jsons"
    manager = PlansManager(load_json(root / "plans.json"))
    configuration = manager.get_configuration("3d_fullres")
    configuration.configuration["spacing"] = list(spacing)  # in-memory only; plans.json is never touched
    dataset = load_json(root / "dataset.json")
    return manager, configuration, dataset


def corrected_target_at_spacing(mask_path, properties, plans_manager, configuration, tile_size):
    """Identical to corrected_label_preprocessing.corrected_target, just
    parametrized by an already-spacing-overridden configuration."""
    from RS2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

    reader = plans_manager.image_reader_writer_class()
    segmentation, _ = reader.read_seg(str(mask_path))
    segmentation = segmentation.transpose([0, *[axis + 1 for axis in plans_manager.transpose_forward]])
    segmentation = segmentation[(slice(None), *bounding_box_to_slice(properties["bbox_used_for_cropping"]))]
    source_spacing = [properties["spacing"][axis] for axis in plans_manager.transpose_forward]
    target_spacing = list(configuration.spacing)
    target_shape = properties["shape_after_resampling"]
    categorical = resample_data_or_seg_to_shape(
        (segmentation > 0).astype(np.uint8), target_shape, source_spacing, target_spacing,
        is_seg=True, order=0, order_z=0, force_separate_z=None)
    unique = set(np.unique(categorical).tolist())
    if not unique <= {0, 1}:
        raise ValueError(f"Noncategorical corrected mask values: {unique}")
    tensor = torch.from_numpy(categorical.astype(np.float32)).unsqueeze(0)
    return _pad_and_center_crop(tensor, tile_size)


def preprocess_image_and_corrected_target_at_spacing(image_path, mask_path, paths, spacing, tile_size):
    """Same as corrected_label_preprocessing.preprocess_image_and_corrected_target
    (byte-for-byte identical image interpolation method -- the shared
    DefaultPreprocessor's own resampling_fn, unchanged), with the target grid
    spacing overridden to `spacing` instead of the baseline's fixed plans.json value."""
    from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    manager, configuration, dataset = _load_manager_and_configuration(paths, spacing)
    data, _, properties = DefaultPreprocessor(False).run_case(
        [str(image_path)], str(mask_path), manager, configuration, dataset)
    properties = dict(properties)
    properties["shape_after_resampling"] = tuple(int(x) for x in data.shape[1:])
    image = _pad_and_center_crop(
        torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0), tile_size)
    target = corrected_target_at_spacing(mask_path, properties, manager, configuration, tile_size)
    if image.shape != target.shape:
        raise ValueError(f"Image/target mismatch: {image.shape} versus {target.shape}")
    return image, target, list(data.shape), properties


def preprocess_at_spacing_for_eval(image_path, mask_path, paths, tile_size, spacing=HIGHER_RESOLUTION_SPACING):
    """Drop-in replacement for evaluate_external_holdout.preprocess with an
    overridden target spacing. Returns the identical 5-tuple
    (image, properties, manager, configuration, dataset) so the unchanged
    sliding_window_logits / export_native / export_probability functions
    work without modification."""
    from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    manager, configuration, dataset = _load_manager_and_configuration(paths, spacing)
    data, segmentation, properties = DefaultPreprocessor(verbose=False).run_case(
        [str(image_path)], str(mask_path), manager, configuration, dataset)
    image = torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0)
    return image, properties, manager, configuration, dataset
