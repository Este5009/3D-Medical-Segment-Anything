"""Segmentation-aware companion to the immutable RS2 image preprocessor.

Only the expert-label interpolation differs from the established pipeline.
Images still come from ``DefaultPreprocessor.run_case`` byte-for-byte.
"""
from __future__ import annotations

import numpy as np
import torch

from train_query_decoder_overfit import _pad_and_center_crop


def corrected_target(mask_path, properties, plans_manager, configuration, tile_size):
    """Return a categorical `{0,1}` target on the existing model tile grid."""
    from RS2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

    reader = plans_manager.image_reader_writer_class()
    segmentation, _ = reader.read_seg(str(mask_path))
    segmentation = segmentation.transpose(
        [0, *[axis + 1 for axis in plans_manager.transpose_forward]])
    segmentation = segmentation[
        (slice(None), *bounding_box_to_slice(properties["bbox_used_for_cropping"]))]
    source_spacing = [properties["spacing"][axis]
                      for axis in plans_manager.transpose_forward]
    target_spacing = list(configuration.spacing)
    target_shape = properties["shape_after_resampling"]
    categorical = resample_data_or_seg_to_shape(
        (segmentation > 0).astype(np.uint8), target_shape,
        source_spacing, target_spacing, is_seg=True, order=0, order_z=0,
        force_separate_z=None)
    unique = set(np.unique(categorical).tolist())
    if not unique <= {0, 1}:
        raise ValueError(f"Noncategorical corrected mask values: {unique}")
    tensor = torch.from_numpy(categorical.astype(np.float32)).unsqueeze(0)
    return _pad_and_center_crop(tensor, tile_size)


def preprocess_image_and_corrected_target(image_path, mask_path, paths, tile_size):
    """Keep MRI preprocessing exact and replace only label interpolation."""
    from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from RS2.utilities.plans_handling.plans_handler import PlansManager
    from train_query_decoder_overfit import load_json

    root = paths.baseline_project / "RS2/jsons"
    manager = PlansManager(load_json(root / "plans.json"))
    configuration = manager.get_configuration("3d_fullres")
    data, _, properties = DefaultPreprocessor(False).run_case(
        [str(image_path)], str(mask_path), manager, configuration,
        load_json(root / "dataset.json"))
    # The official preprocessor records this geometry implicitly through its
    # returned array. Make it explicit for the categorical label resampler.
    properties = dict(properties)
    properties["shape_after_resampling"] = tuple(int(x) for x in data.shape[1:])
    image = _pad_and_center_crop(
        torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0), tile_size)
    target = corrected_target(mask_path, properties, manager, configuration, tile_size)
    if image.shape != target.shape:
        raise ValueError(f"Image/target mismatch: {image.shape} versus {target.shape}")
    return image, target, list(data.shape), properties
