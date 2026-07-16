"""Focused tests for resource resolution, strict loading, and feature shapes."""

import json
import importlib.util
import unittest

from models.rs2net_encoder_adapter import REPOSITORY_ROOT, RS2NetEncoderAdapter, RS2NetPaths


CONFIG = json.loads((REPOSITORY_ROOT / "configs/rs2net_encoder.yaml").read_text())


def _dependencies_available() -> bool:
    return importlib.util.find_spec("monai") is not None


class RS2NetEncoderAdapterTests(unittest.TestCase):
    def test_relative_resources_resolve_outside_baseline_repository(self):
        paths = RS2NetPaths.from_config(CONFIG)
        paths.validate()
        self.assertEqual(paths.baseline_root.name, "RS2-Net-Reproduction")
        self.assertEqual(paths.dataset_root.name, "Datasets")
        self.assertEqual(paths.checkpoint.name, "RS2_pretrained_model_clean.pt")

    @unittest.skipUnless(_dependencies_available(), "baseline MONAI dependencies are not installed")
    def test_checkpoint_loads_strictly(self):
        adapter = RS2NetEncoderAdapter(RS2NetPaths.from_config(CONFIG))
        self.assertFalse(adapter.network.training)

    @unittest.skipUnless(_dependencies_available(), "baseline MONAI dependencies are not installed")
    def test_decoder_ready_feature_shapes(self):
        import torch

        adapter = RS2NetEncoderAdapter(RS2NetPaths.from_config(CONFIG))
        with torch.inference_mode():
            features = adapter(torch.zeros(1, 1, 128, 128, 160))
        self.assertEqual(
            {name: tuple(value.shape) for name, value in features.items()},
            {
                "level0": (1, 48, 128, 128, 160),
                "level1": (1, 48, 64, 64, 80),
                "level2": (1, 96, 32, 32, 40),
                "level3": (1, 192, 16, 16, 20),
                "level4": (1, 384, 8, 8, 10),
            },
        )


if __name__ == "__main__":
    unittest.main()
