"""Focused tests for the fixed external-holdout visualization pipeline."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rebuild_external_holdout_visualizations as fixed


class ExternalHoldoutVisualizationQualityTests(unittest.TestCase):
    def test_output_is_separate_from_previous_visualizations(self):
        self.assertEqual(fixed.OUT.name, "visualizations_fixed")
        self.assertNotEqual(fixed.OUT, fixed.SOURCE / "visualizations")

    def test_required_color_convention_is_distinct(self):
        colors = [fixed.COLORS[k] for k in ("fn", "fp", "overlap", "expert", "prediction")]
        self.assertEqual(len(colors), len(set(colors)))
        self.assertEqual(fixed.COLORS["fn"], "#2166e6")
        self.assertEqual(fixed.COLORS["fp"], "#e52d2d")

    def test_png_validation_rejects_blank_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            Image.fromarray(np.full((700, 1000, 3), 255, dtype=np.uint8)).save(path)
            with self.assertRaises(RuntimeError):
                fixed.validate_images([path])

    def test_dpi_meets_publication_requirement(self):
        self.assertGreaterEqual(fixed.DPI, 150)


if __name__ == "__main__":
    unittest.main()
