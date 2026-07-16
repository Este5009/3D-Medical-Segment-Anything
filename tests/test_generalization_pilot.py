"""Focused safeguards for subject splitting and binary evaluation metrics."""

import unittest

import torch

from scripts.train_generalization_pilot import binary_metrics, make_subject_split


class GeneralizationPilotTests(unittest.TestCase):
    def test_subject_split_has_expected_sizes_and_no_overlap(self):
        pairs = [(f"{index:03d}", f"image-{index}", f"mask-{index}") for index in range(50)]
        config = {
            "seed": 2026, "subset_subject_count": 40,
            "train_count": 28, "validation_count": 6, "test_count": 6,
        }
        split = make_subject_split(pairs, config)
        self.assertEqual({name: len(values) for name, values in split.items()}, {
            "train": 28, "validation": 6, "test": 6,
        })
        ids = [{item[0] for item in split[name]} for name in ("train", "validation", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_binary_metrics(self):
        # Thresholded prediction [1, 1, 0, 0], truth [1, 0, 1, 0].
        logits = torch.tensor([[[[[10.0, 10.0, -10.0, -10.0]]]]])
        target = torch.tensor([[[[[1.0, 0.0, 1.0, 0.0]]]]])
        metrics = binary_metrics(logits, target)
        self.assertAlmostEqual(metrics["dice"], 0.5, places=4)
        self.assertAlmostEqual(metrics["precision"], 0.5, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=4)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)


if __name__ == "__main__":
    unittest.main()
