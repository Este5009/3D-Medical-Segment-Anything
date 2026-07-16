"""Unit tests for metric validation and nonduplicated slice selection."""

import unittest
import numpy as np

from scripts.visualize_external_holdout_results import informative_slices, recompute_binary_metrics


class ExternalVisualizationTests(unittest.TestCase):
    def test_binary_metric_recomputation(self):
        target=np.array([[1,1,0,0]],bool);prediction=np.array([[1,0,1,0]],bool)
        metrics=recompute_binary_metrics(prediction,target)
        self.assertAlmostEqual(metrics["dice"],0.5,places=4);self.assertAlmostEqual(metrics["iou"],1/3,places=4)
        self.assertEqual(metrics["false_positives"],1);self.assertEqual(metrics["false_negatives"],1)

    def test_informative_slice_selection_is_unique(self):
        slices=[]
        for index in range(12):slices.append({"slice_index":index,"brain_voxels":10 if 2<=index<=9 else 0,"dice":0.5+index/30,"error":20-index})
        selected=informative_slices({"slices":slices})
        self.assertEqual(len(selected),6);self.assertEqual(len(set(selected)),6);self.assertTrue(all(2<=value<=9 for value in selected))


if __name__=="__main__":unittest.main()
