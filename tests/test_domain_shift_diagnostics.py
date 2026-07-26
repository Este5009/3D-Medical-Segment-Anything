"""Focused tests for the read-only domain-shift diagnostics."""
import unittest
import numpy as np
from scripts.domain_shift_diagnostics import binary_metrics, spatial_error_regions, safe_correlation


class DomainShiftDiagnosticsTests(unittest.TestCase):
    def test_binary_metrics_identifies_expansion(self):
        target=np.zeros((9,9,9),bool);target[3:6,3:6,3:6]=1
        pred=np.zeros_like(target);pred[2:7,2:7,2:7]=1
        result=binary_metrics(pred,target,(1,1,1),0.2)
        self.assertGreater(result["false_positives"],0)
        self.assertEqual(result["false_negatives"],0)
        self.assertGreater(result["volume_ratio"],1)

    def test_region_percentages_account_for_all_errors(self):
        target=np.zeros((10,10,10),bool);target[2:8,2:8,2:8]=1
        pred=target.copy();pred[0,0,0]=1;pred[5,5,5]=0
        rows=spatial_error_regions(pred,target)
        self.assertEqual(sum(r["fp_voxels"] for r in rows if r["partition"]=="axis_halves"),3)
        self.assertEqual(sum(r["fn_voxels"] for r in rows if r["partition"]=="axis_halves"),3)

    def test_constant_correlation_is_nan(self):
        self.assertTrue(np.isnan(safe_correlation([1,1,1],[1,2,3])[0]))


if __name__ == "__main__": unittest.main()
