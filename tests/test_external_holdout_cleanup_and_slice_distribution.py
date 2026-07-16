import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import clean_and_analyze_external_holdout as analysis


class CleanupAndSliceDistributionTests(unittest.TestCase):
    def row(self,dice,expert=True,prediction=True):
        return {"dice":dice,"expert_nonempty":expert,"prediction_nonempty":prediction}

    def test_range_boundaries(self):
        expected={0.0:"Dice = 0.00",0.5:"0.50 <= Dice < 0.70",0.7:"0.70 <= Dice < 0.80",0.8:"0.80 <= Dice < 0.90",0.9:"0.90 <= Dice < 0.95",0.95:"0.95 <= Dice < 0.97",0.97:"0.97 <= Dice < 0.98",0.98:"0.98 <= Dice < 0.99",0.99:"0.99 <= Dice <= 1.00",1.0:"0.99 <= Dice <= 1.00"}
        for value,label in expected.items(): self.assertEqual(analysis.category(self.row(value)),label)

    def test_true_empty_has_own_category(self):
        self.assertEqual(analysis.category(self.row(1.0,False,False)),"Empty expert and empty prediction")
        self.assertEqual(analysis.category(self.row(0.0,False,True)),"Dice = 0.00")

    def test_cleanup_manifest_never_targets_protected_types(self):
        for path in analysis.cleanup_manifest():
            self.assertNotIn(analysis.FIXED,path.parents)
            self.assertNotIn(path.suffix.lower(),(".csv",".nii",".gz"))

    def test_publication_dpi(self): self.assertGreaterEqual(analysis.DPI,180)


if __name__=="__main__":unittest.main()
