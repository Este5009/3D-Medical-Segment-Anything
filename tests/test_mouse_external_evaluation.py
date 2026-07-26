import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import evaluate_mouse_external_dataset as mouse

class MouseExternalEvaluationTests(unittest.TestCase):
    def test_filename_pair_normalization(self):
        self.assertEqual(mouse.normalized(Path("A_B.nii.gz")),mouse.normalized(Path("a-b.nii.gz")))
    def test_explicit_mouse_and_date(self):
        mid,date=mouse.identity("POLYIC_20190510_mouse30__E2_P1.nii.gz");self.assertEqual(mid,"mouse-30");self.assertEqual(date,"20190510")
    def test_anonymous_identity_is_not_invented(self):
        mid,_=mouse.identity("POLYIC_20190524_polyic___E3_P1_1.nii.gz");self.assertTrue(mid.startswith("anonymous-"))
    def test_publication_dpi(self):self.assertGreaterEqual(mouse.DPI,180)
    def test_thresholds_are_locked(self):self.assertEqual(mouse.THRESHOLDS,(.80,.90,.95,.97,.98,.99))
if __name__=="__main__":unittest.main()
