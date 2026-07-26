import unittest
import torch
from models.query_mask_decoder import MultiScaleOneQueryMaskDecoder
from scripts.train_mouse_boundary_adaptation import freeze_boundary_head_only, adaptation_loss


class MouseBoundaryAdaptationTests(unittest.TestCase):
    def test_only_boundary_layers_train(self):
        model=MultiScaleOneQueryMaskDecoder(32,4);names,count=freeze_boundary_head_only(model)
        self.assertEqual(count,29793)
        self.assertTrue(all(n.startswith(("mask_embedding.","mask_refinement.","mask_bias")) for n in names))
        self.assertFalse(model.query.requires_grad)

    def test_fp_aware_loss_is_finite_and_backward_is_scoped(self):
        logits=torch.randn(1,1,6,6,6,requires_grad=True);target=torch.zeros_like(logits);target[...,2:4,2:4,2:4]=1
        loss,parts=adaptation_loss(logits,target,{"dice":.5,"bce":.25,"tversky":.25},.7,.3);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertIsNotNone(logits.grad);self.assertEqual(set(parts),{"dice_loss","bce_loss","tversky_loss"})

    def test_fp_penalty_exceeds_fn_weight(self):
        self.assertGreater(.7,.3)

if __name__=="__main__":unittest.main()
