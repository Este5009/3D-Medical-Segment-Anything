import unittest

import torch

from models.query_mask_decoder import dice_bce_boundary_loss
from scripts.analyze_mixed_domain_results import leakage_components,per_slice_rows
from scripts.train_mixed_domain_decoder import balanced_epoch_order

class MixedDomainDecoderTests(unittest.TestCase):
    def test_balanced_order_alternates_domains(self):
        order=balanced_epoch_order(list(range(5)),list(range(2)),seed=1)
        self.assertEqual(len(order),10)
        self.assertTrue(all(order[i][0]!=order[i+1][0] for i in range(len(order)-1)))
        self.assertEqual(sum(d=="camri" for d,_ in order),sum(d=="mouse" for d,_ in order))

    def test_boundary_loss_is_symmetric_and_differentiable(self):
        target=torch.zeros(1,1,9,9,9)
        target[:,:,2:7,2:7,2:7]=1
        logits=torch.zeros_like(target,requires_grad=True)
        loss,parts=dice_bce_boundary_loss(logits,target,boundary_weight=.25,boundary_width=1)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(parts["boundary_fraction"]),0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        # Both sides of the contour are supervised: gradient descent raises
        # foreground logits and lowers adjacent background logits.
        self.assertLess(float(logits.grad[0,0,2,4,4]),0)
        self.assertGreater(float(logits.grad[0,0,1,4,4]),0)

    def test_leakage_components_exclude_the_anatomical_component(self):
        target=torch.zeros(8,8,8,dtype=torch.bool).numpy()
        target[2:6,2:6,2:6]=True
        prediction=target.copy()
        prediction[0,0,0]=True
        count,voxels=leakage_components(prediction,target)
        self.assertEqual((count,voxels),(1,1))

    def test_per_slice_rows_label_terminal_regions(self):
        target=torch.zeros(6,6,10,dtype=torch.bool).numpy()
        target[1:5,1:5,2:8]=True
        rows=per_slice_rows(target.copy(),target,"test","case")
        positions={row["brain_position"] for row in rows}
        self.assertEqual(
            positions,
            {"outside","first 20%","middle 60%","last 20%"},
        )
        self.assertTrue(all(row["dice"]==1 for row in rows))

if __name__=="__main__":unittest.main()
