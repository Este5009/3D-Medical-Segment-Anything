"""Fast unit tests for the one-query model; no RS2 checkpoint is required."""

import unittest

import torch
import torch.nn as nn

from models.query_mask_decoder import (
    FrozenEncoderQueryModel,
    MultiScaleOneQueryMaskDecoder,
    OneQueryMaskDecoder,
    dice_bce_loss,
)


class TinyFakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, volume):
        batch = volume.shape[0]
        # Small spatial grids preserve the real RS2 channel contracts.
        return {
            "level1": self.scale * torch.randn(batch, 48, 8, 8, 10, device=volume.device),
            "level2": self.scale * torch.randn(batch, 96, 4, 4, 5, device=volume.device),
            "level3": self.scale * torch.randn(batch, 192, 2, 2, 3, device=volume.device),
            "level4": self.scale * torch.randn(batch, 384, 1, 1, 2, device=volume.device),
        }


class QueryMaskDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = FrozenEncoderQueryModel(TinyFakeEncoder(), OneQueryMaskDecoder(16, 4))
        self.volume = torch.randn(1, 1, 16, 16, 20)
        self.target = (torch.rand(1, 1, 16, 16, 20) > 0.7).float()

    def test_encoder_parameters_are_frozen(self):
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.decoder.parameters()))

    def test_output_shape(self):
        self.assertEqual(tuple(self.model(self.volume, self.target.shape[-3:]).shape), tuple(self.target.shape))

    def test_backward_reaches_only_decoder(self):
        loss, _ = dice_bce_loss(self.model(self.volume, self.target.shape[-3:]), self.target)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.model.encoder.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.decoder.parameters()))

    def test_loss_is_finite(self):
        loss, parts = dice_bce_loss(self.model(self.volume, self.target.shape[-3:]), self.target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"dice_loss", "bce_loss"})

    def test_one_optimizer_step_changes_decoder(self):
        optimizer = torch.optim.AdamW(self.model.decoder.parameters(), lr=1e-3)
        before = self.model.decoder.query.detach().clone()
        loss, _ = dice_bce_loss(self.model(self.volume, self.target.shape[-3:]), self.target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.decoder.query.detach()))

    def test_multiscale_decoder_uses_one_query_and_all_scales(self):
        decoder = MultiScaleOneQueryMaskDecoder(16, 4)
        model = FrozenEncoderQueryModel(TinyFakeEncoder(), decoder)
        logits = model(self.volume, self.target.shape[-3:])
        self.assertEqual(tuple(decoder.query.shape), (1, 1, 16))
        self.assertEqual(tuple(logits.shape), tuple(self.target.shape))
        self.assertEqual(set(decoder.projections), {"level1", "level2", "level3", "level4"})


if __name__ == "__main__":
    unittest.main()
