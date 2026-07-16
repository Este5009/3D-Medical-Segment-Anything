# RS2-Net encoder audit

## Baseline identity

- Baseline project: `../RS2-Net-Reproduction/Rodent-Skull-Stripping`
- Main model class: `RS2.network.RSSNet.RSSNet`
- Inference entry point: `RS2.inference.predict:predict_entry_point` (`RS2_predict`)
- Selected checkpoint: `../RS2-Net-Reproduction/Rodent-Skull-Stripping/models/RS2_pretrained_model_clean.pt`
- Original compiled checkpoint: `../RS2-Net-Reproduction/Rodent-Skull-Stripping/models/RS2_pretrained_model.pt`

The clean checkpoint has the same 122 state-dict entries as the original, with
the `torch.compile` `_orig_mod.` key prefix removed. The adapter also strips
that prefix defensively and always performs a strict state-dict load.

## Architecture boundary

`RSSNet.forward` first calls `swinViT`, then refines selected scales through
`encoder1`, `encoder2`, `encoder3`, and `feature`. The original U-Net-style
decoder begins at `decoder4(feature, hidden_states_out[2])`. The adapter returns
the five tensors available immediately before that call and never executes
`decoder4`, `decoder3`, `decoder2`, `decoder1`, or `out`.

For a `[1, 1, 128, 128, 160]` tile, the decoder-ready pyramid is:

| Name | Source | Shape `[B, C, D, H, W]` |
|---|---|---|
| `level0` | `encoder1(input)` | `[1, 48, 128, 128, 160]` |
| `level1` | `encoder2(Swin stage 0)` | `[1, 48, 64, 64, 80]` |
| `level2` | `encoder3(Swin stage 1)` | `[1, 96, 32, 32, 40]` |
| `level3` | Swin stage 2 | `[1, 192, 16, 16, 20]` |
| `level4` | `feature(Swin stage 3)` | `[1, 384, 8, 8, 10]` |

The four raw Swin outputs have channels 48, 96, 192, and 384 at spatial
resolutions 1/2, 1/4, 1/8, and 1/16 of the inference tile.

## Preprocessing and inference input

The official `DefaultPreprocessor` reads NIfTI with SimpleITK, transposes using
the plan, crops to nonzero content, resamples to spacing
`[0.25, 0.20000000298, 0.15999999642]`, and applies `ZScoreNormalization`
without a foreground mask. Inference uses sliding-window tiles of
`[128, 128, 160]`. The smoke script runs this exact preprocessing and selects a
center tile (padding if necessary) for the encoder-only check.

## Smoke-audit result

The audit completed successfully in the existing `rs2` Conda environment on
CPU using the configured CAMRI rat T2w volume:

- Original NIfTI shape: `[256, 256, 12]`
- Shape after official preprocessing: `[1, 1, 102, 60, 160]`
- Center-padded encoder tile: `[1, 1, 128, 128, 160]`
- Checkpoint loaded strictly: yes
- All five observed feature shapes matched the table above
- Peak accelerator memory: unavailable for CPU

No feature tensors were saved.

## Import and compatibility status

The baseline imports cleanly from its sibling path in the existing `rs2` Conda
environment; it does not need to be copied, installed, or modified. The active
base environment lacks the baseline imaging dependencies, so commands should
use `conda run -n rs2`. MONAI emits a non-fatal deprecation warning about
`pkg_resources`. An OpenMP duplicate-runtime error was observed only while
probing mixed packages in the base environment; it did not occur during the
real audit in `rs2`.

## Next-step decision

The architectural boundary is clear and runtime-validated. The adapter exposes
the exact multi-scale features consumed by the released decoder, so the next
controlled step can safely build a query decoder on this fixed interface while
leaving the encoder and baseline reproduction unchanged.
