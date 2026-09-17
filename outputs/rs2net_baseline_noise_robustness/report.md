# RS2-Net baseline noise/blur/intensity robustness vs. this project's model

Same corruption model and severities as `test_noise_robustness.py`, run against the ORIGINAL RS2-Net model (their own encoder AND their own decoder) on CAMRI + mouse brain data -- RS2-Net has no lesion task. Compared directly against this project's own brain-query result (`outputs/noise_robustness/brain/`).

## camri

| corruption | severity | RS2-Net baseline mean Dice | this project's model mean Dice | difference |
|---|---|---|---|---|
| blur | 0.5 | 0.9926 | 0.9613 | +0.0312 |
| blur | 1 | 0.9910 | 0.9182 | +0.0728 |
| blur | 2 | 0.9807 | 0.6600 | +0.3208 |
| blur | 4 | 0.9257 | 0.4197 | +0.5060 |
| intensity | 0.1 | 0.9927 | 0.9691 | +0.0236 |
| intensity | 0.25 | 0.9924 | 0.9696 | +0.0229 |
| intensity | 0.5 | 0.9920 | 0.9523 | +0.0397 |
| intensity | 0.75 | 0.9892 | 0.9287 | +0.0605 |
| noise | 0.05 | 0.9926 | 0.9650 | +0.0276 |
| noise | 0.1 | 0.9924 | 0.9592 | +0.0332 |
| noise | 0.2 | 0.9906 | 0.9495 | +0.0411 |
| noise | 0.4 | 0.9829 | 0.8013 | +0.1816 |

## mouse

| corruption | severity | RS2-Net baseline mean Dice | this project's model mean Dice | difference |
|---|---|---|---|---|
| blur | 0.5 | 0.9095 | 0.9613 | -0.0518 |
| blur | 1 | 0.8822 | 0.9182 | -0.0360 |
| blur | 2 | 0.7316 | 0.6600 | +0.0716 |
| blur | 4 | 0.4682 | 0.4197 | +0.0485 |
| intensity | 0.1 | 0.9145 | 0.9691 | -0.0546 |
| intensity | 0.25 | 0.9129 | 0.9696 | -0.0567 |
| intensity | 0.5 | 0.9139 | 0.9523 | -0.0384 |
| intensity | 0.75 | 0.8411 | 0.9287 | -0.0875 |
| noise | 0.05 | 0.9123 | 0.9650 | -0.0526 |
| noise | 0.1 | 0.9063 | 0.9592 | -0.0529 |
| noise | 0.2 | 0.8618 | 0.9495 | -0.0877 |
| noise | 0.4 | 0.6385 | 0.8013 | -0.1627 |

_Positive difference = RS2-Net's own decoder scored higher than this project's decoder at that corruption/severity, on brain segmentation._
