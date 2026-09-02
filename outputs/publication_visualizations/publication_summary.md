# Publication visualization summary

These figures reuse existing test MRI, expert masks, and deterministic
largest-26-component predictions. No training or inference was run.
The model, checkpoint, preprocessing, and threshold were not changed.

- Checkpoint: `outputs/mixed_domain_anatomical_training/checkpoints/best_mixed_domain.pt` (epoch 17)
- Subjects: 6 (best, median, and difficult in each domain)
- Visualization/export files generated: 153

## Subjects

| Dataset | Rank | Subject | Existing filtered Dice | Output |
|---|---|---|---:|---|
| CAMRI | difficult | `064` | 0.9782 | `camri/difficult_064` |
| CAMRI | median | `043` | 0.9838 | `camri/median_043` |
| CAMRI | best | `099` | 0.9864 | `camri/best_099` |
| Mouse | difficult | `POLYIC_20190524_polyic___E4_P1_9` | 0.9557 | `mouse/difficult_POLYIC_20190524_polyic___E4_P1_9` |
| Mouse | median | `POLYIC_20190619_mouse38__E3_P1` | 0.9665 | `mouse/median_POLYIC_20190619_mouse38__E3_P1` |
| Mouse | best | `POLYIC_20190517_mouse39__E12_P1` | 0.9714 | `mouse/best_POLYIC_20190517_mouse39__E12_P1` |

## Primary figure

`publication_summary_figure.png` contains all six representative cases.
Each subject directory contains full-volume contact sheets, six surface
views, seven intensity-volume views, orthogonal contours, an MP4, an
expert comparison, and skull-stripped NIfTI/MHA exports.
