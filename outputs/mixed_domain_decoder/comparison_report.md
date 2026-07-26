# Mixed-domain one-query decoder

The unchanged 170,401-parameter one-query decoder was initialized from epoch 14 and trained with balanced 50/50 CAMRI-mouse updates. The RS2-Net encoder remained frozen. CAMRI validation never crossed the -0.01 safety floor.

## Independent test results

### CAMRI test (n=6)

| Model | Dice | Precision | Recall | HD95 mm | FP voxels | FN voxels | Volume ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original CAMRI | 0.9818 | 0.9966 | 0.9675 | 0.1333 | 538.8 | 6049.5 | 0.9708 |
| Mouse-adapted | 0.9280 | 1.0000 | 0.8658 | 0.4477 | 2.8 | 24495.2 | 0.8658 |
| Mixed-domain | 0.9823 | 0.9950 | 0.9699 | 0.1333 | 767.2 | 5717.5 | 0.9747 |

### Mouse test (n=80)

| Model | Dice | Precision | Recall | HD95 mm | FP voxels | FN voxels | Volume ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original CAMRI | 0.8819 | 0.8076 | 0.9727 | 4.4746 | 27685.4 | 3182.5 | 1.2082 |
| Mouse-adapted | 0.9347 | 0.9633 | 0.9081 | 1.3963 | 4140.5 | 10838.2 | 0.9433 |
| Mixed-domain | 0.9618 | 0.9804 | 0.9439 | 0.4440 | 2310.2 | 6553.6 | 0.9629 |

## Decision

The mixed model preserved CAMRI performance: its mean Dice changed by +0.0005, and the validation safety stop was not triggered. On Mouse, it improved Dice by +0.0799 over the original model and by +0.0271 over the domain-adapted comparator. Its improved Mouse precision, HD95, FP burden, and volume ratio show that the gain is primarily removal of the original model's systematic over-segmentation, without the severe CAMRI under-segmentation introduced by mouse-only adaptation.

The universal mixed-domain decoder therefore meets the experiment goal on these untouched test sets. Subject-level values are in `camri_test_comparison.csv` and `mouse_test_comparison.csv`; representative panels are separated into `visualizations/best_improvements/` and `visualizations/worst_failures/`.
