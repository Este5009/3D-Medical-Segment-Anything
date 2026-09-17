# RS2-Net baseline translation-invariance vs. this project's model

Same rotation/translation stress test as `test_orientation_invariance.py` / `test_translation_invariance.py`, run against the ORIGINAL RS2-Net model (their own encoder AND their own decoder, no query mechanism) instead of this project's TwoTaskLevel0OneQueryMaskDecoder. Brain segmentation only (CAMRI + mouse) -- RS2-Net has no lesion task.

| domain | mean self-consistency Dice | mean accuracy delta vs expert |
|---|---|---|
| camri | 0.9934 | -0.0049 |
| mouse | 0.9852 | -0.0032 |

## Versus this project's own two-task model (brain query, mouse domain)

- RS2-Net baseline (their encoder + their decoder, rotation-augmented training): see table above.
- This project's model (frozen RS2-Net encoder + this project's own decoder, NO rotation augmentation): mean self-consistency Dice **0.9883** (`outputs/translation_invariance/brain/summary_by_mask.csv`).

If the baseline number above is meaningfully higher than this project's own 0.9883, that is direct evidence the paper's rotation-augmented training measurably helps -- and a concrete case for adding the same kind of augmentation to this project's own decoder training. If the two are close, the gap this project measured is coming from somewhere other than missing rotation augmentation (e.g. the decoder's skip-fusion compounding, or the lesion task specifically, which this baseline cannot speak to at all).
