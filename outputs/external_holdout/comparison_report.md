# External CAMRI Rat holdout evaluation

## Evaluation integrity

- **No training, fine-tuning, optimizer step, backward pass, or parameter update
  was performed.**
- The only loaded decoder checkpoint was the completed pilot checkpoint from
  epoch 14 (`validation_dice = 0.982801`).
- Strict state-dict loading verified 170,401 decoder parameters.
- The learned query tensor has shape `[1, 1, 32]`, proving the loaded model
  contains exactly one learned query.
- Architecture remained frozen RS2-Net encoder + multi-scale query attention +
  top-down FPN + one mask head.
- All 40 pilot train/validation/test subjects were excluded before discovery of
  the external holdout.

## Cohort audit

- CAMRI Rat image/mask pairs discovered: 131.
- Subjects used anywhere in the pilot and excluded: 40.
- Completely unseen holdout candidates: 91.
- Holdout subjects evaluated: **91**.
- Holdout subjects excluded: **0**.

Seven original masks had affine mismatches (051, 052, 069, 092, 105, 107,
116). Each was safely resampled with nearest-neighbor interpolation into the
image's physical SimpleITK reference grid. Restored size, spacing, origin, and
direction exactly matched the image. All restored masks remained nonempty and
were evaluated. Their mean Dice was 0.97333 (range 0.96366-0.97693).

Volumes wider than the fixed model tile were evaluated with overlapping
sliding windows in preprocessed space and logit averaging. This is an inference
procedure only; it does not alter model parameters or architecture.

## External results

| Statistic | Value |
|---|---:|
| Mean Dice | **0.977904** |
| Median Dice | 0.978958 |
| Standard deviation | 0.005969 |
| Minimum | 0.955824 |
| Maximum | 0.985291 |
| 5th percentile | 0.965266 |
| 95th percentile | 0.984439 |
| Mean IoU | 0.956830 |
| Mean precision | 0.996355 |
| Mean recall | 0.960201 |
| Mean 95HD | 0.19649 mm |
| Mean encoder + decoder inference | 15.968 seconds |

The high precision and lower recall indicate that residual error is dominated
by under-segmentation rather than false-positive expansion.

## Comparison with the pilot

| Cohort | Mean Dice | Pilot minus external |
|---|---:|---:|
| Pilot train | 0.988120 | 0.010216 |
| Pilot validation | 0.982801 | 0.004897 |
| Pilot test | 0.987828 | 0.009923 |
| External holdout | **0.977904** | - |

The external result is lower than every pilot split, as expected for a much
larger and more heterogeneous cohort, but remains strong and tightly
distributed. This is clear evidence that the decoder generalizes beyond the
40-subject pilot rather than memorizing only that subset.

## Failure modes

The worst subjects were 108 (0.95582), 075 (0.96252), 111 (0.96334), 051
(0.96366), and 110 (0.96426). Visual and per-slice review shows:

- missed peripheral brain fragments on the first and last brain-containing
  slices;
- thin under-segmented bands at ventral, lateral, and caudal/rostral boundaries;
- occasional missed disconnected foreground islands near volume edges;
- accurate central brain retrieval even in the worst cases.

Across subjects, 68.1% of the maximum-error slices occurred in the outer 20%
of the volume index. Dice correlated strongly with recall (`r = 0.981`) but
essentially not with precision (`r = 0.020`), quantitatively confirming that
missed boundary anatomy is the dominant failure.

## Hardest acquisition types

Two clear CAMRI acquisition-shape groups were present:

| Native geometry group | N | Mean Dice | Minimum |
|---|---:|---:|---:|
| 12 slices, approximately 0.1 x 0.1 x 1.0 mm | 37 | **0.982752** | 0.966619 |
| 64 slices, approximately 0.2 mm isotropic | 54 | **0.974583** | 0.955824 |

The 64-slice/isotropic group was harder by 0.00817 Dice. Dice also correlated
negatively with brain volume (`r = -0.739`) and slice count (`r = -0.672`).
These factors are correlated in this dataset, so this evaluation cannot claim
which one is causal; it can conclude that the larger isotropic acquisition
cohort is the main source of failures.

## Does the frozen encoder limit performance?

There is no sufficient evidence to unfreeze it yet. A frozen encoder supports
0.97790 Dice across all 91 completely unseen subjects, 0.99635 precision, and a
narrow Dice standard deviation of 0.00597. The acquisition-group gap and
recall-dominated edge errors could arise from insufficient representation in
the 40-subject training subset, not necessarily an encoder ceiling. A full-data
controlled run is required before attributing them to frozen Swin features.

## Recommendation

Choose **A: train on the full geometry-valid dataset using the current
architecture**.

- Do not introduce multiple queries: this remains a single-object brain task,
  and one query generalizes to 0.97790 Dice.
- Do not partially or jointly fine-tune the encoder yet: the frozen model is
  already strong, and no controlled evidence isolates an encoder bottleneck.
- Full-data training directly addresses the observed weakness because the
  hardest acquisition cohort is underrepresented by the 28-subject pilot
  training set.

Only if a full-data frozen-encoder experiment retains the same systematic
64-slice recall deficit should option C (fine-tune the last Swin stage) be tested
as the next single controlled change.
