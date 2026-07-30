# Query-conditioned 3D anatomical grouping experiment

## Controlled result

The frozen epoch-17 mixed-domain model was not changed. A separate 8,537-parameter grouping module used frozen level-2 features, the initial mask logit/probability/uncertainty, normalized coordinates, and optional FiLM from the existing single query. Two depthwise-separable 3D blocks predicted a bounded residual logit correction. Encoder and baseline-decoder gradients remained disabled.

Validation selected `learned_coordinates_no_query` at epoch 1: balanced Dice 0.976714 versus baseline 0.976576 (+0.000138). The coordinate+query model reached 0.976706; query conditioning therefore contributed no measurable benefit. The no-coordinate result was 0.976683, so coordinates also added only 0.000031.

## Architecture

```text
frozen RS2 level2 [B,96,32,32,40] ── 1×1 projection ─┐
frozen decoder logits ── logit/probability/uncertainty ├─ concat + coordinates
existing single query ── optional FiLM ────────────────┘
               → two depthwise-separable 3D residual blocks
               → global context gate → bounded correction
               → frozen initial logits + correction → final mask
```

## Untouched tests

| Domain | Variant | Dice | Precision | Recall | HD95 mm | FP | FN | Volume ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CAMRI | baseline | 0.982518 | 0.994785 | 0.970597 | 0.1333 | 824.7 | 5615.5 | 0.975703 |
| CAMRI | deterministic_filter | 0.982626 | 0.995008 | 0.970597 | 0.1333 | 774.7 | 5615.5 | 0.975487 |
| CAMRI | learned | 0.981967 | 0.995232 | 0.969100 | 0.1333 | 751.3 | 5892.2 | 0.973759 |
| Mouse | baseline | 0.964762 | 0.981696 | 0.948492 | 0.3435 | 2181.9 | 6023.0 | 0.966266 |
| Mouse | deterministic_filter | 0.965685 | 0.983609 | 0.948492 | 0.2163 | 1961.2 | 6023.0 | 0.964366 |
| Mouse | learned | 0.963906 | 0.983584 | 0.945089 | 0.3461 | 1949.3 | 6422.9 | 0.960946 |


Learned Dice changed by -0.000551 on CAMRI and -0.000856 on Mouse. At tolerance 0.0001, CAMRI improved/unchanged/worsened = 0/0/6; Mouse = 1/2/77. The absolute worst learned case was `POLYIC_20190531_polyic_m__E3_P1_3`.

## Grouping and boundary evidence

Across both test sets, baseline detached FP count/voxels/affected subjects were 57/17961/32. The deterministic filter reduced these to 0/0/0; learned grouping produced 56/17471/33. Boundary-error voxels were baseline 675,597, filter 675,545, learned 690,780.

Connected leakage voxels were baseline/filter/learned = 1,385/1,385/1,261; terminal-region errors were 257,735/248,676/261,146.

The deterministic filter improved mean Dice in both domains and reduced Mouse HD95 from 0.3435 to 0.2163 mm without changing recall. The learned residual instead reduced recall and mask volume, degraded Dice in both domains, did not reduce connected components, and slightly worsened Mouse HD95. Its apparent validation advantage did not reproduce on test. The fixed single query is constant for every case, so FiLM supplies no sample-varying object identity; the ablation correctly found no measurable query effect. Global context likewise failed to improve boundaries.

Safety checks show no slice-wise filtering: connectivity is fully 3D and diagonal connections are retained. The deterministic filter's `expert_voxels_outside_primary_component` column explicitly audits any expert-overlapping component removed. The learned model did not introduce a gross localization failure, but it systematically shrank correct anatomy (recall and volume ratio fell), including terminal/thin boundary regions visible in the six fixed figures. Test labels were used only after predictions were saved.

## Conclusion

**B. Deterministic 3D filtering captures most of the available benefit, so the learned grouping decoder is not currently justified.**

The learned module adds complexity without robust generalization; the simpler inference-only 26-connected-component rule gives the stronger Dice, topology, FP, and HD95 result. No further grouping-module revision is warranted on these benchmarks.
