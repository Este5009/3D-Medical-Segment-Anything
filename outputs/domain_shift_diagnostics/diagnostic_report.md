# Comprehensive one-query decoder failure analysis

## Evaluation contract

No training, fine-tuning, checkpoint modification, architecture change, or official-threshold change was performed. All cohort-wide analyses use the existing native predictions. Probability, attention, and feature analyses reran inference only on a prespecified representative subset under `torch.inference_mode`.

## Quantitative diagnosis

Mouse over-segmentation is a **multi-factor domain-shift failure dominated by boundary calibration**. Predicted/expert volume ratio changes from 0.964 on CAMRI to 1.207 on mouse. Mouse mean precision is 0.8087 while recall is 0.9734; mean FP voxels (27,349) exceed FN voxels (3,087). Boundary metrics likewise shift: surface Dice is 0.5251 versus 0.7136, and ASSD is 0.730 versus 0.077 mm.

### Geometry and intensity

CAMRI and mouse geometry differ materially: mean slice spacing is 0.525 vs 0.398 mm, anisotropy 4.67 vs 4.01, and brain occupancy 18.56% vs 9.58%. Robust intensity CNR is 2.320 vs 3.140; boundary sharpness is 0.042 vs 0.067. These measured shifts change the high-resolution evidence available to the frozen decoder.

### Calibration and query behavior

The analytical threshold sweep does not alter the official result. Across the 20 best/worst mouse representatives, Dice rises monotonically from 0.6330 at 0.30 to 0.6911 at 0.70, versus 0.6625 at 0.50. Yet at 0.70 recall remains 0.9795 and volume ratio remains 1.853. The model therefore assigns broadly high probability outside the expert brain; the 0.50 threshold contributes but is not the fundamental cause.

Query attention is **not spatially broader** in mouse scans. Mean level-1 attention spread is 33.460 voxels for mouse versus 33.759 for CAMRI; entropy is 0.9943 versus 0.9965. The attended volume above 10% of peak is smaller, not larger (24,663 versus 290,868 voxels). Attention attribution is highest in true-positive tissue for both domains and does not show preferential FP attention. Thus query expansion is not supported as the proximate mechanism.

### Feature distribution and error morphology

Pooled fused-feature domain centroids remain highly aligned: cosine similarity is 0.9996 at level4, 0.9987 at level3, 0.9984 at level2, and 0.9986 at level1. Modest separation is visible in PCA/t-SNE and feature norms differ at fine scales, but the representations do not form orthogonal latent spaces. This is evidence against gross encoder failure, not proof of identical conditional features.

The mouse prediction has 6.25 connected components on average versus 1.71 for CAMRI, but contour roughness is not higher (8.80 versus 9.93). Together with the 1.207 volume ratio, high recall, and FP heatmaps, this indicates a dominant smooth boundary expansion/leakage plus smaller isolated islands—not primarily jagged contours or missing anatomy. Spatially, mouse FP voxels concentrate rostrally (1,872,171 versus 890,066 caudally) and dorsally (1,536,622 versus 1,225,615 ventrally). Within brain extent, first-20% slices contain 945,798 FP voxels, consistent with the prior weak end-slice analysis.

## Bottleneck assessment

The evidence points most directly to the **frozen decoder/mask head at the transferred boundary distribution**, not a failure of coarse brain retrieval: recall remains high, whereas precision, surface overlap, and volume ratio degrade; coarse/fine latent centroids remain closely aligned; and query attention is not larger. Geometry is a plausible contributor because mouse occupancy is roughly half CAMRI occupancy and preprocessing therefore presents different object-to-field proportions. Lower image quality is not supported: mouse CNR and normalized boundary sharpness are higher. Calibration contributes, but thresholding alone cannot fix the broad high-probability exterior. Decoder capacity is not implicated by these data because the same decoder overfits the tiny set and performs strongly on CAMRI.

## Statistical analysis

The strongest random-forest permutation variable is `volume_ratio` (importance 1.53425). Mouse volume ratio correlates with Dice at r=-0.973 (p=5.05e-65). Mouse spacing, slice count, occupancy, and boundary sharpness are not significant at p<0.05; contrast has r=-0.232 (p=0.019). Level-1 attention spread is also not significant within mouse representatives (r=0.414, p=0.069, n=20). Cross-validated ridge R² is 0.981. Volume ratio is partly mathematically coupled to Dice, so this ranks the observed failure signature rather than identifying an independent cause.

## One allowed improvement

If only one controlled improvement were allowed, the evidence supports **mouse-domain calibration of the mask decision/boundary head while keeping the encoder frozen**. This targets the measured high-recall, low-precision, enlarged-volume failure directly. It should be tested as a separately declared adaptation experiment; it is not applied here.

## Evidence index

- `subject_diagnostics.csv`: geometry, intensity, surface, topology, and volume measurements for all subjects.
- `regional_errors.csv` and `regional_error_summary.csv`: spatial FP/FN localization.
- `threshold_sensitivity.csv`: inference-only analytical sweep.
- `query_attention.csv`: entropy, centroid, spread, sparsity, and attended volume.
- `latent_features.csv`, `latent_embeddings.csv`: selected fused features and PCA/t-SNE.
- `correlations.csv`, `regression_feature_importance.csv`, `statistical_summary.json`: statistical tests.
- `plots/`, `representative_figures/`, and `representative_inference/`: visual evidence and compressed probability/attention arrays.

## Limits

Attention weights are associations, not explanations of causal influence. The representative probability sweep is model-space because full native continuous probabilities were not retained by the original evaluation. PCA/t-SNE are based on representative pooled features and cannot establish that an encoder representation is unusable. No conclusion in this report relies on decreasing loss or post-hoc threshold selection.
