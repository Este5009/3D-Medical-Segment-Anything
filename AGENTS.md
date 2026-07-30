# AGENTS.md

# 3D Medical Segment Anything

## Project Goal

Develop a generalizable, query-conditioned 3D medical segmentation system that
identifies complete anatomical structures by learning semantic and anatomical
grouping rather than memorizing dataset-specific boundaries.

The system should learn which voxels belong together as one anatomical entity,
produce segmentations supported by image evidence, and combine global, regional,
and local 3D context. Its intended scope includes variation across species,
scanners, institutions, acquisition protocols, image quality, contrast,
artifacts, pathology, and anatomy.

Rodent MRI skull stripping is a controlled first benchmark, not the final task.
The long-term system must support multiple anatomical targets through learned
queries and optional interactive refinement.

---

# Research Philosophy

Always prioritize research quality over implementation speed.

Every architectural modification must be supported by controlled experiments and compared against a strong baseline.

Never replace multiple components simultaneously unless explicitly instructed.

Optimize for anatomical correctness, robustness, generalization, and scientific
validity. Dice is an evaluation metric, not the objective of the project.
Avoid dataset-specific heuristics, threshold tuning, aggressive post-processing,
or anatomical shrinking that improves a benchmark without improving anatomical
understanding.

---

# Current Research Stage

Stage 1: controlled frozen-encoder query-decoder research.

Evaluate whether a query-based decoder can replace a fixed segmentation decoder while reusing strong volumetric encoder features.

Current baseline:

- RS2-Net
- Swin Transformer encoder
- Fixed U-Net decoder

The current one-query, multi-scale attention, top-down FPN decoder has
demonstrated sufficient capacity and useful CAMRI-to-Mouse transfer. The
immediate research question is whether improved supervision, optimization,
sampling, and augmentation can produce anatomically correct cross-domain
segmentations without changing the architecture.

Until controlled evidence says otherwise:

- keep the RS2-Net encoder frozen;
- keep exactly one learned query;
- keep the existing attention, FPN, and mask head unchanged;
- evaluate CAMRI and Mouse independently;
- attribute failures before proposing architecture changes.

---

# Coding Principles

- Modular architecture.
- Heavy documentation.
- Clear comments.
- Reproducible experiments.
- Deterministic execution whenever possible.
- Never modify baseline implementations directly.

---

# Repository Structure

src/
    models/
    decoders/
    queries/
    losses/
    datasets/
    evaluation/

configs/

experiments/

tests/

scripts/

outputs/

docs/

---

# Experiment Rules

Every experiment must

- save predictions
- save metrics
- save configuration
- save qualitative figures
- save logs
- use subject-level splits
- prevent known longitudinal leakage
- report each domain independently
- distinguish training changes from architecture changes

Experiments must be reproducible.

---

# Performance Metrics

Primary:

- Dice
- IoU

Secondary:

- Precision
- Recall
- HD95
- Average Surface Distance
- false-positive and false-negative volume
- volume error
- terminal-slice accuracy
- Connected Components
- Inference Time
- GPU Memory

---

# Code Style

Prefer readability over compact code.

Explain every important tensor shape.

Avoid hidden magic.

Every module should be understandable independently.
