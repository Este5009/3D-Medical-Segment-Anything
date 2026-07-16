# AGENTS.md

# 3D Medical Segment Anything

## Project Goal

Develop a foundation model capable of automatically segmenting any coherent anatomical structure from 3D medical images.

The long-term objective is to build a true 3D analogue of Segment Anything for medical imaging, replacing exhaustive prompt generation with learned object queries while preserving optional interactive refinement.

The project begins with rodent MRI skull stripping as the first benchmark before extending to multiple organs, tissues, lesions and anatomical structures.

---

# Research Philosophy

Always prioritize research quality over implementation speed.

Every architectural modification must be supported by controlled experiments and compared against a strong baseline.

Never replace multiple components simultaneously unless explicitly instructed.

---

# Current Research Stage

Stage 1

Evaluate whether a query-based decoder can replace a fixed segmentation decoder while reusing strong volumetric encoder features.

Current baseline:

- RS2-Net
- Swin Transformer encoder
- Fixed U-Net decoder

Research objective:

Replace only the decoder.

Everything else should remain unchanged.

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

Experiments must be reproducible.

---

# Performance Metrics

Primary:

- Dice
- IoU

Secondary:

- Precision
- Recall
- Hausdorff Distance
- Connected Components
- Inference Time
- GPU Memory

---

# Code Style

Prefer readability over compact code.

Explain every important tensor shape.

Avoid hidden magic.

Every module should be understandable independently.