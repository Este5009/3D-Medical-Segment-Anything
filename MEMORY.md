# MEMORY.md

# Long-Term Vision

Build a universal 3D medical segmentation foundation model.

Instead of predicting a single predefined organ, the model should automatically propose every coherent anatomical structure inside a medical volume.

A physician should only need to select the desired structure or optionally refine it.

---

# Core Architecture Vision

3D Volume

↓

Transformer Encoder

↓

Rich Volumetric Feature Pyramid

↓

Learned Object Queries

↓

Transformer Query Decoder

↓

Automatic 3D Mask Proposals

↓

Optional Prompt Refinement

---

# Current Roadmap

Phase 1

Reproduce state-of-the-art baselines.

Completed:

- MedSAM
- MedSAM2
- RS2-Net

Phase 2

Separate encoder from decoder.

Freeze encoder initially.

Train only a learned query decoder.

Compare against the original RS2-Net decoder.

Phase 3

Fine-tune encoder and query decoder jointly.

Phase 4

Train with multiple anatomical structures.

Phase 5

Add interactive prompt refinement.

---

# Research Principles

The encoder should learn anatomy.

Queries should retrieve objects.

The decoder should delineate precise voxel boundaries.

Avoid organ-specific assumptions whenever possible.

Train for anatomical objectness rather than fixed semantic classes.

---

# Current Dataset

Primary benchmark:

Rodent MRI skull stripping.

Ground truth:

Whole brain masks.

Future datasets will include multiple anatomical structures.

---

# Important Rule

Do not modify reproduction repositories.

They are permanent baselines.

Research code belongs only inside this repository.

---

# Long-Term Goal

Produce a publication-quality architecture capable of becoming a true 3D Segment Anything model for medical imaging.