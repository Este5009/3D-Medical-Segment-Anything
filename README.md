# 3D Medical Segment Anything

Research framework for query-conditioned 3D medical segmentation. The long-term
goal is a system that groups voxels into complete anatomical entities using
global, regional, and local volumetric evidence, then generalizes across
species, scanners, protocols, image quality, artifacts, pathology, and
anatomical variation.

Rodent brain extraction is the first controlled benchmark, not the final
objective. Current research keeps the verified RS2-Net Swin encoder frozen and
tests whether exactly one learned query, multi-scale attention, a top-down FPN,
and one mask head can learn robust cross-domain anatomical grouping.

The project prioritizes scientific validity and anatomical correctness over
dataset-specific metric optimization. Dice is reported alongside surface,
volume, terminal-slice, topology, and error-localization evidence.

Verified reproduction repositories and shared datasets are sibling resources
and are never copied or modified.
