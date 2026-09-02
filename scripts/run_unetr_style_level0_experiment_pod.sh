#!/bin/bash
# Pod variant of run_unetr_style_level0_experiment.sh -- no conda env on this
# GPU pod, uses the system python3 (already has the matched package set
# installed). Same pipeline, same scripts, same config -- only the
# interpreter invocation differs from the local CPU orchestration script.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=outputs/unetr_style_level0_decoder/pipeline_gpu.log
mkdir -p outputs/unetr_style_level0_decoder
echo "=== $(date) : GPU PIPELINE START ===" | tee -a "$LOG"

echo "=== $(date) : TRAIN ===" | tee -a "$LOG"
python3 scripts/train_unetr_style_level0.py 2>&1 | tee -a "$LOG"

echo "=== $(date) : EVALUATE ===" | tee -a "$LOG"
python3 scripts/evaluate_unetr_style_level0.py 2>&1 | tee -a "$LOG"

echo "=== $(date) : COMPARE / FIGURES ===" | tee -a "$LOG"
python3 scripts/compare_unetr_style_level0_results.py 2>&1 | tee -a "$LOG"

echo "=== $(date) : REPORT ===" | tee -a "$LOG"
python3 scripts/write_unetr_style_level0_report.py 2>&1 | tee -a "$LOG"

echo "=== $(date) : GPU PIPELINE COMPLETE ===" | tee -a "$LOG"
