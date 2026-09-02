#!/bin/bash
# Full unattended pipeline for the UNETR-style real-decoder-depth level0
# experiment: train -> evaluate -> compare/figures -> report.
# Stops immediately on any failure (set -e) so a partial run is never
# silently reported as complete.
set -euo pipefail
cd "$(dirname "$0")/.."
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate rs2

LOG=outputs/unetr_style_level0_decoder/pipeline.log
mkdir -p outputs/unetr_style_level0_decoder
echo "=== $(date) : PIPELINE START ===" | tee -a "$LOG"

echo "=== $(date) : TRAIN ===" | tee -a "$LOG"
python scripts/train_unetr_style_level0.py 2>&1 | tee -a "$LOG"

echo "=== $(date) : EVALUATE ===" | tee -a "$LOG"
python scripts/evaluate_unetr_style_level0.py 2>&1 | tee -a "$LOG"

echo "=== $(date) : COMPARE / FIGURES ===" | tee -a "$LOG"
python scripts/compare_unetr_style_level0_results.py 2>&1 | tee -a "$LOG"

echo "=== $(date) : REPORT ===" | tee -a "$LOG"
python scripts/write_unetr_style_level0_report.py 2>&1 | tee -a "$LOG"

echo "=== $(date) : PIPELINE COMPLETE ===" | tee -a "$LOG"
