#!/bin/bash
# Runs all 4 points of the embedding_dim/level0_width sweep sequentially:
# train -> evaluate for emb=32, then 48, then 64, then 96. Stops immediately
# on any failure so a partial sweep is never silently reported as complete.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=outputs/level0_width_sweep/pipeline_gpu_stdout.log
mkdir -p outputs/level0_width_sweep
echo "=== $(date) : SWEEP START ===" | tee -a "$LOG"

for emb in 32 48 64 96; do
  echo "=== $(date) : TRAIN emb=$emb ===" | tee -a "$LOG"
  python3 scripts/train_level0_width_sweep.py --config configs/level0_width_sweep_emb${emb}.yaml 2>&1 | tee -a "$LOG"

  echo "=== $(date) : EVALUATE emb=$emb ===" | tee -a "$LOG"
  python3 scripts/evaluate_level0_width_sweep.py --config configs/level0_width_sweep_emb${emb}.yaml 2>&1 | tee -a "$LOG"
done

echo "=== $(date) : SWEEP COMPLETE ===" | tee -a "$LOG"
