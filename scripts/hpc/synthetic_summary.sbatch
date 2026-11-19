#!/bin/bash
# Synthetic track, step 2: write the tables from the cached method outputs.
#
#   $OUTPUT_DIR/results.csv, summary.csv, summary.tex   GED and persistence similarity per method
#   $OUTPUT_DIR/eval_<method>.json                      full report per method, incl. the
#                                                        Wasserstein terms behind persistence similarity
#
# Submit from the repository root after step 1 (see docs/running_on_hpc.md):
#   sbatch --dependency=afterok:<step-1 array jobid> scripts/hpc/synthetic_summary.sbatch
#
# Environment variables (with defaults): DATASET_DIR (data/synthetic), OUTPUT_DIR (results/synthetic),
# METHODS (all five baselines).
#SBATCH --job-name=scsb-syn-sum
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/synthetic_summary_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-.venv}/bin/activate"  # .venv-synthetic for gtda_mapper (needs giotto-tda)

DATASET_DIR=${DATASET_DIR:-data/synthetic}
OUTPUT_DIR=${OUTPUT_DIR:-results/synthetic}

# All samples are cached by step 1, so this only evaluates and writes the tables.
# shellcheck disable=SC2086
python run_benchmark_synthetic.py run-methods \
  --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_DIR" \
  ${METHODS:+--methods $METHODS}

for method_dir in "$OUTPUT_DIR"/cache/*/; do
  method=$(basename "$method_dir")
  python run_benchmark_synthetic.py evaluate dataset \
    --dataset-dir "$DATASET_DIR" \
    --predictions-dir "$method_dir/pointcloud" \
    --output-path "$OUTPUT_DIR/eval_${method}.json" \
    --skip-manifold-render \
    || echo "evaluate dataset failed for $method (a sample may have no cached prediction)"
done
