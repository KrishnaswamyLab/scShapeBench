#!/bin/bash
# Synthetic track, step 1: run the baseline methods on one chunk of samples per array task.
# Outputs are cached per sample under $OUTPUT_DIR/cache, so step 2
# (synthetic_summary.sbatch) can write the tables without recomputing anything.
#
# Submit from the repository root (see docs/running_on_hpc.md); 1000 samples / CHUNK tasks:
#   sbatch --array=1-10 scripts/hpc/synthetic_methods.sbatch
#
# Environment variables (with defaults):
#   DATASET_DIR  synthetic samples                    (data/synthetic)
#   OUTPUT_DIR   cache and result tables              (results/synthetic)
#   CHUNK        samples per array task               (100)
#   METHODS      space-separated subset               (all five baselines)
#SBATCH --job-name=scsb-syn
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/synthetic_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-.venv}/bin/activate"  # .venv-synthetic for gtda_mapper (needs giotto-tda)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CHUNK=${CHUNK:-100}
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK ))
# Each task also rewrites results.csv/summary.* for its own chunk; step 2 overwrites them.
# shellcheck disable=SC2086
python run_benchmark_synthetic.py run-methods \
  --dataset-dir "${DATASET_DIR:-data/synthetic}" \
  --output-dir "${OUTPUT_DIR:-results/synthetic}" \
  ${METHODS:+--methods $METHODS} \
  --start "$START" --limit "$CHUNK" --no-pdfs
