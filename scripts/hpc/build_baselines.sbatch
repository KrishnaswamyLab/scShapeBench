#!/bin/bash
# Build the scReebTower, PAGA and Mapper real-track graphs: one array task per dataset.
#
# Submit from the repository root (see docs/running_on_hpc.md):
#   sbatch --array=1-$(wc -l < ids.txt) --export=ALL,DATA_DIR=/path/to/data/scRNAseq scripts/hpc/build_baselines.sbatch
#
# Environment variables (with defaults):
#   DATA_DIR   directory holding SCD-XXXX.h5ad        (data/scRNAseq)
#   OUT_DIR    submissions root (one folder per method) (submissions/scRNAseq)
#   METHODS    space-separated subset                  (screebtower paga mapper)
#   IDS_FILE   one dataset ID per line                 (ids.txt)
#SBATCH --job-name=scsb-baselines
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/baselines_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${IDS_FILE:-ids.txt}")
# shellcheck disable=SC2086
python scripts/build_real_submissions.py \
  --data-dir "${DATA_DIR:-data/scRNAseq}" \
  --output-dir "${OUT_DIR:-submissions/scRNAseq}" \
  --methods ${METHODS:-screebtower paga mapper} --ids "$ID"
