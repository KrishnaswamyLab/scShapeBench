#!/bin/bash
# Build kNN graphs for the real track: one array task per dataset.
#
# Submit from the repository root (see docs/running_on_hpc.md):
#   sbatch --array=1-$(wc -l < ids.txt) --export=ALL,DATA_DIR=/path/to/data/scRNAseq scripts/hpc/build_knn.sbatch
#
# Environment variables (with defaults):
#   DATA_DIR   directory holding SCD-XXXX.h5ad        (data/scRNAseq)
#   OUT_DIR    method folder to write                  (submissions/scRNAseq/knn2)
#   K          neighbours per cell                     (2)
#   IDS_FILE   one dataset ID per line                 (ids.txt)
#SBATCH --job-name=scsb-knn
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/knn_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${IDS_FILE:-ids.txt}")
python examples/knn_submission.py \
  --data-dir "${DATA_DIR:-data/scRNAseq}" \
  --output-dir "${OUT_DIR:-submissions/scRNAseq/knn2}" \
  --k "${K:-2}" --ids "$ID"
