#!/bin/bash
# Score every method folder under SUBMISSIONS_DIR on the real track.
#
# Submit from the repository root (see docs/running_on_hpc.md):
#   sbatch scripts/hpc/score.sbatch
# On a GPU node add your cluster's GPU flags and DEVICE=cuda, e.g.
#   sbatch --partition=<gpu partition> --gpus=1 --export=ALL,DEVICE=cuda scripts/hpc/score.sbatch
#
# Environment variables (with defaults):
#   SUBMISSIONS_DIR  folder of method folders   (submissions/scRNAseq)
#   OUTPUT_DIR       where results are written  (results/real)
#   LABEL_TYPE       label set                  (high_support)
#   DEVICE           cpu or cuda                (cpu)
#SBATCH --job-name=scsb-score
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/score_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python run_benchmark.py --track real --real-mode all \
  --submissions-dir "${SUBMISSIONS_DIR:-submissions/scRNAseq}" \
  --label-type "${LABEL_TYPE:-high_support}" \
  --splits splits/folds_v1.json \
  --output "${OUTPUT_DIR:-results/real}" \
  --device "${DEVICE:-cpu}"
