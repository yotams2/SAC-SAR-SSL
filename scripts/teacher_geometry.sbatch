#!/bin/bash
#SBATCH --job-name=teacher_geometry
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# Teacher representation analysis, steps 1-3 (Yotam 11-28; CPU, cached embeddings only): code/teacher_geometry.py.
# Usage, from the repo root: sbatch scripts/teacher_geometry.sbatch [<out dir>] [<cache dir>]
date; hostname
OUT=${1:-$HOME/SAC-SAR-SSL-results/results/analysis/teacher_geometry}
CACHE=${2:-/scratch200/yotams2/SAC-SAR-SSL/exp/teacher_geometry}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
/scratch200/yotams2/venvs/sac/bin/python -u code/teacher_geometry.py --out $OUT --cache $CACHE
echo "$(date) exit $?"
