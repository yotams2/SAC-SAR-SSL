#!/bin/bash
#SBATCH --job-name=rir_compare
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-public-pool
#SBATCH --time=0:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# A′-uniform vs A′ design 1 vs v2 (code/rir_compare.py). Usage: sbatch scripts/rir_compare.sbatch <uniform dir> <out json>
set -euo pipefail
source ~/sac-ssamba/env.sh
cd ${SLURM_SUBMIT_DIR:-$(pwd)}/code
$PYTHON -u rir_compare.py --uniform "$1" --out "$2"
echo "$(date) done"
