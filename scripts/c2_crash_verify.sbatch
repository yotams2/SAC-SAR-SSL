#!/bin/bash
#SBATCH --job-name=c2_verify
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# C2 crash check + snapshots before the resume (code/c2_crash_verify.py). Usage: sbatch scripts/c2_crash_verify.sbatch <exp dir>
set -euo pipefail
source ~/sac-ssamba/env.sh
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}/code"
$PYTHON -u c2_crash_verify.py "$@"
echo "$(date) done"
