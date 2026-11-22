#!/bin/bash
#SBATCH --job-name=head_check
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# CPU unit check for the per-encoder LayerNorm concat head (code/joint_head_check.py).
set -euo pipefail
source ~/sac-ssamba/env.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}/code"
$PYTHON -u joint_head_check.py
