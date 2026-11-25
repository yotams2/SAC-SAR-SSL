#!/bin/bash
#SBATCH --job-name=verify_ckpt
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-public-pool
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# Usage: sbatch scripts/verify_ckpt.sbatch <latest.pth> [<reference.pth>]
source ~/sac-ssamba/env.sh
$PYTHON -u scripts/verify_ckpt.py "$@"
