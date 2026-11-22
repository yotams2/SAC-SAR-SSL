#!/bin/bash
#SBATCH --job-name=swap_shufdiag
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,376]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# Check 4: swap shuffle diagnostic on the bin-centred scale (code/swap_shufdiag.py). Usage: sbatch scripts/swap_shufdiag.sbatch <out json> <label>=<exp>:<ckpt> ...
set -euo pipefail
source ~/sac-ssamba/env.sh
export SAC_SSAMBA_SRC=$HOME/sac-ssamba/src
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}/code"
$PYTHON -u swap_shufdiag.py "$@"
echo "$(date) done"
