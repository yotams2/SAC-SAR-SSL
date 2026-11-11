#!/bin/bash
#SBATCH --job-name=check_ssamba_spat
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,58,300,376,419,420]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out

# Usage, from the SAC-SAR-SSL root: sbatch scripts/check_ssamba_spatial.sbatch <ckpt .pth> [binaural wav]
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
SAC_SSAMBA=${SAC_SSAMBA:-$HOME/sac-ssamba}
source "${SAC_SSAMBA}/env.sh"
export SAC_SSAMBA_SRC=${SAC_SSAMBA}/src
cd "${REPO_ROOT}/code"
# the downstream script's imports, in this venv (catches missing packages before a long job)
$PYTHON -c "import dataset, learner, model, common.utils, tensorboardX, scipy.io; print('SAR-SSL imports OK')"
$PYTHON -u "${REPO_ROOT}/scripts/check_ssamba_spatial.py" "$@"
