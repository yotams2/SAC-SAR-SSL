#!/bin/bash
#SBATCH --job-name=sacform_smoke
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,58,300,376,419,420]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# Smoke for the SAC forms (branch sac-forms): unit checks, then 30 steps of each form + resume to step 40.
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source ~/sac-ssamba/env.sh; export SAC_SSAMBA_SRC=~/sac-ssamba/src
cd "$REPO_ROOT/code"
echo "git $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
$PYTHON -u sac_forms_check.py ~/SAC-SAR-SSL/code/model.py
T=/scratch200/yotams2/SAC-SAR-SSL/exp/_smoke_sacforms
TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_pretrain.tar
for form in cwcl sigmoid queue; do
  rm -rf "$T/$form"
  for ms in 30 40; do
    echo "===== $form to step $ms"
    $PYTHON -u ssamba_spatial_pretrain.py --tar $TAR --exp-dir $T/$form --num-workers 5 --nepoch 10 --sac-lambda 0.02 \
      --sac-form $form --sac-queue-warmup 10 --max-steps $ms --ckpt-every 10 --log-every 10 --num-val 256 2>&1 \
      | grep -aE '"step"|resume|Error|Traceback|nan|inf' | tail -4
  done
done
rm -rf "$T"
echo "===== smoke done"
