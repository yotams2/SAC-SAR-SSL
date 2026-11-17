#!/bin/bash
#SBATCH --job-name=joint_smoke
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,376]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# Smoke test of code/joint_pretrain.py: checks (masks, zero-gate fusion, leak), then each mode at batch 128 for 10
# steps on 1280 clips (full epoch-end path), spec with a mid-epoch resume. Output under exp/_smoke_joint/.
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$HOME/sac-ssamba/env.sh"; export SAC_SSAMBA_SRC=$HOME/sac-ssamba/src
TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_pretrain.tar
OUT=/scratch200/yotams2/SAC-SAR-SSL/exp/_smoke_joint; rm -rf "$OUT"; mkdir -p "$OUT"
echo "$(date) $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
cd "$REPO_ROOT/code"
C="--tar $TAR --num-train 1280 --num-val 256 --bs 128 --log-every 2 --ckpt-every 3 --num-workers 5"
"$PYTHON" -u joint_pretrain.py $C --exp-dir $OUT/spec --streams spec --max-steps 4
"$PYTHON" -u joint_pretrain.py $C --exp-dir $OUT/spec --streams spec --max-steps 10   # resumes from step 3
"$PYTHON" -u joint_pretrain.py $C --exp-dir $OUT/spat --streams spat --max-steps 10
"$PYTHON" -u joint_pretrain.py $C --exp-dir $OUT/joint --streams joint --fusion --max-steps 10
nvidia-smi --query-gpu=memory.used --format=csv
"$PYTHON" -u joint_pretrain_check.py $OUT/spat/spat_encoder_final.pth
ls -la $OUT/*/
echo "$(date) smoke done"
