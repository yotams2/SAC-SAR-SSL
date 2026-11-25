#!/bin/bash
#SBATCH --job-name=rir_encoder
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,300,376]  # 300 V100 + 376 RTX 6000: no bfloat16 (rir_encoder autocasts to bf16; 11-25)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# V1 room teacher: two-channel RIR MAE + frozen probes (code/rir_encoder.py). Usage: sbatch scripts/rir_encoder.sbatch <out dir> [args]
set -euo pipefail
source ~/sac-ssamba/env.sh
OUT=$1; shift
echo "$(date) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd ${SLURM_SUBMIT_DIR:-$(pwd)}/code
$PYTHON -u rir_encoder.py --out "$OUT" "$@"
echo "$(date) done"
