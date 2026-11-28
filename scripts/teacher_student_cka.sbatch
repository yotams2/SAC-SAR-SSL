#!/bin/bash
#SBATCH --job-name=teacher_student_cka
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18,compute-0-300   # Blackwell (no kernels); V100 (no bf16)
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nice=100
#SBATCH --open-mode=append
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# Teacher representation analysis, step 4 (Yotam 11-28; nice 100, below all evals): code/teacher_student_cka.py.
# Clips + checkpoints staged node-local. Needs <cache>/heldout_cues.npz from teacher_geometry.sbatch.
set -euo pipefail
source ~/sac-ssamba/env.sh
export HF_HUB_OFFLINE=1
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
OUT=$HOME/SAC-SAR-SSL-results/results/analysis/teacher_geometry
CACHE=/scratch200/yotams2/SAC-SAR-SSL/exp/teacher_geometry
E=/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain
R=()
for x in v1bA_l01:pilotc_v1bA_l01 v1bHC_l01:pilotc_v1bHC_l01 C_v2_10ep:pilotC_v2r v1bA30noanc_s2:pilotc_v1bA30noanc_l01_s2; do
	R+=("${x%%:*}=$E/${x#*:}/spec_encoder_final.pth,$E/${x#*:}/spat_encoder_final.pth")
done
export TMPDIR_LOCAL=$(mktemp -d /tmp/tcka_${SLURM_JOB_ID:-x}_XXXX 2>/dev/null || echo /scratch200/yotams2/tmp/tcka_${SLURM_JOB_ID:-x})
cd code
$PYTHON -u teacher_student_cka.py --cues $CACHE/heldout_cues.npz --out $OUT --cache $CACHE "${R[@]}"
echo "$(date) done"
