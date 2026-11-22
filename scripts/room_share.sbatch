#!/bin/bash
#SBATCH --job-name=room_share
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --constraint=ib_dj
#SBATCH --exclude=compute-0-351
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# Check 3 (code/data_generation/room_share.py): dry-source tar for C3 + room share of the swap target's variance.
# CPU only; runs on an ib_dj node because WSJ0 is on /moshe-data.
set -euo pipefail
source ~/sac-ssamba/env.sh
D=/scratch200/yotams2/SAC-SAR-SSL
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}/code/data_generation"
$PYTHON -u room_share.py --tar $D/data/MicSig/simu_paired_pretrain.tar \
	--src_dir /moshe-data/blaufer-data/yotams2/SAC-SAR-SSL/data/SrcSig/wsj0/tr \
	--stats $D/exp/joint_pretrain/pilotC_swap/joint_final.pth \
	--dry_tar $D/data/MicSig/simu_paired_dry.tar --out $D/exp/room_share "$@"
echo "$(date) done"
