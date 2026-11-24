#!/bin/bash
#SBATCH --job-name=v3_sanity
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# V3 loader sanity check + dry run (CPU, analysis only; 11-24). Output exp/v3/sanity.json
date; hostname
mkdir -p /scratch200/yotams2/SAC-SAR-SSL/exp/v3
cd code && /scratch200/yotams2/venvs/sac/bin/python -u v3_sanity.py /scratch200/yotams2/SAC-SAR-SSL/exp/v3/sanity.json
echo "$(date) exit $?"
