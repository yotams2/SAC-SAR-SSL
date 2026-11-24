#!/bin/bash
#SBATCH --job-name=v3_loadbench
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# V3 load-time benchmark, staged node-local vs /scratch200 (CPU, analysis only; 11-24). Output exp/v3/loadbench.json
set -uo pipefail
date; hostname
source scripts/job_tmp.sh  # TMPDIR on /scratch200, STAGE_DIR node-local; cleanup on exit
df -h "$(dirname "$STAGE_DIR")" | tail -1
cd code && job_tmp_run /scratch200/yotams2/venvs/sac/bin/python -u v3_loadbench.py /scratch200/yotams2/SAC-SAR-SSL/exp/v3/loadbench.json
echo "$(date) exit $?"
