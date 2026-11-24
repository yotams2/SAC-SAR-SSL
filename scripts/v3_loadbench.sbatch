#!/bin/bash
#SBATCH --job-name=v3_loadbench
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# V3 load-time benchmark, staged node-local vs /scratch200 (CPU, analysis only; 11-24). Output exp/v3/loadbench.json
date; hostname; echo "TMPDIR=$TMPDIR"; df -h ${TMPDIR:-/tmp} | tail -1
cd code && /scratch200/yotams2/venvs/sac/bin/python -u v3_loadbench.py /scratch200/yotams2/SAC-SAR-SSL/exp/v3/loadbench.json
rc=$?; rm -rf ${TMPDIR:-/tmp}/v3_stage_$SLURM_JOB_ID; echo "$(date) exit $rc"
