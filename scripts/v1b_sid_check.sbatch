#!/bin/bash
#SBATCH --job-name=v1b_sid_check
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --time=0:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# V1b SID safety check (analysis only, CPU; 11-24). Usage: sbatch scripts/v1b_sid_check.sbatch [<sel dir>] [<out json>]
date; hostname
SEL=${1:-/scratch200/yotams2/SAC-SAR-SSL/exp/v1b_sigma_tau_A1}
OUT=${2:-$SEL/sid_check.json}
/scratch200/yotams2/venvs/sac/bin/python code/v1b_sid_check.py --sel $SEL --out $OUT
echo "$(date) exit $?"
