#!/bin/bash
#SBATCH --job-name=collect_sarssl
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --constraint=ib_dj
#SBATCH --exclude=compute-0-351
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out

# Report every run under /moshe-data/.../SAC-SAR-SSL/exp to the results branch. /moshe-data is mounted only on the
# 19 nodes with the ib_dj feature (all but compute-0-351), so this can't run from the Jupyter node. Submit from the repo root. Safe to rerun:
# a run's folder is rewritten, and the commit stays empty if nothing changed.
EXP=${1:-/moshe-data/blaufer-data/yotams2/SAC-SAR-SSL/exp}
JOB_ID=${2:-runai}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
[ -d "$EXP" ] || { echo "no $EXP on $(hostname)"; exit 1; }
python3 "${REPO_ROOT}/scripts/collect_result.py" --exp_root "$EXP" --job_id "$JOB_ID"
bash "${REPO_ROOT}/scripts/sync_results.sh" "results: SAR-SSL runs from $(basename "$(dirname "$EXP")")/exp"
