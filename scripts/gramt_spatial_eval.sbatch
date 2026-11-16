#!/bin/bash
#SBATCH --job-name=gramt_spat_eval
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,58,300,376,419,420]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out

# Sprint RQ1 (sac-ssamba/SPRINT.md): GRAM-T binaural-frame on SAR-SSL's simulated tasks, same protocol as the spatial
# SSAMBA reads (ssamba_spatial_snapshot_eval.sbatch): R8, 20 epochs, lr 1e-3 / 1e-4 (best by val), finetune + lineareval.
# Upstream: code/run_downstream.py --upstream gramt (wrapper code/gramt_spatial.py). Runs in the sac-ssamba venv.
#
# Usage, from the SAC-SAR-SSL root:
#   sbatch scripts/gramt_spatial_eval.sbatch <label> "<tasks>" ["<modes>"] [extra run_downstream args]
#   e.g. sbatch scripts/gramt_spatial_eval.sbatch gramt_bf TDOA
#        sbatch scripts/gramt_spatial_eval.sbatch gramt_smoke TDOA "finetune lineareval" --ds-nepoch 1 --ds-lr 1e-4
# Results: <exp>/<TASK>/<label>/downstream_test_res_<mode>.json. A requeue skips finished (task, mode) pairs.
set -uo pipefail
LABEL=$1; TASKS=($2); MODES=(${3:-finetune lineareval})
shift $(( $# < 3 ? $# : 3 ))
EXTRA=("$@")
[[ "$LABEL" =~ ^[A-Za-z0-9_]+$ ]] || { echo "label must be letters, digits, _: $LABEL"; exit 1; }
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
SAC_SSAMBA=${SAC_SSAMBA:-$HOME/sac-ssamba}
source "${SAC_SSAMBA}/env.sh"
export SAC_SSAMBA_SRC=${SAC_SSAMBA}/src
export HF_HUB_OFFLINE=1  # weights are cached in $HF_HOME; compute nodes must not depend on the hub
DATA=${SARSSL_DATA_DIR:-/scratch200/yotams2/SAC-SAR-SSL/data}
EXP=${SARSSL_EXP_ROOT:-/scratch200/yotams2/SAC-SAR-SSL/exp}
export TQDM_MININTERVAL=60
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local}: label ${LABEL} tasks ${TASKS[*]} modes ${MODES[*]} extra: ${EXTRA[*]}"
echo "git $(git -C "$REPO_ROOT" rev-parse --short HEAD)$(git -C "$REPO_ROOT" diff --quiet || echo -dirty)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for TASK in "${TASKS[@]}"; do
  for MODE in "${MODES[@]}"; do
    echo "===== $(date) ${TASK} ${MODE}"
    ARGS=(--simu-exp --upstream gramt --ds-trainmode "$MODE" --ds-nsimroom 8 --ds-task "$TASK" --time "$LABEL"
          --gpu-id "${CUDA_VISIBLE_DEVICES:-0}" --workers 0 --data-dir "$DATA" --exp-root "$EXP"
          --my-tmp --ds-lr 1e-3 1e-4 --ds-nepoch 20 "${EXTRA[@]}")
    cd "${REPO_ROOT}/code"
    DONE=${EXP}/${TASK}/${LABEL}/.train_done_${MODE}_R8
    RES=${EXP}/${TASK}/${LABEL}/downstream_test_res_${MODE}.json
    if [ ! -f "$DONE" ]; then
      $PYTHON -u run_downstream.py --ds-train "${ARGS[@]}" || { echo "===== ${TASK} ${MODE} train FAILED"; cd "$REPO_ROOT"; continue; }
      touch "$DONE"
    fi
    if [ -f "$RES" ] && [ "$RES" -nt "$DONE" ]; then
      echo "test result exists, not re-testing: $RES"
    else
      $PYTHON -u run_downstream.py --ds-test --test-mode cal_metric "${ARGS[@]}" || echo "===== ${TASK} ${MODE} test FAILED"
    fi
    cd "$REPO_ROOT"
    python3 scripts/collect_result.py --exp_root "$EXP" --task "$TASK" --time "$LABEL" --job_id "${SLURM_JOB_ID:-local}" || true
    bash scripts/sync_results.sh "results: ${TASK} ${LABEL} (gramt ${MODE} R8) job ${SLURM_JOB_ID:-local}" || true
  done
done
echo "$(date) all done"
