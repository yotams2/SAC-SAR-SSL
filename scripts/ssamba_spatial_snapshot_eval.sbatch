#!/bin/bash
#SBATCH --job-name=ssamba_snap_eval
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,58,300,376,419,420]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out

# Downstream read of a spatial SSAMBA pretraining checkpoint (code/ssamba_spatial_pretrain.py, ipd16): TDOA then T60,
# R8, 20 epochs, lr 1e-3 / 1e-4 (best by val), the same settings as the random-init input probe (ssamba_input_probe.sbatch).
# Each (task, mode) goes through scripts/ssamba_spatial_downstream.sbatch (run as a plain script), which reports to the
# results branch and skips finished training on a requeue.
#
# Usage, from the SAC-SAR-SSL root:
#   sbatch scripts/ssamba_spatial_snapshot_eval.sbatch <encoder .pth> <label> [modes, default "finetune lineareval"]
#   random-init control at the same 256-frame setup:
#   sbatch scripts/ssamba_spatial_snapshot_eval.sbatch random <label> scratchLOW
set -uo pipefail
CKPT=$1 LABEL=$2
MODES=(${3:-finetune lineareval})
TASKS=(${EVAL_TASKS:-TDOA T60})
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO_ROOT"
COMMON=(--my-tmp --ds-lr 1e-3 1e-4 --ds-nepoch 20 --ssamba-fill none --ssamba-input ipd --ssamba-patch 16x16)
[ "$CKPT" = random ] && COMMON+=(--ssamba-nframes 256)
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local}: ckpt $CKPT label $LABEL tasks ${TASKS[*]} modes ${MODES[*]}"
for task in "${TASKS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "===== $(date) ${task} ${mode}"
    bash scripts/ssamba_spatial_downstream.sbatch "$task" "$CKPT" "$LABEL" 8 "$mode" "${COMMON[@]}" \
      || echo "===== ${task} ${mode} FAILED (exit $?)"
  done
done
echo "$(date) all done"
