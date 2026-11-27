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
#   env EVAL_TASKS="TDOA T60 DRR" picks tasks; EXTRA_ARGS is appended to every run_downstream call;
#   CLEAN_FROZEN_CKPT=1 deletes the model copies of finished lineareval runs (results kept).
set -uo pipefail
CKPT=$1 LABEL=$2
MODES=(${3:-finetune lineareval})
TASKS=(${EVAL_TASKS:-TDOA T60})
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO_ROOT"
COMMON=(--my-tmp --ds-lr 1e-3 1e-4 --ds-nepoch 20 --ssamba-fill none --ssamba-input ipd --ssamba-patch 16x16)
[ "$CKPT" = random ] && COMMON+=(--ssamba-nframes 256)
COMMON+=(${EXTRA_ARGS:-})  # e.g. EXTRA_ARGS="--upstream joint --joint-mode concat --joint-ckpt a.pth,b.pth"
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local}: ckpt $CKPT label $LABEL tasks ${TASKS[*]} modes ${MODES[*]}"

# Node-local I/O (11-27; spatial evals went from ~25 min to 2.5-9 h per task on the same nodes, all of it /scratch200
# I/O: ~20 MB/s writes, ~30 ms per small-file read). EVAL_LOCAL=1 (default):
#   - simu_ds (train R1-R8, val, test: ~1.6 GB) is copied once to node-local /tmp; other data entries are symlinks.
#   - the exp dir lives in /tmp while a task runs, so the per-epoch weights (model<N> / latest / best, ~0.8 GB each,
#     2-3 per epoch) never touch /scratch200. After each task (and on SIGTERM) results, curves, *_temporal.mat and
#     best / ensemble weights (only while the frozen test is still pending) go back to /scratch200; per-epoch
#     model<N>.tar and latest_model.tar never do (evals never resume from them).
# Same code, data and settings; only where files live. EVAL_LOCAL=0: the old behaviour (everything on /scratch200).
# EVAL_DRY=1: stage + verify the copy (file counts vs /scratch200), print the commands, exit (no GPU work).
SCR_DATA=${SARSSL_DATA_DIR:-/scratch200/yotams2/SAC-SAR-SSL/data}
SCR_EXP=${SARSSL_EXP_ROOT:-/scratch200/yotams2/SAC-SAR-SSL/exp}
EVAL_LOCAL=${EVAL_LOCAL:-1}
CUR_TASK=
run=  # run=job_tmp_run when job_tmp.sh is sourced (SIGTERM-safe payload)
if [ "$EVAL_LOCAL" = 1 ]; then
  JOB_TMP_MIN_GB=${JOB_TMP_MIN_GB:-20}
  source "$REPO_ROOT/scripts/job_tmp.sh"
  case $TMPDIR in
    /tmp/*) ;;
    *) echo "[eval_local] no node-local /tmp ($TMPDIR): reading and writing /scratch200 as before"; EVAL_LOCAL=0 ;;
  esac
  run=job_tmp_run
fi
sync_back() {  # <task>: local task dir -> /scratch200 (small files first, weights, then the .train_done markers)
  local t=$1 src=$TMPDIR/exp/$1/$LABEL dst=$SCR_EXP/$1/$LABEL
  [ -d "$src" ] || return 0
  mkdir -p "$dst"
  rsync -a --exclude '*.tar' --exclude '.train_done_*' "$src/" "$dst/"
  rsync -a --include '*/' --include 'best_model.tar' --include 'ensemble_model.tar' --exclude '*' "$src/" "$dst/"
  rsync -a --include '*/' --include '.train_done_*' --exclude '*' "$src/" "$dst/"
  echo "[eval_local] $(date '+%H:%M:%S') $t: results copied back to $dst"
}
if [ "$EVAL_LOCAL" = 1 ]; then
  _job_tmp_cleanup() { [ -n "$CUR_TASK" ] && sync_back "$CUR_TASK"; rm -rf "$TMPDIR" "$STAGE_DIR"; }
  LD=$TMPDIR/data; S=$SCR_DATA/MicSig/simu_ds; L=$LD/MicSig/simu_ds
  mkdir -p "$L/train" "$TMPDIR/exp"
  for e in "$SCR_DATA"/*; do [ "${e##*/}" = MicSig ] || ln -s "$e" "$LD/"; done
  for e in "$SCR_DATA"/MicSig/*; do [ "${e##*/}" = simu_ds ] || ln -s "$e" "$LD/MicSig/"; done
  for e in "$S"/*; do case ${e##*/} in train|val|test) ;; *) ln -s "$e" "$L/" ;; esac; done
  for e in "$S"/train/*; do case ${e##*/} in R[1-8]) ;; *) ln -s "$e" "$L/train/" ;; esac; done  # R8 reads R1-R8 only
  t0=$SECONDS
  cp -r "$S"/train/R[1-8] "$L/train/" && cp -r "$S/val" "$S/test" "$L/" || { echo "[eval_local] staging failed: exiting"; exit 1; }
  for d in train/R1 train/R2 train/R3 train/R4 train/R5 train/R6 train/R7 train/R8 val test; do
    a=$(find "$S/$d" -type f | wc -l); b=$(find "$L/$d" -type f | wc -l)
    [ "$a" = "$b" ] || { echo "[eval_local] staging check failed: $d $b files, /scratch200 has $a: exiting"; exit 1; }
  done
  echo "[eval_local] simu_ds (R1-R8, val, test) staged to $L in $(( SECONDS - t0 )) s ($(du -sh "$L" | cut -f1)); exp in $TMPDIR/exp"
  export SARSSL_DATA_DIR=$LD SARSSL_EXP_ROOT=$TMPDIR/exp
fi
for task in "${TASKS[@]}"; do
  if [ "$EVAL_LOCAL" = 1 ] && [ -d "$SCR_EXP/$task/$LABEL" ]; then  # resume state of an earlier attempt (requeue / rerun)
    mkdir -p "$TMPDIR/exp/$task"
    rsync -a --exclude 'model[0-9]*.tar' --exclude 'latest_model.tar' "$SCR_EXP/$task/$LABEL" "$TMPDIR/exp/$task/"
  fi
  CUR_TASK=$task
  for mode in "${MODES[@]}"; do
    echo "===== $(date) ${task} ${mode}"
    if [ -n "${EVAL_DRY:-}" ]; then
      echo "[dry] SARSSL_DATA_DIR=${SARSSL_DATA_DIR:-} SARSSL_EXP_ROOT=${SARSSL_EXP_ROOT:-} bash scripts/ssamba_spatial_downstream.sbatch $task $CKPT $LABEL 8 $mode ${COMMON[*]}"
    else
      $run bash scripts/ssamba_spatial_downstream.sbatch "$task" "$CKPT" "$LABEL" 8 "$mode" "${COMMON[@]}" \
        || echo "===== ${task} ${mode} FAILED (exit $?)"
    fi
    # CLEAN_FROZEN_CKPT=1: once a frozen (lineareval) result is written, drop its 3 full-model copies per lr
    # (~0.8 GB each: frozen encoder + linear head; the head retrains in ~1 h). Results/curves are kept.
    D=${SARSSL_EXP_ROOT:-/scratch200/yotams2/SAC-SAR-SSL/exp}/${task}/${LABEL}
    if [ -n "${CLEAN_FROZEN_CKPT:-}" ] && [ "$mode" = lineareval ] && [ -f "$D/downstream_test_res_lineareval.json" ]; then
      find "$D" -path '*lineareval-*' -name '*_model.tar' -delete && echo "===== ${task}: frozen-eval model copies removed"
    fi
  done
  if [ "$EVAL_LOCAL" = 1 ]; then
    sync_back "$task"; CUR_TASK=
    D=$SCR_EXP/$task/$LABEL  # weights an earlier attempt left on /scratch200
    if [ -n "${CLEAN_FROZEN_CKPT:-}" ] && [ -f "$D/downstream_test_res_lineareval.json" ]; then
      find "$D" -path '*lineareval-*' -name '*model*.tar' -delete
    fi
  fi
done
[ -n "${EVAL_DRY:-}" ] && echo "[dry] done (nothing trained)"
echo "$(date) all done"
