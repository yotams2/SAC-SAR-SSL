#!/bin/bash
#SBATCH --job-name=ssamba_spat_ds
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,58,300,376,419,420]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out

# SAR-SSL spatial downstream tasks (TDOA, DRR, T60, C50, ABS) on the simulated set, with the spatial encoder of a
# binaural dual-encoder SSAMBA checkpoint (sac-ssamba/src/binaural_dual_stream_sac) as the upstream:
# code/run_downstream.py --upstream ssamba_spatial (wrapper: code/ssamba_spatial.py), train then test (cal_metric),
# the same two steps as run_entire_flow.py. Runs in the sac-ssamba venv (it has mamba_ssm).
#
# Usage, from the SAC-SAR-SSL root:
#   sbatch scripts/ssamba_spatial_downstream.sbatch <TASK> <ckpt .pth | random> <label> [nsimroom=8] [trainmode=finetune] [extra run_downstream args]
#   label: names the run (--time); letters, digits and _ only. Results: <exp>/<TASK>/<label>/.
#   trainmode: finetune | lineareval (encoder frozen) | scratchLOW (random encoder, ckpt ignored)
#   extra args, e.g. --my-tmp (1 lr, 20 epochs, 1 trial: smoke test), --ssamba-fill zero, --ds-token cls, --seed 2
# Data: ${SARSSL_DATA_DIR}/MicSig/simu_ds, staged from /moshe-data by scripts/stage_simu_ds.sbatch.
# Preemption: run_downstream.py keeps finished (lr, trial) runs in a *_temporal.mat and skips them when requeued; the
# run that was cut restarts from its first epoch. A training step that finished is not repeated (.train_done marker).
set -euo pipefail
TASK=$1 CKPT=$2 LABEL=$3 NROOM=${4:-8} MODE=${5:-finetune}
shift $(( $# < 5 ? $# : 5 ))
[[ "$LABEL" =~ ^[A-Za-z0-9_]+$ ]] || { echo "label must be letters, digits, _: $LABEL"; exit 1; }
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
SAC_SSAMBA=${SAC_SSAMBA:-$HOME/sac-ssamba}
source "${SAC_SSAMBA}/env.sh"
export SAC_SSAMBA_SRC=${SAC_SSAMBA}/src
DATA=${SARSSL_DATA_DIR:-/scratch200/yotams2/SAC-SAR-SSL/data}
EXP=${SARSSL_EXP_ROOT:-/scratch200/yotams2/SAC-SAR-SSL/exp}
[ -d "${DATA}/MicSig/simu_ds/test" ] || { echo "no ${DATA}/MicSig/simu_ds: run scripts/stage_simu_ds.sbatch first"; exit 1; }
[ "$MODE" = scratchLOW ] || [ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }

echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local}: ${TASK} ${MODE} R${NROOM} label ${LABEL} ckpt ${CKPT} extra: $*"
echo "git $(git -C "$REPO_ROOT" rev-parse --short HEAD)$(git -C "$REPO_ROOT" diff --quiet || echo -dirty)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Data loading in the main process (--workers 0): DataLoader workers are forked from a process that already runs
# threads (tensorboardX writers, tqdm monitor, CUDA), and one inherited a held lock and hung in validation (job
# 23582088, py-spy: worker idle inside np.load). run_downstream.py has no __main__ guard, so spawn/forkserver can't be
# used. Loading costs little next to the GPU step (8 short wavs per batch).
export TQDM_MININTERVAL=${TQDM_MININTERVAL:-60}  # progress bars refresh once a minute instead of flooding the log
ARGS=(--simu-exp --upstream ssamba_spatial --ssamba-ckpt "$CKPT" --ds-trainmode "$MODE" --ds-nsimroom "$NROOM"
      --ds-task "$TASK" --time "$LABEL" --gpu-id "${CUDA_VISIBLE_DEVICES:-0}" --workers "${SARSSL_WORKERS:-0}"
      --data-dir "$DATA" --exp-root "$EXP" "$@")
cd "${REPO_ROOT}/code"
DONE=${EXP}/${TASK}/${LABEL}/.train_done_${MODE}_R${NROOM}
if [ ! -f "$DONE" ]; then
    $PYTHON -u run_downstream.py --ds-train "${ARGS[@]}"
    touch "$DONE"
fi
$PYTHON -u run_downstream.py --ds-test --test-mode cal_metric "${ARGS[@]}"

cd "$REPO_ROOT"
python3 scripts/collect_result.py --exp_root "$EXP" --task "$TASK" --time "$LABEL" --job_id "${SLURM_JOB_ID:-local}" || true
bash scripts/sync_results.sh "results: ${TASK} ${LABEL} (ssamba_spatial ${MODE} R${NROOM}) job ${SLURM_JOB_ID:-local}" || true
echo "$(date) done"
