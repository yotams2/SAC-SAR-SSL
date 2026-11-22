#!/bin/bash
#SBATCH --job-name=swap_pre
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,376]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out

# Pilot C swap-reconstruction pretraining (code/swap_pretrain.py) on the paired scene x utterance set
# (scripts/gen_simu_paired.sbatch -> MicSig/simu_paired_pretrain.tar). Runs in the sac-ssamba venv (mamba_ssm).
#
# Usage, from the SAC-SAR-SSL root:  sbatch scripts/swap_pretrain.sbatch <label> [swap_pretrain.py args]
# Output: ${SARSSL_EXP_ROOT}/joint_pretrain/<label>/ (resumes from latest.pth if present).
# Chunks: the job runs in 12-h pieces (short jobs start sooner through backfill). On its first start each job queues
# the next piece with afterany on itself; a piece that finds .done exits at once. Preemption requeues
# the same job id, which resumes from latest.pth (saved every 200 steps, ~20 min).
set -euo pipefail
LABEL=$1; shift
[[ "$LABEL" =~ ^[A-Za-z0-9_]+$ ]] || { echo "label must be letters, digits, _: $LABEL"; exit 1; }
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
SAC_SSAMBA=${SAC_SSAMBA:-$HOME/sac-ssamba}
source "${SAC_SSAMBA}/env.sh"
export SAC_SSAMBA_SRC=${SAC_SSAMBA}/src
TAR=${SARSSL_DATA_DIR:-/scratch200/yotams2/SAC-SAR-SSL/data}/MicSig/simu_paired_pretrain.tar
EXP=${SARSSL_EXP_ROOT:-/scratch200/yotams2/SAC-SAR-SSL/exp}/joint_pretrain/${LABEL}
[ -f "$TAR" ] || { echo "no $TAR: run scripts/gen_simu_paired.sbatch first"; exit 1; }
mkdir -p "$EXP"
if [ -f "$EXP/.done" ]; then echo "$(date) $LABEL already finished"; exit 0; fi

# queue the next 12-h piece once per job id (a requeued job keeps its id and must not queue a second one)
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${NO_CHAIN:-}" ] && [ ! -f "$EXP/.chained_${SLURM_JOB_ID}" ]; then
	next=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} --chdir="$REPO_ROOT" \
		"$REPO_ROOT/scripts/swap_pretrain.sbatch" "$LABEL" "$@")
	echo "$next" > "$EXP/.chained_${SLURM_JOB_ID}"
	echo "next piece: $next"
fi

echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1): $LABEL $*"
echo "git $(git -C "$REPO_ROOT" rev-parse --short HEAD)$(git -C "$REPO_ROOT" diff --quiet || echo -dirty)"
cd "$REPO_ROOT/code"
rc=0
"$PYTHON" -u swap_pretrain.py --tar "$TAR" --exp-dir "$EXP" --num-workers 5 "$@" || rc=$?
echo "$(date) exit $rc"; if [ "$rc" -eq 0 ] && [ -z "${NO_DONE:-}" ]; then touch "$EXP/.done"; fi
# A crash (Python exits on its own with an error) stops the chain, so a bug can't loop through 12-h pieces.
# Time limit and preemption kill the job before this line runs, so they keep the next piece.
if [ "$rc" -ne 0 ] && [ -n "${SLURM_JOB_ID:-}" ] && [ -s "$EXP/.chained_${SLURM_JOB_ID}" ]; then
	scancel "$(cat "$EXP/.chained_${SLURM_JOB_ID}")" && echo "crash: cancelled next piece $(cat "$EXP/.chained_${SLURM_JOB_ID}")"
fi
exit $rc
