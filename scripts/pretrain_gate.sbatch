#!/bin/bash
#SBATCH --job-name=pre_gate
#SBATCH --account=public-users_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out

# Runs the evals of a chained pretraining (scripts/joint_pretrain.sbatch) once it has really finished.
# The pretraining runs in 12-h pieces; piece J writes <exp>/.chained_J = the next piece's id, and the last piece writes
# <exp>/.done. A plain afterok on the first piece would fire after piece 1 only, so this gate follows the chain:
#   .done exists                      -> run <eval script> (bash, from the SAC-SAR-SSL root), exit
#   newest piece still queued/running -> resubmit itself with afterany on that piece
#   newest piece ended without .done  -> the pretraining crashed or was cancelled: report and exit 1 (no evals)
#
# Usage, from the SAC-SAR-SSL root:
#   sbatch --dependency=afterany:<first piece> scripts/pretrain_gate.sbatch <exp dir> <first piece id> <eval script> [done file]
#   done file: the run's end marker inside <exp dir> (default .done = joint_pretrain; encoder_final.pth = ssamba_spatial_pretrain)
set -uo pipefail
EXP=$1 PIECE=$2 EVAL=$3 DONE=${4:-.done}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO_ROOT"
echo "$(date) gate job ${SLURM_JOB_ID:-local}: $EXP (first piece $PIECE) -> $EVAL"
while [ -s "$EXP/.chained_$PIECE" ]; do PIECE=$(cat "$EXP/.chained_$PIECE"); done
if [ -f "$EXP/$DONE" ]; then
	echo "pretraining done; running $EVAL"
	bash "$EVAL"
	exit $?
fi
state=$(sacct -j "$PIECE" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
case "$state" in
	PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|SUSPENDED|"")
		next=$(sbatch --parsable --dependency=afterany:"$PIECE" --chdir="$REPO_ROOT" "$REPO_ROOT/scripts/pretrain_gate.sbatch" "$EXP" "$PIECE" "$EVAL" "$DONE")
		echo "newest piece $PIECE is ${state:-unknown}; gate requeued as $next";;
	*)
		echo "ERROR: newest piece $PIECE ended ($state) without $EXP/$DONE: pretraining did not finish, no evals"
		exit 1;;
esac
