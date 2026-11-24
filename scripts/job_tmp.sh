# Per-job temp dirs (approved 2026-11-24, after compute-0-291's full /tmp hung two V1 pieces). Source it from a job
# script, after `set -euo pipefail` and before starting Python:
#   TMPDIR    = /tmp/job_<job id> if /tmp has >= 5 GB free, else /scratch200/<user>/tmp/<job id>: Python tempfile and the DataLoader workers' multiprocessing dirs
#               follow it. /dev/shm (shared-memory tensors) is untouched.
#   STAGE_DIR = node-local ${STAGE_ROOT:-/tmp}/v3stage_<job id> (only V3's --v3-stage-local uses it; v3_data.stage_local falls back
#               to /scratch200 reads if it lacks space). Never $TMPDIR. Created lazily by the stager, not here.
# Both are removed on exit. A TERM trap is needed because Slurm preemption / time limit / scancel send SIGTERM: the
# trap (with the payload started by job_tmp_run, below) exits AT ONCE (143), so the script never reaches its post-run lines (e.g. swap_pretrain.sbatch's "crash:
# cancel the next piece"), and the EXIT trap cleans up. A SIGKILL leaves the dirs: prune.py (rule T1) removes per-job
# tmp dirs whose job is gone. JOB_TMP_DRY=1: print the resolved dirs and what Python sees, then exit (dry run).
JOB_TMP_ID=${SLURM_JOB_ID:-local$$}
# v2 (approved 11-25): node-local /tmp if it has >= JOB_TMP_MIN_GB (5) free at job start, else /scratch200 (NFS: works,
# but DataLoader worker exits log harmless ".nfs... busy" tracebacks). JOB_TMP_FORCE_SCRATCH=1 forces the fallback (tests).
JOB_TMP_MIN_GB=${JOB_TMP_MIN_GB:-5}
job_tmp_free_kb=$(df -Pk /tmp 2>/dev/null | awk 'NR == 2 {print $4}')
if [ -z "${JOB_TMP_FORCE_SCRATCH:-}" ] && [ "${job_tmp_free_kb:-0}" -ge $(( JOB_TMP_MIN_GB * 1024 * 1024 )) ] 2>/dev/null; then
	export TMPDIR=/tmp/job_${JOB_TMP_ID}
	job_tmp_why="node-local /tmp, $(( ${job_tmp_free_kb:-0} / 1024 / 1024 )) GB free >= ${JOB_TMP_MIN_GB} GB"
else
	export TMPDIR=/scratch200/${USER}/tmp/${JOB_TMP_ID}
	job_tmp_why="fallback: /tmp $(( ${job_tmp_free_kb:-0} / 1024 / 1024 )) GB free < ${JOB_TMP_MIN_GB} GB"
	[ -n "${JOB_TMP_FORCE_SCRATCH:-}" ] && job_tmp_why="fallback forced by JOB_TMP_FORCE_SCRATCH (/tmp $(( ${job_tmp_free_kb:-0} / 1024 / 1024 )) GB free)"
fi
export STAGE_DIR=${STAGE_ROOT:-/tmp}/v3stage_${JOB_TMP_ID}  # always per job id: chained pieces inherit the env
mkdir -p "$TMPDIR" 2>/dev/null || {  # /tmp unusable after all (filled meanwhile, read-only): fall back
	job_tmp_why="fallback: mkdir $TMPDIR failed"; export TMPDIR=/scratch200/${USER}/tmp/${JOB_TMP_ID}; mkdir -p "$TMPDIR"; }
_job_tmp_cleanup() { rm -rf "$TMPDIR" "$STAGE_DIR"; }
trap _job_tmp_cleanup EXIT
# bash runs a trap only after a FOREGROUND child returns, and Slurm SIGKILLs 30 s (KillWait) after SIGTERM: run the
# payload with job_tmp_run (background + wait; `wait` is interrupted by the trap), which forwards SIGTERM to it.
JOB_TMP_CHILD=
_job_tmp_term() {
	echo "$(date) SIGTERM: stopping ${JOB_TMP_CHILD:-nothing}, removing temp dirs, exiting 143"
	if [ -n "$JOB_TMP_CHILD" ]; then kill -TERM "$JOB_TMP_CHILD" 2>/dev/null; wait "$JOB_TMP_CHILD" 2>/dev/null; fi
	exit 143
}
trap _job_tmp_term TERM
job_tmp_run() {  # job_tmp_run <cmd> [args]: run in the background, wait, return its exit code
	"$@" &
	JOB_TMP_CHILD=$!
	local rc=0
	wait "$JOB_TMP_CHILD" || rc=$?
	JOB_TMP_CHILD=
	return $rc
}
echo "[job_tmp] TMPDIR=$TMPDIR ($job_tmp_why) STAGE_DIR=$STAGE_DIR"
if [ -n "${JOB_TMP_DRY:-}" ]; then
	"${PYTHON:-python3}" -c "import tempfile, multiprocessing.util as u, os; print('[job_tmp] python tempfile', tempfile.gettempdir(), '| multiprocessing', u.get_temp_dir(), '| STAGE_DIR', os.environ['STAGE_DIR'])"
	exit 0
fi
