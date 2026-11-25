# Sourced by the launch scripts (11-25, after the 400 GB quota filled and killed a run + evals mid-write).
# space_guard [min GB]: refuse to submit when /scratch200 has less than MIN_FREE_GB (default 60) free under the user
# quota. Fails closed: if quota cannot be read (or hangs > 30 s, seen on compute-0-425 11-25), it refuses too
# (override for one launch: SPACE_GUARD_SKIP=1).
space_free_gb() {  # free GB under the /scratch200 user quota (quota -w: one line, 1 KB blocks; '*' when over)
	timeout 30 quota -w -f /scratch200 2>/dev/null | awk '/scratch200/ {gsub(/[^0-9]/, "", $2); gsub(/[^0-9]/, "", $3);
		if ($3 > 0) { printf "%d\n", ($3 - $2) / 1048576; found = 1 } } END { exit !found }'
}
space_guard() {
	local min=${1:-${MIN_FREE_GB:-60}} free
	if [ -n "${SPACE_GUARD_SKIP:-}" ]; then echo "[space_guard] skipped (SPACE_GUARD_SKIP set)"; return 0; fi
	if ! free=$(space_free_gb); then
		echo "[space_guard] cannot read the /scratch200 quota (quota -w -f /scratch200): refusing to submit" >&2; return 1
	fi
	if [ "$free" -lt "$min" ]; then
		echo "[space_guard] /scratch200: ${free} GB free < ${min} GB: refusing to submit. Free space first" \
			"(scripts/backup/prune.py, ~/exp-records/quota_log.tsv)." >&2; return 1
	fi
	echo "[space_guard] /scratch200: ${free} GB free (>= ${min} GB), ok"
}
