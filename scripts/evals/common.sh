# Shared by the pilot eval scripts (sourced). Comparison set (11-19): all FROZEN.
#   spatial: TDOA T60 DRR lineareval, one job per variant (ssamba_spatial_snapshot_eval.sbatch, R8, 20 ep, lr 1e-3/1e-4)
#   speech:  SC2 + SID linear probes on the RQ2 subsets, one job per variant (+ random init where noted)
J=${JOINT_EXP:-/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain}
PROBES=/scratch200/yotams2/sac-ssamba/exp/_probes
TASKS="TDOA T60 DRR"
spatial() {  # <job name> <ckpt file that must exist> <label> [extra run_downstream args]
	CLEAN_FROZEN_CKPT=1 EVAL_TASKS="$TASKS" EXTRA_ARGS="${4:-}" sbatch --parsable --job-name=$1 scripts/ssamba_spatial_snapshot_eval.sbatch "$2" "$3" lineareval
}
probe() {  # <job name> <out jsonl> "<extra probe args>" <models...>
	local name=$1 out=$2 extra=$3; shift 3
	(cd ~/sac-ssamba && sbatch --parsable --job-name=$name scripts/ipd16_probe.sbatch "$PROBES/$out" "--tasks sc2,sid $extra" "$@")
}
