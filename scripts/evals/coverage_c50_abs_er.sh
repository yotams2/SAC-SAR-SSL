#!/bin/bash
# Downstream coverage (Yotam 11-28; low priority): C50 + ABS (frozen lineareval, R8, standard protocol, new eval code)
# on each key row's spatial encoder (+ leakage: its spectral encoder on C50 / ABS), and binaural IEMOCAP ER (fold 1, the
# speech probe) on its spectral encoder (+ leakage: its spatial encoder on ER). Labels = the rows' existing labels, so
# C50 / ABS land next to TDOA / T60 / DRR in RUNS.md. All jobs nice 100, after the 30-ep evals (DEP).
# Usage, from the SAC-SAR-SSL root: DEP=<jobid[:jobid...]> bash scripts/evals/coverage_c50_abs_er.sh [--dry]
set -euo pipefail
J=/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain
PROBES=/scratch200/yotams2/sac-ssamba/exp/_probes
DRY=${1:-}
D=(--nice=100 ${DEP:+--dependency=afterany:$DEP})
sp() {  # <label> <ckpt> [extra run_downstream args]
	local cmd=(sbatch --parsable "${D[@]}" --job-name=cov_$1 scripts/ssamba_spatial_snapshot_eval.sbatch "$2" "$1" lineareval)
	[ -n "$DRY" ] && { echo "CLEAN_FROZEN_CKPT=1 EVAL_TASKS='C50 ABS' EXTRA_ARGS='${3:-}' ${cmd[*]}"; return; }
	echo "$1 $(CLEAN_FROZEN_CKPT=1 EVAL_TASKS="C50 ABS" EXTRA_ARGS="${3:-}" "${cmd[@]}")"
}
ER_GATE=  # the first row's two ER jobs gate the others (afterok): ER has not run through the joint encoders before
er() {  # <label> "<input_repr args>" <model>
	local dep=${DEP:+afterany:$DEP}
	[ -n "$ER_GATE" ] && dep=${dep:+$dep,}afterok:$ER_GATE
	local cmd=(sbatch --parsable --nice=100 ${dep:+--dependency=$dep} --job-name=cov_$1 scripts/ipd16_probe.sbatch "$PROBES/$1.jsonl" "--tasks iemocap $2" "$3")
	[ -n "$DRY" ] && { echo "(cd ~/sac-ssamba && ${cmd[*]})"; return; }
	echo "$1 $(cd ~/sac-ssamba && "${cmd[@]}")"
}
# row label : spec ckpt : spat ckpt
ROWS=(
	"jointA:$J/pilotA_spec/spec_encoder_final.pth:$J/pilotA_spat/spat_encoder_final.pth"
	"jointCv2r:$J/pilotC_v2r/spec_encoder_final.pth:$J/pilotC_v2r/spat_encoder_final.pth"
	"jointC30v2r:$J/pilotC30_v2r/spec_encoder_final.pth:$J/pilotC30_v2r/spat_encoder_final.pth"
	"jointcv1bAl01:$J/pilotc_v1bA_l01/spec_encoder_final.pth:$J/pilotc_v1bA_l01/spat_encoder_final.pth"
	"jointcv1bHCl01:$J/pilotc_v1bHC_l01/spec_encoder_final.pth:$J/pilotc_v1bHC_l01/spat_encoder_final.pth"
	"jointcv1bBl01:$J/pilotc_v1bB_l01/spec_encoder_final.pth:$J/pilotc_v1bB_l01/spat_encoder_final.pth"
	"jointcv1al1:$J/pilotc_v1a_l1/spec_encoder_final.pth:$J/pilotc_v1a_l1/spat_encoder_final.pth"
	"jointcv1bAnoancl01:$J/pilotc_v1bA_noanc_l01/spec_encoder_final.pth:$J/pilotc_v1bA_noanc_l01/spat_encoder_final.pth"
	"jointcv1bA30noancl01s1:$J/pilotc_v1bA30noanc_l01_s1/spec_encoder_final.pth:$J/pilotc_v1bA30noanc_l01_s1/spat_encoder_final.pth"
	"jointcv1bA30noancl01s2:$J/pilotc_v1bA30noanc_l01_s2/spec_encoder_final.pth:$J/pilotc_v1bA30noanc_l01_s2/spat_encoder_final.pth"
)
for r in "${ROWS[@]}"; do
	IFS=: read -r L SPEC SPAT <<<"$r"
	for f in "$SPEC" "$SPAT"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
	sp ${L}_spat "$SPAT"
	sp ${L}_spec_on_spatial "$SPEC" "--upstream joint --joint-mode spec --joint-ckpt $SPEC"
	if [ -z "$ER_GATE" ] && [ -z "$DRY" ]; then
		a=$(er ${L}_spec_er "--input_repr joint_spec" "$SPEC"); echo "$a"
		b=$(er ${L}_spat_er "" "$SPAT"); echo "$b"
		ER_GATE=${a##* }:${b##* }
	else
		er ${L}_spec_er "--input_repr joint_spec" "$SPEC"
		er ${L}_spat_er "" "$SPAT"
	fi
done
