#!/bin/bash
# V1a / V1b launcher (built 2026-11-23, rewritten 11-24 for the decided teachers; launch only after the C v2 baseline's
# evals have been compared with C on v1, per your call).
# Runs, all 10 epochs on paired v2 with the new recipe (cosine + warmup 0.05 + running-median clip x3), the base's config
# (BASE=C: as pilotC_v2r; BASE=C2: + --ctr-lambda 1), gradient share logged every 500 steps:
#   v1bA_l1 / v1bA_l01 : V1b, teacher-cue SAC on both encoders, lambda 1 / 0.1 each; spatial teacher = learned A′
#                        (rir_encoder a1_design1, cosine cues, full 768-d spatial PCA) -- the method
#   v1bB_l1 / v1bB_l01 : same with the oracle B vectors (Euclidean cues, no spatial PCA) -- analysis row
#   v1a_l1             : V1a, --lat-lambda 1 (dry-speech teacher tokens + A′ RIR embedding, linear predictors, cosine)
#   (v1a_l3 : V1a, --lat-lambda 3, only with LAT3=1)
# sigma / temperature per teacher from <sel dir>/selected.json (ESS pipeline on that teacher's cues, 11-24):
#   A′: spec 0.7565 / 0.6, spat 0.5722 / 0.7;  B: spec 0.7565 / 0.6, spat 2.1544 / 0.8.
# Trainer: code/_swap_pretrain_v1.py via TRAINER (swap_pretrain.sbatch; exported to the chained pieces), so
# swap_pretrain.py, which the running pilots' pieces re-read, stays untouched.
# Each run gets a gate -> scripts/evals/<label>.sh: shuffle diagnostic (check 4a), ridge dump + ridge (with A, C, Cs2 and
# the C v2 baseline if its dump exists), Pilot C's eval set (each encoder alone, concat, leakage both ways).
# Usage, from the SAC-SAR-SSL root: BASE=C bash scripts/launch_v1.sh [--hold]   (DRY=1: print the commands only)
set -euo pipefail
source scripts/space_guard.sh  # 11-25: refuse to submit with < 60 GB free on /scratch200 (MIN_FREE_GB, SPACE_GUARD_SKIP)
[ -n "${DRY:-}" ] || space_guard
BASE=${BASE:?set BASE=C or BASE=C2}
X=/scratch200/yotams2/SAC-SAR-SSL/exp
RIR_A=$X/rir_encoder/a1_design1; SEL_A=$X/v1b_sigma_tau_A1
RIR_B=$X/rir_encoder/oracle_B;   SEL_B=$X/v1b_sigma_tau_oracleB
for f in $RIR_A/emb_paired_v2.npy $RIR_B/emb_paired_v2.npy $SEL_A/selected.json $SEL_B/selected.json \
	$SEL_A/pca/pca_spec.npz $SEL_A/pca/pca_spat.npz $SEL_B/pca/pca_spec.npz code/_swap_pretrain_v1.py; do
	[ -f $f ] || { echo "missing $f"; exit 1; }
done
g() { python3 -c "import json; print(json.load(open('$1/selected.json'))['selected']['$2']['$3'])"; }
ts() {  # <sel dir> <lambda>: the V1b flags for one teacher
	local P="--tsac-pca-spec $1/pca/pca_spec.npz"; [ -f $1/pca/pca_spat.npz ] && P+=" --tsac-pca-spat $1/pca/pca_spat.npz"
	echo "--tsac-lambda-spec $2 --tsac-lambda-spat $2 $P --tsac-sigma-spec $(g $1 spec sigma) --tsac-sigma-spat $(g $1 spat sigma)" \
		"--tsac-temp-spec $(g $1 spec temp) --tsac-temp-spat $(g $1 spat temp)"
}
export PAIRED_TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar
export TRAINER=_swap_pretrain_v1.py
COMMON=(--snapshot-epochs --lr-schedule cosine --warmup-frac 0.05 --grad-clip-median-mult 3 --grad-share-every 500)
[ "$BASE" = C2 ] && COMMON+=(--ctr-lambda 1)
b=$(echo $BASE | tr C c)
declare -A RUNS=(
	[v1bA_l1]="--v1-rir-dir $RIR_A --tsac-spat-cue cosine $(ts $SEL_A 1)"
	[v1bA_l01]="--v1-rir-dir $RIR_A --tsac-spat-cue cosine $(ts $SEL_A 0.1)"
	[v1bB_l1]="--v1-rir-dir $RIR_B --tsac-spat-cue euclid $(ts $SEL_B 1)"
	[v1bB_l01]="--v1-rir-dir $RIR_B --tsac-spat-cue euclid $(ts $SEL_B 0.1)"
	[v1a_l1]="--v1-rir-dir $RIR_A --lat-lambda 1")
[ "${LAT3:-}" = 1 ] && RUNS[v1a_l3]="--v1-rir-dir $RIR_A --lat-lambda 3"
for r in "${!RUNS[@]}"; do
	run=pilot${b}_${r}; L=joint$(echo ${b}_${r} | tr -d _)
	ev=scripts/evals/$run.sh
	if [ -n "${DRY:-}" ]; then
		echo "$run (evals $L): TRAINER=$TRAINER sbatch ${1:-} scripts/swap_pretrain.sbatch $run ${COMMON[*]} ${RUNS[$r]}"; continue
	fi
	sed -e "s/pilotC_v2r/$run/g; s/jointCv2r/$L/g" -e "2s/.*/# $run (V1, launched by scripts\/launch_v1.sh, base $BASE): check 4a + ridge + Pilot C's eval set (labels ${L}_*)./" \
		scripts/evals/pilotC_v2r.sh > $ev
	python3 - "$ev" <<'P'
import sys; p = sys.argv[1]; s = open(p).read()
s = s.replace(' Cs2=$R/Cs2.npz ', ' Cs2=$R/Cs2.npz $extra ')
s = s.replace('rd=$(sbatch', 'extra=""; [ -f $R/jointCv2r.npz ] && extra="jointCv2r=$R/jointCv2r.npz"  # the C v2 baseline\nrd=$(sbatch', 1)
open(p, 'w').write(s)
P
	j=$(sbatch --parsable ${1:-} scripts/swap_pretrain.sbatch $run "${COMMON[@]}" ${RUNS[$r]})
	gt=$(sbatch --parsable --dependency=afterany:$j scripts/pretrain_gate.sbatch $X/joint_pretrain/$run $j $ev)
	echo "$run: piece $j, gate $gt, eval $ev"
done
