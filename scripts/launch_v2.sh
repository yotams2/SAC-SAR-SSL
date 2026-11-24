#!/bin/bash
# V2 launcher (built 2026-11-24): debiased-CKA independence penalty between the encoders (--cka-lambda, agreed 11-23),
# weights 0.1 / 1. Same as the C v2 baseline pilotC_v2r (paired v2, 10 ep, cosine + warmup 0.05 + running-median clip x3)
# + --cka-lambda; BASE=C2 adds --ctr-lambda 1. Trainer = installed swap_pretrain.py (has --cka-lambda since 11-23).
# Each run gets a gate -> scripts/evals/<run>.sh (copy of pilotC_v2r.sh: check 4a + ridge incl. the C v2 baseline + Pilot C's eval set).
# Usage, from the SAC-SAR-SSL root: BASE=C bash scripts/launch_v2.sh [--hold]   (DRY=1: print the commands only)
set -euo pipefail
BASE=${BASE:?set BASE=C or BASE=C2}
X=/scratch200/yotams2/SAC-SAR-SSL/exp
export PAIRED_TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar
[ -f $PAIRED_TAR ] || { echo "missing $PAIRED_TAR"; exit 1; }
COMMON=(--lr-schedule cosine --warmup-frac 0.05 --grad-clip-median-mult 3)
[ "$BASE" = C2 ] && COMMON+=(--ctr-lambda 1)
b=$(echo $BASE | tr C c)
declare -A RUNS=([v2cka_l01]="--cka-lambda 0.1" [v2cka_l1]="--cka-lambda 1")
for r in "${!RUNS[@]}"; do
	run=pilot${b}_${r}; L=joint$(echo ${b}_${r} | tr -d _)
	ev=scripts/evals/$run.sh
	if [ -n "${DRY:-}" ]; then
		echo "$run (evals $L): sbatch ${1:-} scripts/swap_pretrain.sbatch $run ${COMMON[*]} ${RUNS[$r]}"; continue
	fi
	sed -e "s/pilotC_v2r/$run/g; s/jointCv2r/$L/g" -e "2s/.*/# $run (V2, launched by scripts\/launch_v2.sh, base $BASE): check 4a + ridge + Pilot C's eval set (labels ${L}_*)./" \
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
