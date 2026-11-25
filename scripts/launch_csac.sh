#!/bin/bash
# C + SAC lambda control for the V1b claim (Yotam, 11-25): without it, a V1b gain could just be lambda (the old C + SAC
# ran at lambda 0.02). Same as the V1 runs except the cue source: paired v2, new recipe (cosine + warmup 0.05 +
# running-median clip x3), 10 epochs, seed 1, base C, gradient share every 500 steps, trainer _swap_pretrain_v1.py
# (the V1b code path; its --sac-* is the swap_pretrain.py C + SAC form).
# Cues = the hand-crafted SAC spatial cues exactly as C + SAC: 5 features of the spatial encoder's own input
# (spatial_features_from_stft), z-norm stats recomputed at the start, in-batch SAC on the SPATIAL encoder only
# (V1b puts its teacher-cue SAC on both encoders; there is no hand-crafted spectral cue in this trainer).
# sigma / temperature: SAC_SIGMA / SAC_TEMP, default 0.613 / 0.15 = the old C + SAC values (Interspeech selection).
#   pilotc_csac_l1  : --sac-lambda 1
#   pilotc_csac_l01 : --sac-lambda 0.1
# Each run gets a gate -> scripts/evals/<run>.sh (as V1: check 4a, ridge with A / C / Cs2 / C v2, Pilot C's eval set).
# Retention: --snapshot-epochs (empty) = latest + best + final only.
# Usage, from the SAC-SAR-SSL root: bash scripts/launch_csac.sh [--hold]   (DRY=1: print the commands only)
# Refuses to submit with < 60 GB free on /scratch200 (scripts/space_guard.sh).
set -euo pipefail
source scripts/space_guard.sh
[ -n "${DRY:-}" ] || space_guard
X=/scratch200/yotams2/SAC-SAR-SSL/exp
SIGMA=${SAC_SIGMA:-0.613}; TEMP=${SAC_TEMP:-0.15}
export PAIRED_TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar
export TRAINER=_swap_pretrain_v1.py
for f in $PAIRED_TAR code/$TRAINER scripts/evals/pilotC_v2r.sh; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
COMMON=(--snapshot-epochs --lr-schedule cosine --warmup-frac 0.05 --grad-clip-median-mult 3 --grad-share-every 500
	--sac-sigma $SIGMA --sac-temp $TEMP)
declare -A RUNS=([csac_l1]="--sac-lambda 1" [csac_l01]="--sac-lambda 0.1")
for r in "${!RUNS[@]}"; do
	run=pilotc_${r}; L=joint$(echo c_${r} | tr -d _)
	[ -e $X/joint_pretrain/$run ] && { echo "$run exists already: not relaunching"; exit 1; }
	ev=scripts/evals/$run.sh
	if [ -n "${DRY:-}" ]; then
		echo "$run (evals $L): TRAINER=$TRAINER sbatch ${1:-} scripts/swap_pretrain.sbatch $run ${COMMON[*]} ${RUNS[$r]}"; continue
	fi
	sed -e "s/pilotC_v2r/$run/g; s/jointCv2r/$L/g" -e "2s/.*/# $run (C + SAC lambda control, scripts\/launch_csac.sh): check 4a + ridge + Pilot C's eval set (labels ${L}_*)./" \
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
