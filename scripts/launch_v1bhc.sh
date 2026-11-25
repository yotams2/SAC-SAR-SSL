#!/bin/bash
# V1b-HC: the MATCHED control for V1b A′ (Yotam 11-25; replaces the C + SAC control, launch_csac.sh). Identical to V1b A′
# (scripts/launch_v1.sh v1bA_*) except the spatial cue source:
#   spectral SAC: V1b A′'s, exactly (dry-speech content-teacher cues, PCA pca_spec.npz of v1b_sigma_tau_A1, sigma / temp
#                 from v1b_sigma_tau_A1/selected.json);
#   spatial SAC:  the hand-crafted C + SAC cues (5 features, z-normed, scene mean of the 4 renderings; code/v1b_hc_cues.py)
#                 through --tsac-cue-file-spat, sigma = the V1b ESS step on those cues, temp = A′'s spatial temp
#                 (v1b_sigma_tau_HC/selected.json, scripts/v1b_hc_select.sbatch).
# lambda 1 / 0.1 on both encoders, paired v2, new recipe, 10 ep, seed 1; evals as V1 (check 4a, ridge, Pilot C's eval set).
# Retention: --snapshot-epochs (none) = latest + best + final (snapshots do not change training).
# Usage, from the SAC-SAR-SSL root: bash scripts/launch_v1bhc.sh [--hold]   (DRY=1: print the commands only)
set -euo pipefail
source scripts/space_guard.sh  # refuses with < 60 GB free on /scratch200
[ -n "${DRY:-}" ] || space_guard
X=/scratch200/yotams2/SAC-SAR-SSL/exp
RIR_A=$X/rir_encoder/a1_design1; SEL_A=$X/v1b_sigma_tau_A1; SEL=$X/v1b_sigma_tau_HC; CUES=$SEL/hc_cues.npy
for f in $RIR_A/emb_paired_v2.npy $SEL_A/pca/pca_spec.npz $SEL/selected.json $CUES code/_swap_pretrain_v1.py; do
	[ -f $f ] || { echo "missing $f (run scripts/v1b_hc_select.sbatch first)"; exit 1; }
done
g() { python3 -c "import json; print(json.load(open('$1/selected.json'))['selected']['$2']['$3'])"; }
[ "$(g $SEL spec sigma)" = "$(g $SEL_A spec sigma)" ] && [ "$(g $SEL spec temp)" = "$(g $SEL_A spec temp)" ] && \
	[ "$(g $SEL spat temp)" = "$(g $SEL_A spat temp)" ] || { echo "HC selection does not match A′ on spec sigma/temp or spat temp"; exit 1; }
export PAIRED_TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar
export TRAINER=_swap_pretrain_v1.py
COMMON=(--snapshot-epochs --lr-schedule cosine --warmup-frac 0.05 --grad-clip-median-mult 3 --grad-share-every 500)
hc() {  # <lambda>
	echo "--v1-rir-dir $RIR_A --tsac-lambda-spec $1 --tsac-lambda-spat $1 --tsac-pca-spec $SEL_A/pca/pca_spec.npz" \
		"--tsac-cue-file-spat $CUES --tsac-sigma-spec $(g $SEL spec sigma) --tsac-sigma-spat $(g $SEL spat sigma)" \
		"--tsac-temp-spec $(g $SEL spec temp) --tsac-temp-spat $(g $SEL spat temp)"
}
declare -A RUNS=([v1bHC_l1]="$(hc 1)" [v1bHC_l01]="$(hc 0.1)")
for r in v1bHC_l1 v1bHC_l01; do
	run=pilotc_${r}; L=jointc$(echo ${r} | tr -d _)
	[ -e $X/joint_pretrain/$run ] && { echo "$run exists already: not relaunching"; exit 1; }
	ev=scripts/evals/$run.sh
	if [ -n "${DRY:-}" ]; then
		echo "$run (evals $L): TRAINER=$TRAINER sbatch ${1:-} scripts/swap_pretrain.sbatch $run ${COMMON[*]} ${RUNS[$r]}"; continue
	fi
	sed -e "s/pilotc_v1bA_l1/$run/g; s/jointcv1bAl1/$L/g" -e "2s/.*/# $run (V1b-HC matched control, scripts\/launch_v1bhc.sh): check 4a + ridge + Pilot C's eval set (labels ${L}_*)./" \
		scripts/evals/pilotc_v1bA_l1.sh > $ev
	j=$(sbatch --parsable ${1:-} scripts/swap_pretrain.sbatch $run "${COMMON[@]}" ${RUNS[$r]})
	gt=$(sbatch --parsable --dependency=afterany:$j scripts/pretrain_gate.sbatch $X/joint_pretrain/$run $j $ev)
	echo "$run: piece $j, gate $gt, eval $ev"
done
