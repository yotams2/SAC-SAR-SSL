#!/bin/bash
# V1b-U-noanc (Yotam's candidate configuration, prepared 11-25; launch only on Yotam's go, with his lambda):
#   V1b trainer (_swap_pretrain_v1.py), --own-scale 0 (no MPG / MPC anchors: swap + V1b terms only);
#   spectral SAC = V1b A′'s exactly (dry-speech teacher cues, PCA pca_spec.npz of v1b_sigma_tau_A1, sigma 0.7565 / temp 0.6);
#   spatial SAC  = A′-uniform's embedding as the teacher cue (exp/rir_encoder/a1u_uniform; centre -> PCA by the V1b rule
#                  -> L2, cosine cue), sigma from the ESS step, temp = A′'s 0.7 (exp/v1b_sigma_tau_U, scripts/v1bU_select.sbatch);
#   lambda = LAMBDA (both encoders), paired v2, new recipe, 10 ep, seed 1; gate + evals + check 4a as V1.
# GPUs: V100 (compute-0-300) excluded for this run and its chained pieces (~2x slower: V1b-HC ~20 s/step there vs ~9-13 on
# L40S), plus the default exclusions of swap_pretrain.sbatch.
# Usage, from the SAC-SAR-SSL root: LAMBDA=0.1|1 bash scripts/launch_v1bU.sh [--hold]   (DRY=1: print the command only)
set -euo pipefail
source scripts/space_guard.sh
[ -n "${DRY:-}" ] || space_guard
X=/scratch200/yotams2/SAC-SAR-SSL/exp
RIR_U=$X/rir_encoder/a1u_uniform; SEL_A=$X/v1b_sigma_tau_A1; SEL=$X/v1b_sigma_tau_U
LAM=${LAMBDA:-}
if [ -z "$LAM" ]; then
	[ -n "${DRY:-}" ] || { echo "set LAMBDA=0.1 or LAMBDA=1 (Yotam's pick after V1b A′ λ1's evals)"; exit 1; }
	LAM='<λ>'
fi
[ -n "${DRY:-}" ] || [ "$LAM" = 0.1 ] || [ "$LAM" = 1 ] || { echo "LAMBDA must be 0.1 or 1"; exit 1; }
missing=0
for f in $RIR_U/emb_paired_v2.npy $RIR_U/emb_stats.npz $SEL_A/pca/pca_spec.npz $SEL/pca/pca_spat.npz $SEL/selected.json code/_swap_pretrain_v1.py; do
	[ -f $f ] || { echo "missing $f (A′-uniform run + scripts/v1bU_select.sbatch first)"; missing=1; }
done
[ $missing = 0 ] || [ -n "${DRY:-}" ] || exit 1
g() { [ -f $1/selected.json ] && python3 -c "import json; print(json.load(open('$1/selected.json'))['selected']['$2']['$3'])" || echo "<$2 $3>"; }
if [ -f $SEL/selected.json ]; then
	[ "$(g $SEL spec sigma)" = "$(g $SEL_A spec sigma)" ] && [ "$(g $SEL spec temp)" = "$(g $SEL_A spec temp)" ] && \
		[ "$(g $SEL spat temp)" = "$(g $SEL_A spat temp)" ] || { echo "U selection does not match A′ on spec sigma/temp or spat temp"; exit 1; }
fi
export PAIRED_TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar
export TRAINER=_swap_pretrain_v1.py
export EXCLUDE='compute-0-[18,53,291,300,376]'  # swap_pretrain.sbatch defaults + 300 (V100); forwarded to chained pieces
ARGS=(--snapshot-epochs --lr-schedule cosine --warmup-frac 0.05 --grad-clip-median-mult 3 --grad-share-every 500 --own-scale 0
	--v1-rir-dir $RIR_U --tsac-spat-cue cosine --tsac-lambda-spec $LAM --tsac-lambda-spat $LAM
	--tsac-pca-spec $SEL_A/pca/pca_spec.npz --tsac-pca-spat $SEL/pca/pca_spat.npz
	--tsac-sigma-spec $(g $SEL spec sigma) --tsac-sigma-spat $(g $SEL spat sigma) --tsac-temp-spec $(g $SEL spec temp) --tsac-temp-spat $(g $SEL spat temp))
tag=$(echo "$LAM" | tr -d '.'); run=pilotc_v1bU_noanc_l${tag}; L=jointcv1bUnoancl${tag}
if [ -n "${DRY:-}" ]; then
	echo "$run (evals $L): TRAINER=$TRAINER EXCLUDE=$EXCLUDE sbatch ${1:-} --exclude=$EXCLUDE scripts/swap_pretrain.sbatch $run ${ARGS[*]}"; exit 0
fi
[ -e $X/joint_pretrain/$run ] && { echo "$run exists already: not relaunching"; exit 1; }
ev=scripts/evals/$run.sh
sed -e "s/pilotc_v1bA_l1/$run/g; s/jointcv1bAl1/$L/g" -e "2s/.*/# $run (V1b-U-noanc, scripts\/launch_v1bU.sh): check 4a + ridge + Pilot C's eval set (labels ${L}_*)./" \
	scripts/evals/pilotc_v1bA_l1.sh > $ev
j=$(sbatch --parsable ${1:-} --exclude=$EXCLUDE scripts/swap_pretrain.sbatch $run "${ARGS[@]}")
gt=$(sbatch --parsable --dependency=afterany:$j scripts/pretrain_gate.sbatch $X/joint_pretrain/$run $j $ev)
echo "$run: piece $j, gate $gt, eval $ev"
