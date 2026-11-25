#!/bin/bash
# V1b-A′-noanc (Yotam 11-25): V1b A′ λ0.1 (pilotc_v1bA_l01) with --own-scale 0 (MPG / MPC anchors off), nothing else changed:
# same spectral + spatial cues (A′ teacher, PCA spec / spat), sigma / temp, lambda 0.1, recipe, 10 ep, paired v2, seed 1.
# Retention: --snapshot-epochs (none; does not change training). V100 (compute-0-300) excluded, incl. chained pieces.
# Before submitting, the command is parsed with the trainer's own argparse and compared with pilotc_v1bA_l01/args.json:
# any difference other than own_scale / snapshot_epochs / exp_dir / tar / num_workers stops the launch.
# Usage, from the SAC-SAR-SSL root: bash scripts/launch_v1bA_noanc.sh [--hold]   (DRY=1: check + print only)
set -euo pipefail
source scripts/space_guard.sh
[ -n "${DRY:-}" ] || space_guard
X=/scratch200/yotams2/SAC-SAR-SSL/exp; A=$X/v1b_sigma_tau_A1; REF=$X/joint_pretrain/pilotc_v1bA_l01
export PAIRED_TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar
export TRAINER=_swap_pretrain_v1.py
export EXCLUDE='compute-0-[18,53,291,300,376]'  # swap_pretrain.sbatch defaults + 300 (V100); forwarded to chained pieces
ARGS=(--lr-schedule cosine --warmup-frac 0.05 --grad-clip-median-mult 3 --grad-share-every 500
	--v1-rir-dir $X/rir_encoder/a1_design1 --tsac-spat-cue cosine --tsac-lambda-spec 0.1 --tsac-lambda-spat 0.1
	--tsac-pca-spec $A/pca/pca_spec.npz --tsac-pca-spat $A/pca/pca_spat.npz
	--tsac-sigma-spec 0.7564633275546291 --tsac-sigma-spat 0.5722367659350217 --tsac-temp-spec 0.6 --tsac-temp-spat 0.7
	--snapshot-epochs --own-scale 0)
run=pilotc_v1bA_noanc_l01; L=jointcv1bAnoancl01
python3 - "$REF/args.json" code/$TRAINER "$PAIRED_TAR" "${ARGS[@]}" <<'P'
import argparse, json, sys, textwrap
ref, src, tar, argv = json.load(open(sys.argv[1])), open(sys.argv[2]).read(), sys.argv[3], sys.argv[4:]
i = src.rindex('\n', 0, src.index('ap = argparse.ArgumentParser()')) + 1
block = textwrap.dedent(src[i:src.index('args = ap.parse_args()')].expandtabs(4))
ns = {'argparse': argparse}; exec(block, ns)
new = vars(ns['ap'].parse_args(['--tar', tar, '--exp-dir', 'x'] + argv))
skip = {'own_scale', 'snapshot_epochs', 'exp_dir', 'tar', 'num_workers'}
diff = {k: (ref.get(k), new.get(k)) for k in set(ref) | set(new) if k not in skip and ref.get(k) != new.get(k)}
print(f"[check] vs pilotc_v1bA_l01: own_scale {ref['own_scale']} -> {new['own_scale']}; other differences: {diff or 'none'}")
sys.exit(1 if diff or new['own_scale'] != 0 else 0)
P
if [ -n "${DRY:-}" ]; then
	echo "$run (evals $L): TRAINER=$TRAINER EXCLUDE=$EXCLUDE sbatch ${1:-} --exclude=$EXCLUDE scripts/swap_pretrain.sbatch $run ${ARGS[*]}"; exit 0
fi
[ -e $X/joint_pretrain/$run ] && { echo "$run exists already: not relaunching"; exit 1; }
ev=scripts/evals/$run.sh
sed -e "s/pilotc_v1bA_l1/$run/g; s/jointcv1bAl1/$L/g" -e "2s/.*/# $run (V1b-A′-noanc, scripts\/launch_v1bA_noanc.sh): check 4a + ridge + Pilot C's eval set (labels ${L}_*)./" \
	scripts/evals/pilotc_v1bA_l1.sh > $ev
j=$(sbatch --parsable ${1:-} --exclude=$EXCLUDE scripts/swap_pretrain.sbatch $run "${ARGS[@]}")
gt=$(sbatch --parsable --dependency=afterany:$j scripts/pretrain_gate.sbatch $X/joint_pretrain/$run $j $ev)
echo "$run: piece $j, gate $gt, eval $ev"
