#!/bin/bash
# Final 30-ep run (Yotam 11-26 evening): V1b A′ λ0.1 (pilotc_v1bA_l01) config at 30 ep, new recipe, paired v2, seeds 1 + 2.
# Nothing else changed: same A′ spectral + spatial teacher cues (PCA spec / spat), sigma / temp, lambda 0.1, recipe.
#   ANCHORS=on   own_scale 0.1 (as pilotc_v1bA_l01)  -> runs pilotc_v1bA30_l01_s<seed>,      labels jointcv1bA30l01s<seed>
#   ANCHORS=off  own_scale 0   (as V1b-A′-noanc)     -> runs pilotc_v1bA30noanc_l01_s<seed>, labels jointcv1bA30noancl01s<seed>
# ANCHORS rule (Yotam 11-26): off if V1b-A′-noanc is no worse than V1b A′ λ0.1 by > ~5% on any of TDOA / T60 / DRR and no
# worse on SC2 / SID; otherwise on. If noanc's evals are not in by Fri 27 Nov 09:00, launch with ANCHORS=on.
# Baseline row = C 30 ep (pilotC30_v2r): its ridge dump goes into each run's ridge probe (not the C v2 10-ep one).
# Retention: --snapshot-epochs empty (latest + best + final; does not change training). V100 (compute-0-300) excluded,
# incl. chained pieces (~11 h per run on L40S at ~3.5 s/step -> 1-2 pieces of 12 h, auto-chained by the gate).
# Before submitting, each command is parsed with the trainer's own argparse and compared with pilotc_v1bA_l01/args.json:
# allowed differences are nepoch (10 -> 30), seed, own_scale (only with ANCHORS=off), + exp_dir / tar / num_workers /
# snapshot_epochs; anything else stops the launch. Each run gets a gate -> scripts/evals/<run>.sh (check 4a + ridge +
# Pilot C's eval set, from pilotc_v1bA_l1.sh).
# Usage, from the SAC-SAR-SSL root: ANCHORS=on|off [SEEDS="1 2"] bash scripts/launch_v1bA30.sh [--hold]  (DRY=1: check + print)
set -euo pipefail
source scripts/space_guard.sh  # refuse to submit with < 60 GB free on /scratch200
[ -n "${DRY:-}" ] || space_guard
ANCHORS=${ANCHORS:?set ANCHORS=on or ANCHORS=off}
SEEDS=${SEEDS:-1 2}
X=/scratch200/yotams2/SAC-SAR-SSL/exp; A=$X/v1b_sigma_tau_A1; REF=$X/joint_pretrain/pilotc_v1bA_l01; R=$X/ridge
export PAIRED_TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar
export TRAINER=_swap_pretrain_v1.py
export EXCLUDE='compute-0-[18,53,291,300,376]'  # swap_pretrain.sbatch defaults + 300 (V100); forwarded to chained pieces
for f in $PAIRED_TAR $REF/args.json $R/jointC30v2r.npz $A/pca/pca_spec.npz $A/pca/pca_spat.npz $X/rir_encoder/a1_design1; do
	[ -e $f ] || { echo "missing $f"; exit 1; }
done
case $ANCHORS in
on)	OWN=0.1; stem=pilotc_v1bA30_l01; lstem=jointcv1bA30l01 ;;
off)	OWN=0; stem=pilotc_v1bA30noanc_l01; lstem=jointcv1bA30noancl01 ;;
*)	echo "ANCHORS must be on or off (got $ANCHORS)"; exit 1 ;;
esac
for seed in $SEEDS; do
	run=${stem}_s$seed; L=${lstem}s$seed
	ARGS=(--lr-schedule cosine --warmup-frac 0.05 --grad-clip-median-mult 3 --grad-share-every 500
		--v1-rir-dir $X/rir_encoder/a1_design1 --tsac-spat-cue cosine --tsac-lambda-spec 0.1 --tsac-lambda-spat 0.1
		--tsac-pca-spec $A/pca/pca_spec.npz --tsac-pca-spat $A/pca/pca_spat.npz
		--tsac-sigma-spec 0.7564633275546291 --tsac-sigma-spat 0.5722367659350217 --tsac-temp-spec 0.6 --tsac-temp-spat 0.7
		--nepoch 30 --seed $seed --own-scale $OWN --snapshot-epochs)
	python3 - "$REF/args.json" code/$TRAINER "$PAIRED_TAR" "$seed" "$OWN" "${ARGS[@]}" <<'P'
import argparse, json, sys, textwrap
ref, src, tar, seed, own, argv = (json.load(open(sys.argv[1])), open(sys.argv[2]).read(), sys.argv[3],
                                  int(sys.argv[4]), float(sys.argv[5]), sys.argv[6:])
i = src.rindex('\n', 0, src.index('ap = argparse.ArgumentParser()')) + 1
block = textwrap.dedent(src[i:src.index('args = ap.parse_args()')].expandtabs(4))
ns = {'argparse': argparse}; exec(block, ns)
new = vars(ns['ap'].parse_args(['--tar', tar, '--exp-dir', 'x'] + argv))
allowed = {'nepoch', 'seed', 'own_scale', 'snapshot_epochs', 'exp_dir', 'tar', 'num_workers'}
diff = {k: (ref.get(k), new.get(k)) for k in set(ref) | set(new) if k not in allowed and ref.get(k) != new.get(k)}
print(f"[check] vs pilotc_v1bA_l01: nepoch {ref['nepoch']} -> {new['nepoch']}, seed {ref['seed']} -> {new['seed']}, "
      f"own_scale {ref['own_scale']} -> {new['own_scale']}; retention only: snapshot_epochs {ref.get('snapshot_epochs')} -> "
      f"{new['snapshot_epochs']}; other differences: {diff or 'none'}")
ok = not diff and new['nepoch'] == 30 and new['seed'] == seed and new['own_scale'] == own and new['tar'] == ref['tar']
sys.exit(0 if ok else 1)
P
	if [ -n "${DRY:-}" ]; then
		echo "$run (evals $L): TRAINER=$TRAINER EXCLUDE=$EXCLUDE sbatch ${1:-} --exclude=$EXCLUDE scripts/swap_pretrain.sbatch $run ${ARGS[*]}"; continue
	fi
	[ -e $X/joint_pretrain/$run ] && { echo "$run exists already: not relaunching"; exit 1; }
	ev=scripts/evals/$run.sh
	sed -e "s/pilotc_v1bA_l1/$run/g; s/jointcv1bAl1/$L/g" \
		-e "2s/.*/# $run (final 30 ep, scripts\/launch_v1bA30.sh, ANCHORS=$ANCHORS, seed $seed): check 4a + ridge + Pilot C's eval set (labels ${L}_*)./" \
		-e 's/extra=""; \[ -f $R\/jointCv2r.npz \] && extra="jointCv2r=$R\/jointCv2r.npz"  # the C v2 baseline/extra="jointC30v2r=$R\/jointC30v2r.npz"  # baseline = C 30 ep/' \
		scripts/evals/pilotc_v1bA_l1.sh > $ev
	grep -q 'extra="jointC30v2r=' $ev || { echo "eval template changed: C 30 ep baseline not wired into $ev"; rm -f $ev; exit 1; }
	j=$(sbatch --parsable ${1:-} --exclude=$EXCLUDE scripts/swap_pretrain.sbatch $run "${ARGS[@]}")
	gt=$(sbatch --parsable --dependency=afterany:$j scripts/pretrain_gate.sbatch $X/joint_pretrain/$run $j $ev)
	echo "$run: piece $j, gate $gt, eval $ev"
done
