#!/bin/bash
# V2 launcher (built 2026-11-24, base selectable 11-25): debiased-CKA independence penalty between the encoders
# (--cka-lambda, agreed 11-23), weights 0.1 / 1, on top of one base run, with nothing else changed:
#   BASE=C     pilotC_v2r (C 10 ep, paired v2, cosine + warmup 0.05 + running-median clip x3, anchors on); trainer
#              swap_pretrain.py (installed, has --cka-lambda) -> runs pilotc_v2cka_l01 / _l1, labels jointcv2ckal01 / l1
#   BASE=V1bA  pilotc_v1bA_l01 (V1b A′ λ0.1: + A′ spectral / spatial teacher SAC terms); trainer _swap_pretrain_v1.py
#              -> runs pilotc_v1bAcka_l01 / _l1, labels jointcv1bAckal01 / l1
# Retention: --snapshot-epochs (none; does not change training). V100 (compute-0-300) excluded, incl. chained pieces.
# Before submitting, each command is parsed with the trainer's own argparse and compared with the base's args.json:
# any difference other than cka_lambda (+ exp_dir / num_workers / snapshot_epochs), or any trainer option missing from
# that args.json that is not at its default, stops the launch. Each run gets a gate -> scripts/evals/<run>.sh
# (from pilotc_v1bA_l1.sh: check 4a + ridge incl. the C v2 baseline + Pilot C's eval set).
# Usage, from the SAC-SAR-SSL root: BASE=C|V1bA bash scripts/launch_v2.sh [--hold]   (DRY=1: check + print only)
set -euo pipefail
source scripts/space_guard.sh  # 11-25: refuse to submit with < 60 GB free on /scratch200 (MIN_FREE_GB, SPACE_GUARD_SKIP)
[ -n "${DRY:-}" ] || space_guard
BASE=${BASE:?set BASE=C or BASE=V1bA}
X=/scratch200/yotams2/SAC-SAR-SSL/exp
export PAIRED_TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar
[ -f $PAIRED_TAR ] || { echo "missing $PAIRED_TAR"; exit 1; }
export EXCLUDE='compute-0-[18,53,291,300,376]'  # swap_pretrain.sbatch defaults + 300 (V100); forwarded to chained pieces
COMMON=(--lr-schedule cosine --warmup-frac 0.05 --grad-clip-median-mult 3 --snapshot-epochs)
case $BASE in
C)	REF=$X/joint_pretrain/pilotC_v2r; export TRAINER=swap_pretrain.py; stem=pilotc_v2cka; lstem=jointcv2cka
	BASEARGS=() ;;
V1bA)	REF=$X/joint_pretrain/pilotc_v1bA_l01; export TRAINER=_swap_pretrain_v1.py; stem=pilotc_v1bAcka; lstem=jointcv1bAcka
	A=$X/v1b_sigma_tau_A1
	BASEARGS=(--grad-share-every 500 --v1-rir-dir $X/rir_encoder/a1_design1 --tsac-spat-cue cosine
		--tsac-lambda-spec 0.1 --tsac-lambda-spat 0.1 --tsac-pca-spec $A/pca/pca_spec.npz --tsac-pca-spat $A/pca/pca_spat.npz
		--tsac-sigma-spec 0.7564633275546291 --tsac-sigma-spat 0.5722367659350217 --tsac-temp-spec 0.6 --tsac-temp-spat 0.7) ;;
*)	echo "BASE must be C or V1bA (got $BASE)"; exit 1 ;;
esac
for lam in 0.1 1; do
	tag=l$(echo $lam | tr -d .); run=${stem}_$tag; L=$lstem$tag
	ARGS=("${COMMON[@]}" "${BASEARGS[@]}" --cka-lambda $lam)
	python3 - "$REF/args.json" code/$TRAINER "$PAIRED_TAR" "$lam" "${ARGS[@]}" <<'P'
import argparse, json, os, sys, textwrap
ref, src, tar, lam, argv = json.load(open(sys.argv[1])), open(sys.argv[2]).read(), sys.argv[3], float(sys.argv[4]), sys.argv[5:]
i = src.rindex('\n', 0, src.index('ap = argparse.ArgumentParser()')) + 1
block = textwrap.dedent(src[i:src.index('args = ap.parse_args()')].expandtabs(4))
ns = {'argparse': argparse}; exec(block, ns)
ap = ns['ap']
new = vars(ap.parse_args(['--tar', tar, '--exp-dir', 'x'] + argv))
dflt = vars(ap.parse_args(['--tar', 't', '--exp-dir', 'x']))
skip = {'exp_dir', 'num_workers', 'snapshot_epochs', 'cka_lambda'}
diff = {k: (ref[k], new.get(k)) for k in ref if k not in skip and ref[k] != new.get(k)}
extra = {k: (dflt[k], new[k]) for k in new if k not in ref and new[k] != dflt[k]}
base = os.path.basename(os.path.dirname(sys.argv[1]))
print(f"[check] vs {base}: cka_lambda {ref.get('cka_lambda')} -> {new['cka_lambda']}; retention only: snapshot_epochs {ref.get('snapshot_epochs')} -> {new['snapshot_epochs']}")
print(f"[check] other differences: {diff or 'none'}; trainer options not in the reference and not at default: {extra or 'none'}")
sys.exit(0 if not diff and not extra and new['cka_lambda'] == lam else 1)
P
	if [ -n "${DRY:-}" ]; then
		echo "$run (evals $L): TRAINER=$TRAINER EXCLUDE=$EXCLUDE sbatch ${1:-} --exclude=$EXCLUDE scripts/swap_pretrain.sbatch $run ${ARGS[*]}"; continue
	fi
	[ -e $X/joint_pretrain/$run ] && { echo "$run exists already: not relaunching"; exit 1; }
	ev=scripts/evals/$run.sh
	sed -e "s/pilotc_v1bA_l1/$run/g; s/jointcv1bAl1/$L/g" -e "2s/.*/# $run (V2, scripts\/launch_v2.sh, base $BASE, CKA $lam): check 4a + ridge + Pilot C's eval set (labels ${L}_*)./" \
		scripts/evals/pilotc_v1bA_l1.sh > $ev
	j=$(sbatch --parsable ${1:-} --exclude=$EXCLUDE scripts/swap_pretrain.sbatch $run "${ARGS[@]}")
	gt=$(sbatch --parsable --dependency=afterany:$j scripts/pretrain_gate.sbatch $X/joint_pretrain/$run $j $ev)
	echo "$run: piece $j, gate $gt, eval $ev"
done
