#!/bin/bash
# V3 launcher (Yotam 11-25): unpaired swap. C v2 10 ep's recipe (pilotC_v2r: cosine + warmup 0.05 + running-median clip x3,
# 10 ep, anchors ON = own-scale 0.1, seed 1) trained on the unpaired 8.224-s set with --v3 (v3_data.py: target = h1 as
# recorded, spectral input = EQ(WPE(h1) * RIR from another paired-v2 training scene), spatial input = h2 of the same clip).
# Primary readout: all five tasks vs C v2 10 ep ("does the swap work without paired simulation?"); spectral leakage secondary.
# Trainer = _swap_pretrain_v1.py (has --v3; V1 / ctr terms off). Retention: --snapshot-epochs (none; does not change training).
# V100 (compute-0-300) excluded, incl. chained pieces. Gate -> scripts/evals/pilotc_v3.sh: check 4a on C v2's paired-v2
# validation split (SHUFDIAG_TAR; PairedDataset cannot read the unpaired tar), ridge incl. the C v2 baseline, Pilot C's eval set.
# Before submitting, the command is parsed with the trainer's own argparse and compared with pilotC_v2r/args.json: any
# difference other than v3 / tar / val_scenes (+ exp_dir / num_workers / snapshot_epochs), or any trainer option missing from
# that args.json that is not at its default, stops the launch.
# Usage, from the SAC-SAR-SSL root: bash scripts/launch_v3.sh [--hold]   (DRY=1: check + print only)
set -euo pipefail
source scripts/space_guard.sh
[ -n "${DRY:-}" ] || space_guard
X=/scratch200/yotams2/SAC-SAR-SSL/exp; REF=$X/joint_pretrain/pilotC_v2r
M=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig
export PAIRED_TAR=$M/simu_unpaired8_pretrain.tar  # swap_pretrain.sbatch passes it as --tar
export TRAINER=_swap_pretrain_v1.py
export EXCLUDE='compute-0-[18,53,291,300,376]'  # swap_pretrain.sbatch defaults + 300 (V100); forwarded to chained pieces
for f in $PAIRED_TAR $PAIRED_TAR.meta.npz $PAIRED_TAR.index.json $M/simu_unpaired8_wpe $M/simu_paired_pretrain_v2.tar.rirs.npz; do
	[ -e $f ] || { echo "missing $f"; exit 1; }
done
ARGS=(--lr-schedule cosine --warmup-frac 0.05 --grad-clip-median-mult 3 --snapshot-epochs --v3 --val-scenes 2500)
run=pilotc_v3; L=jointcv3
python3 - "$REF/args.json" code/$TRAINER "$PAIRED_TAR" "${ARGS[@]}" <<'P'
import argparse, json, sys, textwrap
ref, src, tar, argv = json.load(open(sys.argv[1])), open(sys.argv[2]).read(), sys.argv[3], sys.argv[4:]
i = src.rindex('\n', 0, src.index('ap = argparse.ArgumentParser()')) + 1
block = textwrap.dedent(src[i:src.index('args = ap.parse_args()')].expandtabs(4))
ns = {'argparse': argparse}; exec(block, ns)
ap = ns['ap']
new = vars(ap.parse_args(['--tar', tar, '--exp-dir', 'x'] + argv))
dflt = vars(ap.parse_args(['--tar', 't', '--exp-dir', 'x']))
expected = {'v3', 'tar', 'val_scenes'}
skip = {'exp_dir', 'num_workers', 'snapshot_epochs'}
diff = {k: (ref[k], new.get(k)) for k in ref if k not in skip | expected and ref[k] != new.get(k)}
extra = {k: (dflt[k], new[k]) for k in new if k not in ref and k not in expected and new[k] != dflt[k]}
print('[check] vs pilotC_v2r: ' + ', '.join(f"{k} {ref.get(k)!r} -> {new[k]!r}" for k in sorted(expected)))
print(f"[check] retention only: snapshot_epochs {ref.get('snapshot_epochs')} -> {new['snapshot_epochs']}")
print(f"[check] trainer options not in the reference, all at default: {sorted(k for k in new if k not in ref and k not in expected)}")
print(f"[check] other differences: {diff or 'none'}; non-default new options: {extra or 'none'}")
ok = not diff and not extra and new['v3'] and new['val_scenes'] == 2500 and new['own_scale'] == 0.1 and new['nepoch'] == 10
sys.exit(0 if ok else 1)
P
if [ -n "${DRY:-}" ]; then
	echo "$run (evals $L): PAIRED_TAR=$PAIRED_TAR TRAINER=$TRAINER EXCLUDE=$EXCLUDE sbatch ${1:-} --exclude=$EXCLUDE scripts/swap_pretrain.sbatch $run ${ARGS[*]}"; exit 0
fi
[ -e $X/joint_pretrain/$run ] && { echo "$run exists already: not relaunching"; exit 1; }
ev=scripts/evals/$run.sh
sed -e "s/pilotc_v1bA_l1/$run/g; s/jointcv1bAl1/$L/g" -e "2s/.*/# $run (V3 unpaired swap, scripts\/launch_v3.sh): check 4a (on C v2's paired-v2 validation split) + ridge + Pilot C's eval set (labels ${L}_*)./" \
	-e "s|^sbatch --parsable scripts/swap_shufdiag.sbatch|SHUFDIAG_TAR=$M/simu_paired_pretrain_v2.tar SHUFDIAG_VAL_SCENES=500 sbatch --parsable scripts/swap_shufdiag.sbatch|" \
	scripts/evals/pilotc_v1bA_l1.sh > $ev
grep -q '^SHUFDIAG_TAR=' $ev || { echo "eval script: check 4a override not applied"; rm -f $ev; exit 1; }
j=$(sbatch --parsable ${1:-} --exclude=$EXCLUDE scripts/swap_pretrain.sbatch $run "${ARGS[@]}")
gt=$(sbatch --parsable --dependency=afterany:$j scripts/pretrain_gate.sbatch $X/joint_pretrain/$run $j $ev)
echo "$run: piece $j, gate $gt, eval $ev"
