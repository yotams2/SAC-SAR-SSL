#!/bin/bash
# C + SAC (agreed 2026-11-22; base C or C2, see args.json ctr_lambda): Pilot C's eval set on the final encoders, plus
# check 4a (swap_shufdiag.py: bin-centred shuffle diagnostic with valid donors) on joint_best and joint_final.
set -euo pipefail
J=${JOINT_EXP:-/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain}
D=$J/pilotCsac
sbatch --parsable scripts/swap_shufdiag.sbatch /scratch200/yotams2/SAC-SAR-SSL/exp/shufdiag/pilotCsac.json \
	Csac_best=$D:joint_best.pth Csac_final=$D:joint_final.pth
PC_RUN=pilotCsac PC_LABEL=jointCsac exec bash scripts/evals/pilotC.sh
