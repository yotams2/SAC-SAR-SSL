#!/bin/bash
# Pilot C2 seed 2 (agreed 2026-11-22): C2 seed 1's config, --seed 2. Pilot C's eval set + check 4a on joint_final.
set -euo pipefail
J=${JOINT_EXP:-/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain}
D=$J/pilotC2_ctr_s2
sbatch --parsable scripts/swap_shufdiag.sbatch /scratch200/yotams2/SAC-SAR-SSL/exp/shufdiag/pilotC2_s2.json \
	C2s2_final=$D:joint_final.pth
PC_RUN=pilotC2_ctr_s2 PC_LABEL=jointC2s2 exec bash scripts/evals/pilotC.sh
