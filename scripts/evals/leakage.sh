#!/bin/bash
# Leakage probes (11-20): each Pilot A encoder on the OTHER family, frozen. Dedication metric for Pilot C:
# A-spec on spatial TDOA/T60/DRR (--joint-mode spec), A-spat on speech SC2/SID (ipd16 probe path, as RQ2).
# Usage, from the SAC-SAR-SSL root: bash scripts/evals/leakage.sh [<jobid the spatial eval waits for (afterok)>]
set -euo pipefail
source scripts/evals/common.sh
DEP=${1:+--dependency=afterok:$1}
SPEC=$J/pilotA_spec/spec_encoder_final.pth
CLEAN_FROZEN_CKPT=1 EVAL_TASKS="$TASKS" EXTRA_ARGS="--upstream joint --joint-mode spec --joint-ckpt $SPEC" \
	sbatch --parsable $DEP --job-name=leak_spec_spatial scripts/ssamba_spatial_snapshot_eval.sbatch $SPEC jointA_spec_on_spatial lineareval
probe leak_spat_probe jointA_spat_probe.jsonl "" $J/pilotA_spat/spat_encoder_final.pth
