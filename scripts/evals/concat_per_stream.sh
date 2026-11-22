#!/bin/bash
# Concat probe check (11-23): C2 concat < C2 spatial alone on T60/DRR. Rerun the spatial concat evals with one
# LayerNorm per encoder half (--joint-head-norm per_stream) for A, C s1, C s2, C2. Everything else as the original rows.
set -euo pipefail
source scripts/evals/common.sh
for x in A:pilotA_spec:pilotA_spat C:pilotC_swap:pilotC_swap C2:pilotC2_ctr:pilotC2_ctr Cs2:pilotC_swap_s2:pilotC_swap_s2; do
	IFS=: read L RS RT <<< "$x"
	SPEC=$J/$RS/spec_encoder_final.pth; SPAT=$J/$RT/spat_encoder_final.pth
	spatial joint${L}_concat_ps $SPEC joint${L}_concat_ps "--upstream joint --joint-mode concat --joint-head-norm per_stream --joint-ckpt $SPEC,$SPAT"
done
