#!/bin/bash
# Early leakage read (agreed 2026-11-21): the two cheap leakage evals on one snapshot of a Pilot C-family run:
#   spatial encoder on speech (SC2/SID probe, ~10 min) + spectral encoder on T60 (frozen, ~2 h).
# Usage, from the SAC-SAR-SSL root: bash scripts/evals/early_leak.sh <run dir under joint_pretrain> <label prefix> [tag]
set -euo pipefail
source scripts/evals/common.sh
RUN=$1 L=$2 TAG=${3:-epoch2}
SPEC=$J/$RUN/spec_encoder_$TAG.pth; SPAT=$J/$RUN/spat_encoder_$TAG.pth
probe ${L}_spat_probe ${L}_spat_probe.jsonl "" $SPAT
CLEAN_FROZEN_CKPT=1 EVAL_TASKS="T60" EXTRA_ARGS="--upstream joint --joint-mode spec --joint-ckpt $SPEC" \
	sbatch --parsable --job-name=${L}_spec_T60 scripts/ssamba_spatial_snapshot_eval.sbatch $SPEC ${L}_spec_on_spatial lineareval
