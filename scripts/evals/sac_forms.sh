#!/bin/bash
# SAC-form comparison (branch sac-forms, 11-19): one 10-epoch ipd16 SAC run per form, evaluated exactly as the
# existing ipd16 ep10 snapshots (pre_sac_ep10): TDOA + T60, finetune + frozen, R8, 20 ep, lr 1e-3/1e-4.
# Usage (by pretrain_gate.sbatch, from the repo root): bash scripts/evals/sac_forms.sh <label>
set -euo pipefail
L=$1
D=/scratch200/yotams2/SAC-SAR-SSL/exp/ssamba_pretrain/$L
sbatch --parsable --job-name=${L}_eval scripts/ssamba_spatial_snapshot_eval.sbatch $D/encoder_final.pth $L
