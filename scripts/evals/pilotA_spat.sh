#!/bin/bash
# Pilot A spatial encoder alone: spatial tasks (frozen). Then A-concat if A-spec is also done.
set -euo pipefail
source scripts/evals/common.sh
spatial jointA_spat $J/pilotA_spat/spat_encoder_final.pth jointA_spat
bash scripts/evals/pilotA_concat.sh
