#!/bin/bash
# Paired-set A spatial control (Pilot A spatial recipe on the Pilot C paired set): spatial tasks + leakage on speech.
set -euo pipefail
source scripts/evals/common.sh
SPAT=$J/pilotA_spat_paired/spat_encoder_final.pth
spatial jointAp_spat $SPAT jointAp_spat
probe jointAp_spat_probe jointAp_spat_probe.jsonl "" $SPAT
