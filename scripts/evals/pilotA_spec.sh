#!/bin/bash
# Pilot A spectral encoder alone: speech probes (final + random init). Then A-concat if A-spat is also done.
set -euo pipefail
source scripts/evals/common.sh
probe jointA_spec_probe jointA_spec_probe.jsonl "--input_repr joint_spec" $J/pilotA_spec/spec_encoder_final.pth random
bash scripts/evals/pilotA_concat.sh
