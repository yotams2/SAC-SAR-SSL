#!/bin/bash
# A-concat: both frozen Pilot A encoders, concatenated, on both families. Called by both A gates; submits once, when
# both pretrainings are done (mkdir is atomic, so the two gates can't both submit).
set -euo pipefail
source scripts/evals/common.sh
[ -f $J/pilotA_spec/.done ] && [ -f $J/pilotA_spat/.done ] || { echo "A-concat: waiting for the other A run"; exit 0; }
mkdir $J/.pilotA_concat_submitted 2>/dev/null || { echo "A-concat already submitted"; exit 0; }
CK=$J/pilotA_spec/spec_encoder_final.pth,$J/pilotA_spat/spat_encoder_final.pth
spatial jointA_concat $J/pilotA_spec/spec_encoder_final.pth jointA_concat "--upstream joint --joint-mode concat --joint-ckpt $CK"
probe jointA_concat_probe jointA_concat_probe.jsonl "--input_repr joint_concat" $CK
