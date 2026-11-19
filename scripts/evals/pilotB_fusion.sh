#!/bin/bash
# Pilot B: fusion OFF (each encoder alone, on its own family) and fusion ON (both streams, both families).
set -euo pipefail
source scripts/evals/common.sh
D=$J/pilotB_fusion
spatial jointB_spat_off $D/spat_encoder_final.pth jointB_spat_off
probe jointB_spec_off_probe jointB_spec_off_probe.jsonl "--input_repr joint_spec" $D/spec_encoder_final.pth
spatial jointB_fusion_on $D/joint_final.pth jointB_fusion_on "--upstream joint --joint-mode fusion --joint-ckpt $D/joint_final.pth"
probe jointB_fusion_on_probe jointB_fusion_on_probe.jsonl "--input_repr joint_fusion" $D/joint_final.pth
