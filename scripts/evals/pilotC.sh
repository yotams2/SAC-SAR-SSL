#!/bin/bash
# Pilot C family (swap reconstruction): same comparison set as A/B (frozen spatial TDOA/T60/DRR, speech SC2/SID) for each
# encoder on its own family, the concat on both, and the leakage probes (each encoder on the other family).
# PC_RUN = run dir under joint_pretrain, PC_LABEL = label prefix, PC_TAG = checkpoint tag (defaults: Pilot C seed 1, final). Wrappers: pilotC2.sh, pilotC_s2.sh.
set -euo pipefail
source scripts/evals/common.sh
RUN=${PC_RUN:-pilotC_swap}; L=${PC_LABEL:-jointC}; TAG=${PC_TAG:-final}  # TAG: final | best | epoch2 | epoch5
D=$J/$RUN
SPEC=$D/spec_encoder_$TAG.pth; SPAT=$D/spat_encoder_$TAG.pth
spatial ${L}_spat $SPAT ${L}_spat
probe ${L}_spec_probe ${L}_spec_probe.jsonl "--input_repr joint_spec" $SPEC
spatial ${L}_concat $SPEC ${L}_concat "--upstream joint --joint-mode concat --joint-ckpt $SPEC,$SPAT"
probe ${L}_concat_probe ${L}_concat_probe.jsonl "--input_repr joint_concat" $SPEC,$SPAT
spatial ${L}_spec_on_spatial $SPEC ${L}_spec_on_spatial "--upstream joint --joint-mode spec --joint-ckpt $SPEC"
probe ${L}_spat_probe ${L}_spat_probe.jsonl "" $SPAT
