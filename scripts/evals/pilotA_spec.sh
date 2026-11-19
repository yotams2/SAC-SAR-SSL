#!/bin/bash
# Pilot A spectral encoder: frozen speech probes (RQ2 subsets), final + epoch 2 + random init.
set -euo pipefail
D=/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain/pilotA_spec
(cd ~/sac-ssamba && sbatch --parsable --job-name=jointA_spec_probe scripts/ipd16_probe.sbatch \
	/scratch200/yotams2/sac-ssamba/exp/_probes/jointA_spec_probe.jsonl "--input_repr joint_spec" \
	$D/spec_encoder_final.pth $D/spec_encoder_epoch2.pth random)
