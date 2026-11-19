#!/bin/bash
# Pilot B, fusion OFF (each encoder alone): spatial encoder on the 5 SAR-SSL tasks x finetune/frozen, both encoders on
# the speech probes. Fusion-on evals need joint_{best,final}.pth and code that is not written yet.
set -euo pipefail
D=/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain/pilotB_fusion
for t in TDOA T60 DRR C50 ABS; do
	EVAL_TASKS=$t sbatch --parsable --job-name=jointB_spat_$t scripts/ssamba_spatial_snapshot_eval.sbatch $D/spat_encoder_final.pth jointB_spat_off
done
(cd ~/sac-ssamba && sbatch --parsable --job-name=jointB_spec_probe scripts/ipd16_probe.sbatch \
	/scratch200/yotams2/sac-ssamba/exp/_probes/jointB_spec_probe.jsonl "--input_repr joint_spec" $D/spec_encoder_final.pth)
(cd ~/sac-ssamba && sbatch --parsable --job-name=jointB_spat_probe scripts/ipd16_probe.sbatch \
	/scratch200/yotams2/sac-ssamba/exp/_probes/jointB_spat_probe.jsonl "" $D/spat_encoder_final.pth)
