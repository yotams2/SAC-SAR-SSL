#!/bin/bash
# Pilot A spatial encoder: 5 SAR-SSL tasks x finetune/frozen (one job per task) + frozen speech probes (RQ2 subsets).
# Run from the SAC-SAR-SSL root by scripts/pretrain_gate.sbatch.
set -euo pipefail
D=/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain/pilotA_spat
for t in TDOA T60 DRR C50 ABS; do
	EVAL_TASKS=$t sbatch --parsable --job-name=jointA_spat_$t scripts/ssamba_spatial_snapshot_eval.sbatch $D/spat_encoder_final.pth jointA_spat_final
done
(cd ~/sac-ssamba && sbatch --parsable --job-name=jointA_spat_probe scripts/ipd16_probe.sbatch \
	/scratch200/yotams2/sac-ssamba/exp/_probes/jointA_spat_probe.jsonl "" $D/spat_encoder_final.pth)
