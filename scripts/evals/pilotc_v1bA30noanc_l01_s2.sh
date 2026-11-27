#!/bin/bash
# pilotc_v1bA30noanc_l01_s2 (final 30 ep, scripts/launch_v1bA30.sh, ANCHORS=off, seed 2): check 4a + ridge + Pilot C's eval set (labels jointcv1bA30noancl01s2_*).
# Check 4a on joint_final, ridge dump + ridge over the v1 dumps and this run, then Pilot C's eval set (labels jointcv1bA30noancl01s2_*).
set -euo pipefail
J=${JOINT_EXP:-/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain}
D=$J/pilotc_v1bA30noanc_l01_s2
R=/scratch200/yotams2/SAC-SAR-SSL/exp/ridge
sbatch --parsable scripts/swap_shufdiag.sbatch /scratch200/yotams2/SAC-SAR-SSL/exp/shufdiag/pilotc_v1bA30noanc_l01_s2.json jointcv1bA30noancl01s2_final=$D:joint_final.pth
extra="jointC30v2r=$R/jointC30v2r.npz"  # baseline = C 30 ep
rd=$(sbatch --parsable --job-name=ridge_dump_jointcv1bA30noancl01s2 scripts/ridge_dump.sbatch $D/spec_encoder_final.pth $D/spat_encoder_final.pth $R/jointcv1bA30noancl01s2.npz)
sbatch --parsable --dependency=afterok:$rd scripts/ridge_probe.sbatch $R/ridge_jointcv1bA30noancl01s2.json A=$R/A.npz C=$R/C.npz Cs2=$R/Cs2.npz $extra jointcv1bA30noancl01s2=$R/jointcv1bA30noancl01s2.npz
PC_RUN=pilotc_v1bA30noanc_l01_s2 PC_LABEL=jointcv1bA30noancl01s2 exec bash scripts/evals/pilotC.sh
