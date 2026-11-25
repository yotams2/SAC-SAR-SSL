#!/bin/bash
# pilotc_v3 (V3 unpaired swap, scripts/launch_v3.sh): check 4a (on C v2's paired-v2 validation split) + ridge + Pilot C's eval set (labels jointcv3_*).
# Check 4a on joint_final, ridge dump + ridge over the v1 dumps and this run, then Pilot C's eval set (labels jointcv3_*).
set -euo pipefail
J=${JOINT_EXP:-/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain}
D=$J/pilotc_v3
R=/scratch200/yotams2/SAC-SAR-SSL/exp/ridge
SHUFDIAG_TAR=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar SHUFDIAG_VAL_SCENES=500 sbatch --parsable scripts/swap_shufdiag.sbatch /scratch200/yotams2/SAC-SAR-SSL/exp/shufdiag/pilotc_v3.json jointcv3_final=$D:joint_final.pth
extra=""; [ -f $R/jointCv2r.npz ] && extra="jointCv2r=$R/jointCv2r.npz"  # the C v2 baseline
rd=$(sbatch --parsable --job-name=ridge_dump_jointcv3 scripts/ridge_dump.sbatch $D/spec_encoder_final.pth $D/spat_encoder_final.pth $R/jointcv3.npz)
sbatch --parsable --dependency=afterok:$rd scripts/ridge_probe.sbatch $R/ridge_jointcv3.json A=$R/A.npz C=$R/C.npz Cs2=$R/Cs2.npz $extra jointcv3=$R/jointcv3.npz
PC_RUN=pilotc_v3 PC_LABEL=jointcv3 exec bash scripts/evals/pilotC.sh
