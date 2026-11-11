#!/bin/bash
#SBATCH --job-name=stage_simu_ds
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --constraint=ib_dj
#SBATCH --exclude=compute-0-351
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out

# Copies the simulated downstream set (MicSig/simu_ds) from the /moshe-data backup to /scratch200, where GPU jobs can
# read it. Only train rooms R1..R${NROOM} are copied (default 256): run_downstream.py uses rooms
# 1..ntrial*nsimroom, which is at most 256 for --ds-nsimroom <= 256. val/ and test/ are copied whole.
# Safe to rerun: rsync skips what is already there. Submit from anywhere: sbatch scripts/stage_simu_ds.sbatch [NROOM]
set -euo pipefail
NROOM=${1:-256}
SRC=/moshe-data/blaufer-data/yotams2/SAC-SAR-SSL/data/MicSig/simu_ds
DST=${SARSSL_DATA_DIR:-/scratch200/yotams2/SAC-SAR-SSL/data}/MicSig/simu_ds
[ -d "$SRC" ] || { echo "no $SRC on $(hostname)"; exit 1; }
mkdir -p "$DST/train"
echo "$(date) copying val, test and train/R1..R${NROOM} from $SRC to $DST"
rsync -a "$SRC/val" "$SRC/test" "$DST/"
cp -p "$SRC/train/all_info.npz" "$DST/train/" 2>/dev/null || true
for i in $(seq 1 "$NROOM"); do rsync -a "$SRC/train/R$i" "$DST/train/"; done
for s in val test; do echo "$s: $(find "$DST/$s" -name '*.wav' | wc -l) wav (source $(find "$SRC/$s" -name '*.wav' | wc -l))"; done
echo "train rooms: $(ls "$DST/train" | grep -c '^R')"
touch "$DST/.staged_R${NROOM}"
echo "$(date) done"
