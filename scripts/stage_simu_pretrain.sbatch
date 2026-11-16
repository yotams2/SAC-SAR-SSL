#!/bin/bash
#SBATCH --job-name=stage_simu_pre
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --constraint=ib_dj
#SBATCH --exclude=compute-0-351
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out

# Packs the SAR-SSL pretraining set (MicSig/simu: pretrain 50k, preval, pretest; wav + _info.npz, no _spec_feats.pt)
# from the /moshe-data backup into ONE uncompressed tar on /scratch200. Why a tar: the shared /scratch200 filesystem is
# at 100% of its inodes (~310k free on 2026-11-12) and the loose set is ~112k files. Training jobs unpack the tar into
# node RAM (/dev/shm) at start; see scripts/unpack_simu_pretrain.sh.
# Safe to rerun: writes to a .partial file and renames only on success.
# Submit from anywhere: sbatch scripts/stage_simu_pretrain.sbatch
set -euo pipefail
SRC=/moshe-data/blaufer-data/yotams2/SAC-SAR-SSL/data/MicSig/simu
DST_DIR=${SARSSL_DATA_DIR:-/scratch200/yotams2/SAC-SAR-SSL/data}/MicSig
TAR=$DST_DIR/simu_pretrain.tar
[ -d "$SRC" ] || { echo "no $SRC on $(hostname)"; exit 1; }
mkdir -p "$DST_DIR"
for s in pretrain preval pretest; do
	echo "source $s: $(find "$SRC/$s" -name '*.wav' | wc -l) wav, $(find "$SRC/$s" -name '*_info.npz' | wc -l) info.npz"
done
echo "$(date) writing $TAR.partial on $(hostname)"
n=$(tar -cvf "$TAR.partial" -C "$SRC" --exclude='*_spec_feats.pt' pretrain preval pretest | grep -vc '/$')
mv "$TAR.partial" "$TAR"
echo "$(date) done: $n files, $(stat -c %s "$TAR" | numfmt --to=iec) in $TAR"
echo "$n" > "$TAR.nfiles"
