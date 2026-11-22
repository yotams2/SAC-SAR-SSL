#!/bin/bash
#SBATCH --job-name=gen_paired
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/a/home/cc/students/enginer/yotams2/logs/%x_%j.out
# Paired scene x utterance set (Pilot C). Usage: sbatch [--time ...] scripts/gen_simu_paired.sbatch <out tar> "<gen_simu_paired.py args>" [test]
# 'test' renders the args' set, a rerun of its scenes 0-1, then runs paired_check.py.
set -euo pipefail
OUT=$1; ARGS=($2); MODE=${3:-}
D=/a/home/cc/students/enginer/yotams2/SAC-SAR-SSL/code/data_generation
SRC=/scratch200/yotams2/SAC-SAR-SSL/data/SrcSig/wsj0/tr
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
quota -s 2>/dev/null | grep -A2 scratch200 || true
# shared venv (numpy/scipy/soundfile/torch) + gpuRIR from the old sac-sar-ssl venv (it lacks soundfile), via a link dir
source ~/sac-ssamba/env.sh
module load CUDA/CUDA-12.2 2>/dev/null || true
export PYTHONPATH=/scratch200/yotams2/venvs/gpurir_shim${PYTHONPATH:+:$PYTHONPATH}
cd $D
$PYTHON -c "import gpuRIR, soundfile; print('gpuRIR', gpuRIR.__file__)"
$PYTHON -u gen_simu_paired.py --src_dir $SRC --out_tar $OUT "${ARGS[@]}"
if [ "$MODE" = test ]; then
	$PYTHON -u gen_simu_paired.py --src_dir $SRC --out_tar ${OUT%.tar}_rerun.tar "${ARGS[@]}" --last_scene 2
	$PYTHON -u paired_check.py $OUT ${OUT%.tar}_rerun.tar
fi
echo "$(date) done"
