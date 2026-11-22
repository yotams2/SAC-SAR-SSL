#!/bin/bash
#SBATCH --job-name=swap_smoke
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,376]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# Pilot C smoke: unit checks on the 96-clip test set, then 40 training steps at bs 128 on the paired set (2560 targets,
# so two 20-step epochs: val, snapshot, best), then a resumed run to step 60. Output dir deleted at the end if all passed.
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
source ~/sac-ssamba/env.sh
export SAC_SSAMBA_SRC=$HOME/sac-ssamba/src
DATA=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig
EXP=/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain/_smoke_swap
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd "$REPO_ROOT/code"
rm -rf "$EXP"
$PYTHON -u swap_pretrain_check.py $DATA/_test_paired/paired_test.tar "$EXP/check"
A=(--tar $DATA/simu_paired_pretrain.tar --exp-dir "$EXP/run" --num-train 2560 --num-val 256 --log-every 10 --ckpt-every 20 --num-workers 5 ${SMOKE_ARGS:-})  # SMOKE_ARGS: e.g. C2 "--ctr-lambda 1", C + SAC "--sac-lambda 0.02 --grad-clip-median-mult 3"
$PYTHON -u swap_pretrain.py "${A[@]}" --max-steps 40
ls "$EXP/run"
$PYTHON -u swap_pretrain.py "${A[@]}" --max-steps 60 | tee "$EXP/resume.log"
grep -q '^\[resume\] step 40' "$EXP/resume.log"
if [[ "${SMOKE_ARGS:-}" == *ctr-lambda* ]]; then grep -q '"ctr_spec_hard"' "$EXP/run/log.jsonl" && grep '"val"' "$EXP/run/log.jsonl" | tail -1 | grep -q '"ctr_spat_hard"'; fi
if [[ "${SMOKE_ARGS:-}" == *sac-lambda* ]]; then  # SAC logged (train + val); the resume restores its stats instead of recomputing
	grep -q '"sac"' "$EXP/run/log.jsonl"; grep '"val"' "$EXP/run/log.jsonl" | tail -1 | grep -q '"sac"'
	if grep -q '^\[stats\] SAC' "$EXP/resume.log"; then echo "[smoke] resume recomputed SAC stats"; exit 1; fi
fi
if [[ "${SMOKE_ARGS:-}" == *grad-clip-median* ]]; then grep -q '"grad_clip_thr"' "$EXP/run/log.jsonl"; fi
for f in spat_encoder_final.pth spec_encoder_final.pth joint_final.pth spat_encoder_epoch2.pth; do [ -f "$EXP/run/$f" ]; done
echo "[smoke] all passed"; rm -rf "$EXP"
echo "$(date) done"
