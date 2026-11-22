#!/bin/bash
#SBATCH --job-name=swap_resume_test
#SBATCH --account=public-users_v2
#SBATCH --partition=gpu-general-pool
#SBATCH --exclude=compute-0-[18,53,376]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:30:00
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# Resume audit (agreed 2026-11-21) for swap_pretrain.py, C config, 2560-target subset (20 steps / epoch):
#   A   60 steps straight
#   B   40 steps, resume, 20 more                                  (should match A up to GPU nondeterminism)
#   CTL 40 steps, resume with the Adam state wiped, 20 more        (what a missing optimiser state looks like)
# then the gradient-norm distribution of Pilot C at random init, epoch 9 (best) and final (for --grad-clip).
set -euo pipefail
source ~/sac-ssamba/env.sh
export SAC_SSAMBA_SRC=$HOME/sac-ssamba/src
DATA=/scratch200/yotams2/SAC-SAR-SSL/data/MicSig
EXP=/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain/_resume_test
C=/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain/pilotC_swap
echo "$(date) $(hostname) job ${SLURM_JOB_ID:-local} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}/code"
rm -rf "$EXP"
A=(--tar $DATA/simu_paired_pretrain.tar --num-train 2560 --num-val 256 --log-every 10 --ckpt-every 20 --num-workers 5)
$PYTHON -u swap_pretrain.py "${A[@]}" --exp-dir $EXP/A --max-steps 60 > $EXP.A.log 2>&1
$PYTHON -u swap_pretrain.py "${A[@]}" --exp-dir $EXP/B --max-steps 40 > $EXP.B.log 2>&1
$PYTHON -u swap_pretrain.py "${A[@]}" --exp-dir $EXP/B --max-steps 60 >> $EXP.B.log 2>&1
$PYTHON -u swap_pretrain.py "${A[@]}" --exp-dir $EXP/CTL --max-steps 40 > $EXP.CTL.log 2>&1
$PYTHON - <<'PY'
import torch
p = '/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain/_resume_test/CTL/latest.pth'
st = torch.load(p, map_location='cpu', weights_only=False)
st['opt']['state'] = {}  # Adam moments and step counts gone, as if not saved
torch.save(st, p)
print('[ctl] wiped the Adam state in', p)
PY
$PYTHON -u swap_pretrain.py "${A[@]}" --exp-dir $EXP/CTL --max-steps 60 >> $EXP.CTL.log 2>&1
grep -h "^\[resume\]" $EXP.B.log $EXP.CTL.log
$PYTHON - <<'PY'
import json, torch
E = '/scratch200/yotams2/SAC-SAR-SSL/exp/joint_pretrain/_resume_test'
def log(r):
    return {(d['step'], 'val' in d): d.get('val', d) for d in map(json.loads, open(f'{E}/{r}/log.jsonl'))}
L = {r: log(r) for r in ('A', 'B', 'CTL')}
print('step  kind   ' + '  '.join(f'{r:>22s}' for r in L) + '    (swap / loss / grad_norm)')
for step in (30, 40, 50, 60):
    for val in (False, True):
        row = []
        for r in L:
            d = L[r].get((step, val))
            row.append('-' if d is None else f"{d['swap']:.4f} / {d['loss']:.3f} / {d.get('grad_norm', float('nan')):.3g}")
        if any(x != '-' for x in row):
            print(f'{step:4d}  {"val" if val else "train":5s}  ' + '  '.join(f'{x:>22s}' for x in row))
W = {r: torch.load(f'{E}/{r}/joint_final.pth', map_location='cpu')['model'] for r in L}
for r in ('B', 'CTL'):
    num = sum(float((W[r][k].float() - W['A'][k].float()).norm() ** 2) for k in W['A'] if W['A'][k].is_floating_point())
    den = sum(float(W['A'][k].float().norm() ** 2) for k in W['A'] if W['A'][k].is_floating_point())
    print(f'[weights] |{r} - A| / |A| = {(num / den) ** 0.5:.3e}')
PY
for tag in init best final; do $PYTHON -u swap_gradnorm.py $C $tag 60 2>&1 | grep "^\[gradnorm\]"; done
rm -rf "$EXP"
echo "$(date) done"
