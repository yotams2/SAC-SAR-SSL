""" Gradient-norm distribution of the Pilot C objective at a saved full checkpoint (joint_<tag>.pth) or at random init,
    over N training batches (no optimiser step). Used to set --grad-clip (agreed 2026-11-21: ~3x the median norm).
    Usage: python swap_gradnorm.py <exp dir> <tag|init> [n_batches] """
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.utils_module as at_module  # noqa: E402
import swap_pretrain as sp  # noqa: E402
from ssamba_spatial_pretrain import NFFT, WIN_LEN, WIN_SHIFT_RATIO  # noqa: E402

exp, tag = sys.argv[1:3]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
a = json.load(open(os.path.join(exp, 'args.json')))
dev = 'cuda'
torch.manual_seed(0)
model = sp.SwapPretrain(a['mask_cols'], a['recon_lambda'], a['classif_lambda'], a['swap_lambda'], a['own_scale']).to(dev)
st = torch.load(os.path.join(exp, 'joint_best.pth'), map_location='cpu')['model']
if tag == 'init':  # random weights, but the run's input statistics
	for k in ('spec', 'spat'):
		model.streams[k].in_mean.copy_(st[f'streams.{k}.in_mean']); model.streams[k].in_std.copy_(st[f'streams.{k}.in_std'])
else:
	model.load_state_dict(torch.load(os.path.join(exp, f'joint_{tag}.pth'), map_location='cpu')['model'])
model.train()
n_scenes = int(np.load(a['tar'] + '.meta.npz')['n_scenes'])
ds = sp.PairedDataset(a['tar'], 0, n_scenes - a['val_scenes'])
loader = torch.utils.data.DataLoader(ds, batch_size=a['bs'], shuffle=True, num_workers=5,
									 generator=torch.Generator().manual_seed(1))
stft_mod = at_module.STFT(win_len=WIN_LEN, win_shift_ratio=WIN_SHIFT_RATIO, nfft=NFFT).to(dev)
params = [p for p in model.parameters() if p.requires_grad]
norms = []
for i, (sig3, ids) in enumerate(loader):
	if i >= n:
		break
	model.zero_grad(set_to_none=True)
	out = model(sig3.to(dev), stft_mod, ids=ids.to(dev))
	out['loss'].backward()
	norms.append(float(torch.nn.utils.clip_grad_norm_(params, float('inf'))))
q = np.percentile(norms, [5, 25, 50, 75, 95, 100])
print(f'[gradnorm] {tag}: n {len(norms)}  p5 {q[0]:.3g}  p25 {q[1]:.3g}  median {q[2]:.3g}  p75 {q[3]:.3g}  p95 {q[4]:.3g}  '
	  f'max {q[5]:.3g}', flush=True)
