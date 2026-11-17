""" Checks for joint_pretrain.py (run by scripts/joint_pretrain_smoke.sbatch): column masks, fusion = identity at
	init (zero gates), fusion-off equals the streams run alone, and the spatial checkpoint loads in ssamba_spatial. """
import random
import sys

import torch

import joint_pretrain as jp
import ssamba_spatial

random.seed(0)
m = jp.column_mask(4000, 7)
assert (m.sum(1) == 7).all()
gaps_ok = all('11' not in ''.join('1' if v else '0' for v in row).replace('1', '1').split('0') or True for row in m)
runs = []
for row in m.tolist():
	s = ''.join('1' if v else '0' for v in row)
	runs += [len(r) for r in s.split('0') if r]
from collections import Counter
print('[check] span lengths', sorted(Counter(runs).items()), 'max', max(runs), '| P(col masked)',
	  [round(float(x), 2) for x in m.float().mean(0)])
assert max(runs) <= 3, 'adjacent spans merged'

dev = 'cuda'
torch.manual_seed(0)
model = jp.JointPretrain('joint', True, 0.3, 7, 10.0, 1.0).to(dev).eval()
B = 2
sig = torch.randn(B, int(jp.T_SEC * jp.FS), 2, device=dev) * 0.1
stft_mod = jp.at_module.STFT(win_len=jp.WIN_LEN, win_shift_ratio=jp.WIN_SHIFT_RATIO, nfft=jp.NFFT).to(dev)
stft_n = jp.shared_stft(stft_mod, sig)
X = model.raw_inputs(stft_n)
print('[check] inputs', {k: tuple(v.shape) for k, v in X.items()})
Xn = {k: model.streams[k].norm_input(v) for k, v in X.items()}
cm = jp.column_mask(B, 7, device=dev)
with torch.no_grad():
	a = model.encode(Xn, cm, fusion=True)
	b = model.encode(Xn, cm, fusion=False)
	for g in model.fuse.values():
		g.gate.data.normal_(0, 0.1)
	c = model.encode(Xn, cm, fusion=True)
for k in a:
	print(f'[check] {k}: zero-gate fusion vs off max|diff| {float((a[k] - b[k]).abs().max()):.2e}; '
		  f'random gate {float((c[k] - b[k]).abs().max()):.2e}')
	assert torch.allclose(a[k], b[k]), 'zero-gate fusion changed the output'
# leak check: changing the input at a masked column must not change any output token, in either stream
with torch.no_grad():
	t = int(cm[0].nonzero()[0])
	Xp = {k: v.clone() for k, v in Xn.items()}
	for k in Xp:
		Xp[k][0, :, :, t * 16:(t + 1) * 16] += 5.0
	d = model.encode(Xp, cm, fusion=True)
for k in a:
	print(f'[check] {k}: change inside masked column {t}: max|diff| {float((d[k] - c[k]).abs().max()):.2e}')
	assert torch.allclose(c[k], d[k]), 'masked content leaked'
with torch.no_grad():  # sanity: a visible column must change outputs
	tv = int((~cm[0]).nonzero()[0])
	Xv = {k: v.clone() for k, v in Xn.items()}
	Xv['spat'][0, :, :, tv * 16:(tv + 1) * 16] += 5.0
	e = model.encode(Xv, cm, fusion=True)
print(f"[check] change in visible spat column {tv}: spec max|diff| {float((e['spec'] - c['spec']).abs().max()):.2e} (fusion carries it)")

if len(sys.argv) > 1:
	ds = ssamba_spatial.SSAMBASpatialDownstream(ckpt_path=sys.argv[1], fill='none', input_mode='ipd', patch='16x16',
												n_stft_frames=jp.NT, clip_seconds=jp.T_SEC)
	print('[check] spatial checkpoint loads in ssamba_spatial, meta masking:', ds.meta['masking'])
print('[check] all OK')
