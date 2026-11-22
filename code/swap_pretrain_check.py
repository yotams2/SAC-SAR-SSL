""" Checks for swap_pretrain.py (Pilot C). GPU. Usage: python swap_pretrain_check.py <paired tar> <tmp dir> """
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.utils_module as at_module  # noqa: E402
import joint_pretrain as jp  # noqa: E402
import joint_spatial  # noqa: E402
import swap_pretrain as sp  # noqa: E402
from ssamba_spatial_pretrain import stft_reim  # noqa: E402

tar, tmp = sys.argv[1:3]
os.makedirs(tmp, exist_ok=True)
dev = 'cuda'
torch.manual_seed(0)
m = np.load(tar + '.meta.npz', allow_pickle=True)

# 1. partners: spectral = same utterance, other scene; spatial = same scene, other utterance
ds = sp.PairedDataset(tar, 0, int(m['n_scenes']))
k = ds.k
for c in range(len(ds)):
	for _ in range(3):
		cs, cp = ds.partners(c)
		assert m['utt'][cs] == m['utt'][c] and m['scene'][cs] != m['scene'][c], ('spec partner', c, cs)
		assert m['scene'][cp] == m['scene'][c] and m['utt'][cp] != m['utt'][c], ('spat partner', c, cp)
		assert m['src_md5'][cs] == m['src_md5'][c] and np.allclose(m['TDOA'][cp], m['TDOA'][c])
x, ids5 = ds[5]
assert x.shape == (3, 65792, 2) and not np.allclose(x[0], x[1]) and not np.allclose(x[0], x[2])
for i in range(len(ds)):  # ids = [utterance, scene] of (target, spectral input, spatial input)
	_, ii = ds[i]
	assert ii[0, 0] == ii[1, 0] and ii[0, 1] != ii[1, 1] and ii[0, 1] == ii[2, 1] and ii[0, 0] != ii[2, 0], (i, ii)
	assert ii[0, 0] == m['utt'][ds.targets[i]] and ii[0, 1] == m['scene'][ds.targets[i]]
dsf = sp.PairedDataset(tar, 0, int(m['n_scenes']), fixed_seed=12345)
a1 = [dsf.partners(c) for c in range(len(dsf))]
torch.manual_seed(99)
assert a1 == [dsf.partners(c) for c in range(len(dsf))], 'fixed pairing not reproducible'
assert len(set(a1)) > len(dsf) // 2
print(f'[check] partners + ids OK for {len(ds)} targets (k {k}); fixed validation pairing reproducible; item {x.shape}')

# 2. forward / backward, both settings
stft_mod = at_module.STFT(win_len=jp.WIN_LEN, win_shift_ratio=jp.WIN_SHIFT_RATIO, nfft=jp.NFFT).to(dev)
items = [ds[i] for i in range(0, 16, 2)]  # 8 targets from 4 scenes: shared scenes/utterances exercise the masking
sig3 = torch.as_tensor(np.stack([a for a, _ in items])).float().to(dev)
ids = torch.as_tensor(np.stack([b for _, b in items])).to(dev)
for own, ctr, nolog in ((0.1, 0.0, False), (0.0, 0.0, False), (0.1, 1.0, False), (0.1, 1.0, True)):
	torch.manual_seed(1)
	model = sp.SwapPretrain(7, 10.0, 1.0, 10.0, own, ctr_lambda=ctr, spat_mpg_no_logL=nolog).to(dev).train()
	torch.manual_seed(2); random.seed(2)  # same column masks (python random) for the repeat below
	out = model(sig3, stft_mod, diag=True, ids=ids)
	assert ('ctr_spec' in out) == (ctr > 0)
	if ctr > 0:
		for kk in ('spec', 'spat'):
			assert torch.isfinite(out[f'ctr_{kk}']) and 0 <= float(out[f'ctr_{kk}_hard']) <= 1, out
		print(f'[check] ctr spec {out["ctr_spec"].item():.3f} (acc {out["ctr_spec_acc"].item():.2f}, hard '
			  f'{out["ctr_spec_hard"].item():.2f}), spat {out["ctr_spat"].item():.3f} (acc {out["ctr_spat_acc"].item():.2f}, '
			  f'hard {out["ctr_spat_hard"].item():.2f}); spat MPG no logL {nolog}: {out["spat_mpg"].item():.3f}')
	elif own > 0:
		# C's objective must not change with the new code paths off: same pass (eval mode, no grad, same seeds) with the
		# new call (ids) and with C's call signature (no ids)
		model.eval()
		with torch.no_grad():
			torch.manual_seed(3); random.seed(3)
			l_new = model(sig3, stft_mod, ids=ids)['loss'].item()
			torch.manual_seed(3); random.seed(3)
			l_old = model(sig3, stft_mod)['loss'].item()
		model.train()
		assert abs(l_new - l_old) < 1e-5 * abs(l_old), (l_new, l_old)
		print(f'[check] C objective identical with the new code paths off: {l_new:.5f} vs {l_old:.5f}')
	assert torch.isfinite(out['loss']), out
	assert ('spat_mpg' in out) == (own > 0) and 'swap_shufspat' in out and 'swap_sinIPD_shufspec' in out
	out['loss'].backward()
	for name, mod in (('spec', model.streams['spec'].enc.v.layers[0]), ('spat', model.streams['spat'].enc.v.layers[0]),
					  ('decoder', model.decoder.head)):
		g = sum(p.grad.abs().sum() for p in mod.parameters() if p.grad is not None)
		assert g > 0, (own, ctr, name)
	chans = ', '.join('%s %.3f' % (n, out['swap_' + n].item()) for n in sp.CH_NAMES)
	print(f'[check] own {own} ctr {ctr}: loss {out["loss"].item():.3f}, swap {out["swap"].item():.3f} '
		  f'({chans}); grads reach both encoders + decoder')

# 2b. C + SAC (11-22): features of the spatial input (view 2) = the ipd16_sac path; finite; grads reach the SAC head and
# the spatial encoder; the SAC term alone gives the spectral encoder no gradient
from ssamba_spatial_pretrain import sac_feature_stats, SAC_FEATURES, FS  # noqa: E402
torch.manual_seed(1)
ms = sp.SwapPretrain(7, 10.0, 1.0, 10.0, 0.1, ctr_lambda=1.0, sac_lambda=0.02).to(dev).train()
ms.feature_stats = sac_feature_stats(stft_mod, [sig3[:, 0].cpu(), sig3[:, 2].cpu()], dev)
out = ms(sig3, stft_mod, ids=ids)
_, reim_s = stft_reim(stft_mod, sig3[:, 2])
f_ref = at_module.spatial_features_from_stft(torch.view_as_complex(reim_s.contiguous()), sample_rate=FS)
f_mod = at_module.spatial_features_from_stft(ms._stft_spat, sample_rate=FS)
dmax = max(float((f_ref[k] - f_mod[k]).abs().max()) for k in SAC_FEATURES)
print(f'[check] SAC features: model view 2 vs ipd16_sac path (stft_reim) max|diff| {dmax:.2e}')
assert dmax < 1e-4, dmax
assert 'sac' in out and torch.isfinite(out['sac']), out
ms.zero_grad(); (0.02 * out['sac']).backward()
g = lambda mod: sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
assert g(ms.feat_sim_proj) > 0 and g(ms.streams['spat'].enc.v.layers[0]) > 0, 'SAC gradient missing'
assert g(ms.streams['spec'].enc.v.layers[0]) == 0, 'SAC must not reach the spectral encoder'
print(f'[check] SAC {out["sac"].item():.4f}: grads reach the head + spatial encoder only; total loss {out["loss"].item():.3f}')
assert sp.SwapPretrain(7, 10.0, 1.0, 10.0, 0.1).feat_sim_proj is None  # off: no head, old checkpoints load unchanged
del ms

# 3. saved encoders load in the Pilot A/B eval path (concat) and give the model's own embeddings
model.eval()
model.streams['spec'].in_mean.fill_(0.3); model.streams['spec'].in_std.fill_(1.7)
model.streams['spat'].in_mean.uniform_(-1, 1); model.streams['spat'].in_std.uniform_(0.5, 2)
args = argparse.Namespace(exp_dir=tmp, streams='joint', fusion=False)
model.save_encoders(args, 'final')
cc = joint_spatial.JointSpatialDownstream(f'{tmp}/spec_encoder_final.pth,{tmp}/spat_encoder_final.pth', 'concat').to(dev).eval()
sig3 = sig3[:4]
_, reim = stft_reim(stft_mod, sig3[:, 0])
e = cc.embed(reim)
with torch.no_grad():
	stft = cc.stft_n(reim)
	X = model.raw_inputs(stft)
	h = model.encode({kk: model.streams[kk].norm_input(v) for kk, v in X.items()}, None, fusion=False)
	ref = torch.cat([h[kk][:, 1:].reshape(4, model.streams[kk].n_rows, jp.N_COLS, -1).float().mean(dim=(1, 2))
					 for kk in ('spec', 'spat')], -1)
d = float((e - ref).abs().max())
print(f'[check] saved encoders through joint_spatial concat vs model encode: max|diff| {d:.2e}')
assert d < 1e-3
assert os.path.isfile(f'{tmp}/joint_final.pth')
print('[check] all OK')
