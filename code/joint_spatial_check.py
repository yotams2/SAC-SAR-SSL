""" Checks for joint_spatial.JointSpatialDownstream (A-concat / B fusion-on upstream). GPU.
Usage: python joint_spatial_check.py <out dir>   (writes fake checkpoints there, in the formats joint_pretrain.py saves)
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_pretrain as jp  # noqa: E402
import joint_spatial  # noqa: E402
import ssamba_spatial  # noqa: E402
from ssamba_spatial_pretrain import stft_reim  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
dev = 'cuda'
torch.manual_seed(0)

# fake checkpoints from a random joint model, saved with joint_pretrain.py's own writers
m = jp.JointPretrain('joint', True, 0.3, 7, 10.0, 1.0, use_checkpointing=False)
for k, s in m.streams.items():  # non-trivial input statistics
	s.in_mean.uniform_(-1, 1); s.in_std.uniform_(0.5, 2)
args = argparse.Namespace(streams='joint', fusion=True)
torch.save(m.spec_checkpoint(args), f'{out}/spec_encoder_final.pth')
torch.save(m.spat_checkpoint(args), f'{out}/spat_encoder_final.pth')
torch.save({'model': m.state_dict(), 'args': vars(args)}, f'{out}/joint_zero_gate.pth')
for g in m.fuse.values():
	g.gate.data.normal_(0, 0.1)
torch.save({'model': m.state_dict(), 'args': vars(args)}, f'{out}/joint_final.pth')

# 1. DC bin carries no mel weight
fb = jp.Inputs(None).mel_fb
print(f'[check] mel filterbank weight on DC bin: {float(fb[0].abs().max()):.2e}')
assert float(fb[0].abs().max()) == 0.0

# SAR-SSL-style input: STFTLearner normalisation, bins 1..256; one full clip and one short (TDOA-like 1 s) clip
stft_mod = jp.at_module.STFT(win_len=jp.WIN_LEN, win_shift_ratio=jp.WIN_SHIFT_RATIO, nfft=jp.NFFT).to(dev)
for name, n in (('4.1 s', int(jp.T_SEC * jp.FS)), ('1 s', 16384)):
	sig = torch.randn(3, n, 2, device=dev) * 0.1
	_, reim = stft_reim(stft_mod, sig)  # (3, 2, 256, nt, 2), what STFTLearner hands the model

	concat = joint_spatial.JointSpatialDownstream(f'{out}/spec_encoder_final.pth,{out}/spat_encoder_final.pth', 'concat').to(dev).eval()
	e = concat.embed(reim)
	# 2a. spatial half == SSAMBASpatialDownstream (the single-encoder SAR-SSL eval path)
	w = ssamba_spatial.SSAMBASpatialDownstream(ckpt_path=f'{out}/spat_encoder_final.pth', frozen=True, fill='none',
											   input_mode='ipd', n_stft_frames=jp.NT).to(dev).eval()
	with torch.no_grad():
		_, e_spat, _ = w(reim)
	d_spat = float((e[:, 768:] - e_spat).abs().max())
	# 2b. spectral half == standalone spec encoder on the full STFT (with its DC bin)
	enc = jp.build_spectral_encoder(False).to(dev).eval()
	ck = torch.load(f'{out}/spec_encoder_final.pth', weights_only=False)
	enc.load_state_dict(ck['state_dict'])
	with torch.no_grad():
		stft_full = jp.shared_stft(stft_mod, sig)
		nr = min(stft_full.shape[-1], jp.NT)
		X = (jp.Inputs(None).to(dev).spec(stft_full) - ck['meta']['in_mean']) / ck['meta']['in_std']
		X[..., nr:] = 0
		h = enc._encode_with_mamba(X)[:, 1:].reshape(3, 8, 16, -1)
		ntr = -(-nr // 16)
		e_spec = h[:, :, :ntr].mean(dim=(1, 2))
	d_spec = float((e[:, :768] - e_spec).abs().max())
	print(f'[check] {name}: concat spat half vs ssamba_spatial {d_spat:.2e}; spec half vs standalone encoder {d_spec:.2e} '
		  f'(|e| max {float(e.abs().max()):.2f})')
	assert d_spat < 1e-3 and d_spec < 1e-3

	# 3. fusion: zero gates == concat of the same weights; random gates differ
	fz = joint_spatial.JointSpatialDownstream(f'{out}/joint_zero_gate.pth', 'fusion').to(dev).eval()
	fr = joint_spatial.JointSpatialDownstream(f'{out}/joint_final.pth', 'fusion').to(dev).eval()
	d0, d1 = float((fz.embed(reim) - e).abs().max()), float((fr.embed(reim) - e).abs().max())
	print(f'[check] {name}: fusion zero-gate vs concat {d0:.2e}; random gate vs concat {d1:.2e}')
	assert d0 < 1e-3 and d1 > 1e-2

# frozen: only the head trains
n_train = sum(p.numel() for p in fr.parameters() if p.requires_grad)
print(f'[check] trainable params (head only): {n_train}')
assert n_train == 2 * 768 * 2 + 2 * 768 + 1
print('[check] all OK')
