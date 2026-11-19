"""Diagnose Pilot A encoders that probe at or below random (2026-11-19).

On 64 'preval' clips:
  1. latest.pth in JointPretrain: masked MPC acc / MPG as logged (are the weights the trained ones?).
  2. Same model, no mask: tokens vs the downstream path (wrapper / AMBAModel._encode_with_mamba on the saved
     *_encoder_final.pth): max |diff| (does downstream read the same network?).
  3. Collapse stats of clip embeddings (mean over tokens) after pair 3/6/9/12 (= layers 6/12/18/24) and after the
     final norm, unmasked and with a training-style column mask: mean off-diagonal cosine between clips, per-dim std
     across clips, NaN fraction, token norm. References: old ipd16 no-SAC encoder (random-patch masking), random init.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_pretrain as jp  # noqa: E402
import ssamba_spatial  # noqa: E402
from ssamba_spatial_pretrain import TarMicSigDataset, stft_reim  # noqa: E402

EXP = '/scratch200/yotams2/SAC-SAR-SSL/exp'
TAR = '/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_pretrain.tar'
dev = 'cuda'
torch.manual_seed(0)

ds_val = TarMicSigDataset(TAR, 'preval', 64)
sig = torch.stack([torch.as_tensor(ds_val[i]).float() for i in range(len(ds_val))]).to(dev)
stft_mod = jp.at_module.STFT(win_len=jp.WIN_LEN, win_shift_ratio=jp.WIN_SHIFT_RATIO, nfft=jp.NFFT).to(dev)
stft_n = jp.shared_stft(stft_mod, sig)


def stats(tag, h):
	""" h: (B, 1+N, D) tokens. """
	h = h.float()
	nan = torch.isnan(h).float().mean().item()
	e = h[:, 1:].mean(1)  # (B, D) clip embedding
	en = F.normalize(e, dim=-1)
	cos = en @ en.T
	B = cos.shape[0]
	off = (cos.sum() - cos.diagonal().sum()) / (B * (B - 1))
	print(f'  {tag:34s} nan {nan:.3f} | token norm {h[:, 1:].norm(dim=-1).mean():8.2f} | clip cos {off:.4f} | '
		  f'dim std across clips {e.std(0).mean():.4f} | within-clip token std {h[:, 1:].std(1).mean():.4f}', flush=True)


@torch.no_grad()
def layer_stats(name, stream, Xn, cmask):
	h, r = stream.embed(Xn, cmask), None
	for i in range(stream.n_pairs):
		h, r = stream.pair(i, h, r)
		if (i + 1) in (3, 6, 9, 12):
			stats(f'{name} layer {2 * (i + 1)} (h+r)', h + r)
	stats(f'{name} final', stream.final(h, r))


for kind in ('spat', 'spec'):
	print(f'\n===== Pilot A {kind}', flush=True)
	m = jp.JointPretrain(kind, False, 0.3, 7, 10.0, 1.0, use_checkpointing=False).to(dev)
	st = torch.load(f'{EXP}/joint_pretrain/pilotA_{kind}/latest.pth', map_location='cpu', weights_only=False)
	m.load_state_dict(st['model'])
	m.eval()
	s = m.streams[kind]
	print(f'  in_mean {s.in_mean.flatten().tolist()} in_std {s.in_std.flatten().tolist()}')
	with torch.no_grad():
		torch.manual_seed(1234); jp.random.seed(1234)
		out = m(stft_n, fusion=False)
	print('  1. masked losses (val log ep10: spat mpc acc 0.60 mpg 0.79; spec acc 0.64 mpg 0.30):',
		  {k: round(float(v), 4) for k, v in out.items()})
	Xn = s.norm_input(m.raw_inputs(stft_n)[kind])
	with torch.no_grad():
		ref = m.encode({kind: Xn}, None, fusion=False)[kind]
		if kind == 'spat':
			w = ssamba_spatial.SSAMBASpatialDownstream(ckpt_path=f'{EXP}/joint_pretrain/pilotA_spat/spat_encoder_final.pth',
													   frozen=True, fill='none', input_mode='ipd', n_stft_frames=jp.NT).to(dev).eval()
			_, reim = stft_reim(stft_mod, sig)
			Xw, _ = w.stft_frontend(reim)
			print(f'  wrapper input vs pretraining input max|diff| {(Xw - Xn).abs().max().item():.3e}')
			hw = w.encoder._encode_with_mamba(Xw)
		else:
			enc = jp.build_spectral_encoder(False).to(dev).eval()
			ck = torch.load(f'{EXP}/joint_pretrain/pilotA_spec/spec_encoder_final.pth', map_location='cpu', weights_only=False)
			enc.load_state_dict(ck['state_dict'], strict=True)
			print(f"  saved meta stats {ck['meta']['in_mean']:.4f} / {ck['meta']['in_std']:.4f}")
			hw = enc._encode_with_mamba(Xn)
	print(f'  2. downstream path vs pretraining encode (no mask): max|diff| {(hw - ref).abs().max().item():.3e} '
		  f'(ref max |h| {ref.abs().max().item():.2f})')
	print('  3. collapse stats')
	layer_stats(f'A-{kind} nomask', s, Xn, None)
	cm = jp.column_mask(Xn.shape[0], 7, device=dev)
	layer_stats(f'A-{kind} colmask7', s, Xn, cm)
	if kind == 'spat':
		for label, ck in (('ipd16 nosac final', f'{EXP}/ssamba_pretrain/ipd16_nosac/encoder_final.pth'),
						  ('random', None)):
			r = jp.JointPretrain('spat', False, 0.3, 7, 10.0, 1.0, use_checkpointing=False).to(dev).eval()
			rs = r.streams['spat']
			if ck is not None:
				w = ssamba_spatial.SSAMBASpatialDownstream(ckpt_path=ck, frozen=True, fill='none', input_mode='ipd',
														   n_stft_frames=jp.NT).to(dev).eval()
				rs.enc = w.encoder
				rs.in_mean.copy_(w.in_mean.view_as(rs.in_mean)); rs.in_std.copy_(w.in_std.view_as(rs.in_std))
			else:
				rs.in_mean.copy_(s.in_mean); rs.in_std.copy_(s.in_std)
			layer_stats(f'{label} nomask', rs, rs.norm_input(r.raw_inputs(stft_n)['spat']), None)
print('\ndone')
