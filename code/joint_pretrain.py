"""
	Joint spectral + spatial SSAMBA pretraining (sprint 2026-11-16 -> 11-30, pilots A and B; design agreed with the
	user and the cloud Claude on 2026-11-17, Notion "Agent Handoff").

	Two SSAMBA encoders (24 layers, d=768, linear 16x16 patch embeddings, no stems) on ONE shared STFT, so both token
	grids are 16 time columns of 256 ms per 4.112-s clip:
		spectral: 128-mel log-mel of the mid channel (L+R)/2, from the SAR-SSL STFT (Hann 512, hop 256, divided by the
		          mean |STFT| of mic 0), all 257 bins, scalar z-norm  -> 1 x 128 x 256 -> 8 x 16 = 128 tokens;
		spatial:  ipd16 unchanged (log|L|, ILD, cos IPD, sin IPD; bins 1..256), per-channel z-norm -> 4 x 256 x 256
		          -> 16 x 16 = 256 tokens (exactly ssamba_spatial_pretrain.py's input).
	Masking: whole time columns, the SAME columns in both streams (one mask per clip), exactly --mask-cols of 16, as
	spans of 1/2/3 columns (p 0.3/0.4/0.3) with >= 1 visible column between spans. Masked tokens -> mask embedding
	after the patch embedding (as SSAMBA). As in the ipd16 recipe, MPG and MPC each run their own forward pass with
	their own mask draw.
	Objective per stream: recon_lambda * MPG + classif_lambda * MPC on its own masked tokens (SSAMBA's heads, incl.
	the extra norm_f before the heads). No SAC, no stream dropout.
	Modes:
		--streams spec | spat   Pilot A halves: one encoder alone.
		--streams joint         both encoders in one model; with --fusion (Pilot B): after layers 6, 12, 18 (bidirectional
		                        pair blocks 3, 6, 9), for each column t, mean-pool the other stream's tokens at t
		                        (state = hidden + residual) -> LayerNorm -> Linear(768, 768) -> per-channel gate (zero init)
		                        -> added to every token of this stream at t. Per-direction fusion dropout per sample
		                        (--fusion-drop, 0.3): a dropped direction adds nothing at any fusion layer.
	Validation: fusion on, and for --fusion also fusion off (each encoder alone).

	Outputs in --exp-dir: latest.pth (resume), log.jsonl, args.json, and per encoder
		spat_encoder_{epochN,best,final}.pth  same format as ssamba_spatial_pretrain.py (loads in ssamba_spatial.py),
		spec_encoder_{epochN,best,final}.pth  {'state_dict': AMBAModel weights, 'meta': log-mel settings + stats},
		joint_{best,final}.pth               (joint) both encoders + fusion, for fusion-on evaluation.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.utils_module as at_module  # noqa: E402
import ssamba_spatial  # noqa: E402
from ssamba_spatial_pretrain import (FS, NFFT, NT, T_SEC, WIN_LEN, WIN_SHIFT_RATIO, FRE_RANGE,  # noqa: E402
									 TarMicSigDataset, save_atomic)

N_MELS = 128
N_COLS = NT // 16  # 16 time columns of 16 frames = 256 ms
FUSION_AFTER_PAIRS = (3, 6, 9)  # after layers 6, 12, 18 of 24 (Vim layers run as forward/backward pairs)


# ---------------------------------------------------------------- inputs
def shared_stft(stft_mod, sig, eps=1e-6):
	""" (nb, nsample, 2) -> normalised complex STFT (nb, 2, 257, nt), as STFTLearner.data_preprocess (/ mean |L|). """
	stft = stft_mod(signal=sig).permute(0, 3, 1, 2)
	mean_value = stft[:, 0].abs().reshape(stft.shape[0], -1).mean(dim=1)[:, None, None, None]
	return stft / (mean_value + eps)


def fit_frames(X, n=NT):
	return F.pad(X, (0, n - X.shape[-1])) if X.shape[-1] < n else X[..., :n]


class Inputs(nn.Module):
	""" Raw (unstandardised) stream inputs from the normalised STFT. """
	def __init__(self, spat_frontend):
		super().__init__()
		import torchaudio
		fb = torchaudio.functional.melscale_fbanks(NFFT // 2 + 1, 0.0, FS / 2, N_MELS, FS, norm=None, mel_scale='htk')
		self.register_buffer('mel_fb', fb)  # (257, 128)
		self.spat_frontend = spat_frontend  # SSAMBASpatialDownstream.stft_frontend (its in_mean is None: raw)

	def spec(self, stft_n, eps=1e-6):
		mid = 0.5 * (stft_n[:, 0] + stft_n[:, 1])  # (nb, 257, nt)
		mel = torch.einsum('bft,fm->bmt', mid.abs() ** 2, self.mel_fb)
		return fit_frames(torch.log(mel + eps)[:, None])  # (nb, 1, 128, 256)

	def spat(self, stft_n):
		reim = torch.view_as_real(stft_n)[:, :, FRE_RANGE]  # (nb, 2, 256, nt, 2), as stft_reim
		X, _ = self.spat_frontend(reim)
		return fit_frames(X)  # (nb, 4, 256, 256)


# ---------------------------------------------------------------- masking
def column_mask(nb, n_mask, n_cols=N_COLS, span_p=(0.3, 0.4, 0.3), device='cpu'):
	""" (nb, n_cols) bool, True = masked. Exactly n_mask columns per clip, spans of 1/2/3 (span_p), >= 1 visible
		column between spans, gaps placed uniformly (stars and bars). New draw per clip (Python RNG, seeded). """
	out = torch.zeros(nb, n_cols, dtype=torch.bool)
	for b in range(nb):
		spans, left = [], n_mask
		while left > 0:
			s = min(random.choices((1, 2, 3), weights=span_p)[0], left)
			spans.append(s); left -= s
		k = len(spans)
		free = n_cols - n_mask - (k - 1)  # visible columns beyond the mandatory 1 between spans
		assert free >= 0, f'{n_mask} masked columns in {k} spans do not fit in {n_cols}'
		cuts = sorted(random.randint(0, free) for _ in range(k))  # k cut points -> k + 1 extra-gap sizes
		extra = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, k)] + [free - cuts[-1]]
		pos = extra[0]
		for i, s in enumerate(spans):
			out[b, pos:pos + s] = True
			pos += s + 1 + extra[i + 1]
	return out.to(device)


# ---------------------------------------------------------------- model
def build_spectral_encoder(use_checkpointing):
	if ssamba_spatial.SAC_SSAMBA_SRC not in sys.path:
		sys.path.append(ssamba_spatial.SAC_SSAMBA_SRC)
	from models.both_models import AMBAModel
	cfg = {} if torch.cuda.is_available() else {'fused_add_norm': False}
	enc = AMBAModel(fshape=16, tshape=16, fstride=16, tstride=16, input_fdim=N_MELS, input_tdim=NT,
					model_size='base', in_chans=1, pretrain_stage=True, vision_mamba_config=cfg)
	enc.use_checkpointing = enc.v.use_checkpointing = use_checkpointing
	return enc


class Stream(nn.Module):
	""" One SSAMBA encoder run layer pair by layer pair, so fusion can act between pairs. """
	def __init__(self, enc, n_chans, n_rows):
		super().__init__()
		self.enc, self.n_rows = enc, n_rows
		self.register_buffer('in_mean', torch.zeros(1, n_chans, 1, 1))
		self.register_buffer('in_std', torch.ones(1, n_chans, 1, 1))

	def norm_input(self, X):
		return (X - self.in_mean) / self.in_std

	def embed(self, X, cmask):
		v = self.enc.v
		x = v.patch_embed(X)  # (B, rows*cols, D), frequency-major
		if cmask is not None:
			m = cmask[:, None, :].expand(-1, self.n_rows, -1).reshape(x.shape[0], -1, 1)
			x = torch.where(m, self.enc.mask_embed.expand_as(x), x)
		x = torch.cat((v.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + v.pos_embed
		return v.pos_drop(x)

	def pair(self, i, h, r):
		""" Layers 2i and 2i+1, exactly as AMBAModel._forward_mamba_layers: one after the other for SSAMBA's Vim
			config (if_bidirectional False; each bimamba v2 layer is bidirectional by itself), the flip-pair form
			only when if_bidirectional is set. (Pilots A/B before 2026-11-19 always used the flip-pair form, so their
			encoders did not match the downstream forward.) """
		L = self.enc.v.layers

		if self.enc.v.if_bidirectional:
			def f(h, r):
				hf, rf = L[2 * i](h, r)
				hb, rb = L[2 * i + 1](h.flip([1]), None if r is None else r.flip([1]))
				return hf + hb.flip([1]), rf + rb.flip([1])
		else:
			def f(h, r):
				h, r = L[2 * i](h, r)
				return L[2 * i + 1](h, r)
		if self.enc.v.use_checkpointing and self.training:
			return torch.utils.checkpoint.checkpoint(f, h, r, use_reentrant=False)
		return f(h, r)

	@property
	def n_pairs(self):
		return len(self.enc.v.layers) // 2

	def final(self, h, r):
		""" _encode_with_mamba's final norm. """
		v = self.enc.v
		if not v.fused_add_norm:
			r = h if r is None else r + v.drop_path(h)
			return v.norm_f(r.to(dtype=v.norm_f.weight.dtype))
		# both_models put Vim's own mamba_ssm (mamba-1p1p1: ops/triton/layer_norm.py) first on sys.path; take its
		# imports from there. If unavailable, the unfused form, as _encode_with_mamba's fallback.
		from models.both_models import rms_norm_fn, layer_norm_fn, RMSNorm
		if layer_norm_fn is None:
			return v.norm_f(r + h)
		fn = rms_norm_fn if (RMSNorm is not None and isinstance(v.norm_f, RMSNorm)) else layer_norm_fn
		return fn(v.drop_path(h), v.norm_f.weight, v.norm_f.bias, eps=v.norm_f.eps, residual=r, prenorm=False,
				  residual_in_fp32=v.residual_in_fp32)

	def losses(self, Xn, out_final, cmask, kind):
		""" Xn: standardised input; out_final: final-normed tokens (B, 1+N, D). SSAMBA's mpg / mpc on masked tokens,
			including the second norm_f that AMBAModel.mpg/mpc apply before the heads. """
		enc = self.enc
		x = enc.v.norm_f(out_final)[:, 1:]
		patches = enc.unfold(Xn).transpose(1, 2)  # (B, N, P), same order as patch_embed
		m = cmask[:, None, :].expand(-1, self.n_rows, -1).reshape(x.shape[0], -1)  # (B, N)
		M = int(m[0].sum())
		idx = m.float().argsort(dim=1, descending=True, stable=True)[:, :M]  # masked positions
		h = torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
		tgt = torch.gather(patches, 1, idx[..., None].expand(-1, -1, patches.shape[-1])).detach()
		if kind == 'mpg':
			return {'mpg': torch.mean((enc.gpredlayer(h).float() - tgt.float()) ** 2)}
		pred = enc.cpredlayer(h).float()
		with torch.autocast(device_type=pred.device.type, enabled=False):
			total = torch.bmm(tgt.float(), pred.transpose(1, 2))  # (B, M, M): rows targets, cols predictions
		ar = torch.arange(M, device=total.device)
		acc = (total.softmax(-1).argmax(dim=1) == ar).float().mean()
		nce = -total.log_softmax(-1).diagonal(dim1=1, dim2=2).mean()
		return {'mpc': nce, 'acc': acc}


class FusionDir(nn.Module):
	""" src -> dst at every time column: mean over src rows -> LN -> Linear -> gate (zero init). """
	def __init__(self, D):
		super().__init__()
		self.ln, self.lin = nn.LayerNorm(D), nn.Linear(D, D)
		self.gate = nn.Parameter(torch.zeros(D))

	def forward(self, src_state, src_rows, dst_rows, keep):
		B, _, D = src_state.shape
		pooled = src_state[:, 1:].reshape(B, src_rows, N_COLS, D).mean(1)  # (B, cols, D)
		delta = self.gate * self.lin(self.ln(pooled)) * keep[:, None, None]
		delta = delta[:, None].expand(-1, dst_rows, -1, -1).reshape(B, dst_rows * N_COLS, D)
		return F.pad(delta, (0, 0, 1, 0))  # nothing added to the CLS token


class JointPretrain(nn.Module):
	def __init__(self, streams, fusion, fusion_drop, mask_cols, recon_lambda, classif_lambda, use_checkpointing=True):
		super().__init__()
		self.use_spec, self.use_spat = streams in ('spec', 'joint'), streams in ('spat', 'joint')
		self.fusion_on = fusion
		assert not fusion or streams == 'joint', '--fusion needs --streams joint'
		self.fusion_drop, self.mask_cols = fusion_drop, mask_cols
		self.recon_lambda, self.classif_lambda = recon_lambda, classif_lambda
		ds = ssamba_spatial.SSAMBASpatialDownstream(ckpt_path=None, fill='none', input_mode='ipd', patch='16x16',
													n_stft_frames=NT, clip_seconds=T_SEC, use_checkpointing=use_checkpointing)
		del ds.mlp_head
		self.inputs = Inputs(ds.stft_frontend)
		self.streams = nn.ModuleDict()
		if self.use_spat:
			self.streams['spat'] = Stream(ds.encoder, 4, 16)
		if self.use_spec:
			self.streams['spec'] = Stream(build_spectral_encoder(use_checkpointing), 1, N_MELS // 16)
		for s in self.streams.values():
			for name, p in s.enc.named_parameters():
				if name.split('.')[0] in ['cpredlayer', 'gpredlayer', 'mask_embed']:
					p.requires_grad = True  # the spatial wrapper freezes the pretraining heads
		D = 768
		self.fuse = nn.ModuleDict({'spec2spat': FusionDir(D), 'spat2spec': FusionDir(D)}) if fusion else None

	def raw_inputs(self, stft_n):
		out = {}
		if self.use_spec:
			out['spec'] = self.inputs.spec(stft_n)
		if self.use_spat:
			out['spat'] = self.inputs.spat(stft_n)
		return out

	def encode(self, Xn, cmask, fusion=True):
		""" Xn: {stream: standardised input}. Returns {stream: final-normed tokens (B, 1+N, D)}. """
		names = list(Xn)
		hs = {k: self.streams[k].embed(Xn[k], cmask) for k in names}
		rs = {k: None for k in names}
		use_f = fusion and self.fuse is not None
		if use_f:
			B = next(iter(hs.values())).shape[0]
			dev = next(iter(hs.values())).device
			if self.training and self.fusion_drop > 0:
				keep = {d: (torch.rand(B, device=dev) >= self.fusion_drop).float() for d in self.fuse}
			else:
				keep = {d: torch.ones(B, device=dev) for d in self.fuse}
		for i in range(self.streams[names[0]].n_pairs):
			for k in names:
				hs[k], rs[k] = self.streams[k].pair(i, hs[k], rs[k])
			if use_f and (i + 1) in FUSION_AFTER_PAIRS:
				st = {k: hs[k] + rs[k] for k in names}  # stream state seen by the next layer's residual add
				rows = {k: self.streams[k].n_rows for k in names}
				d_spat = self.fuse['spec2spat'](st['spec'], rows['spec'], rows['spat'], keep['spec2spat'])
				d_spec = self.fuse['spat2spec'](st['spat'], rows['spat'], rows['spec'], keep['spat2spec'])
				hs['spat'] = hs['spat'] + d_spat.to(hs['spat'].dtype)
				hs['spec'] = hs['spec'] + d_spec.to(hs['spec'].dtype)
		return {k: self.streams[k].final(hs[k], rs[k]) for k in names}

	def forward(self, stft_n, fusion=True):
		X = self.raw_inputs(stft_n)
		Xn = {k: self.streams[k].norm_input(v) for k, v in X.items()}
		B = next(iter(Xn.values())).shape[0]
		dev = next(iter(Xn.values())).device
		out, loss = {}, 0.0
		for kind, lam in (('mpg', self.recon_lambda), ('mpc', self.classif_lambda)):
			if lam <= 0:
				continue
			cmask = column_mask(B, self.mask_cols, device=dev)  # one draw per pass, shared by both streams
			enc = self.encode(Xn, cmask, fusion=fusion)
			for k in Xn:
				res = self.streams[k].losses(Xn[k], enc[k], cmask, kind)
				for name, v in res.items():
					out[f'{k}_{name}'] = v
				loss = loss + lam * res[kind]
		out['loss'] = loss
		return out

	# ------------------------------------------------ checkpoints
	def spat_checkpoint(self, args):
		s = self.streams['spat']
		sd = {ssamba_spatial.SPATIAL_PREFIX + k: v.detach().cpu() for k, v in s.enc.state_dict().items()}
		meta = {'input': 'ipd', 'patch': '16x16', 'n_frames': NT, 'n_freq': 256,
				'in_mean': s.in_mean.flatten().tolist(), 'in_std': s.in_std.flatten().tolist(),
				'masking': f'columns {self.mask_cols}/{N_COLS}', 'trained_with': args.streams + ('+fusion' if args.fusion else ''),
				'args': vars(args)}
		return {'state_dict': sd, 'meta': meta}

	def spec_checkpoint(self, args):
		s = self.streams['spec']
		meta = {'input': 'logmel_mid_sarssl_stft', 'n_mels': N_MELS, 'n_fft': NFFT, 'hop': int(WIN_LEN * WIN_SHIFT_RATIO),
				'window': 'hann', 'stft_norm': 'mean |STFT| of mic 0', 'mel': 'htk, norm None, 0-8000 Hz, all 257 bins',
				'log_eps': 1e-6, 'patch': '16x16', 'n_frames': NT, 'fs': FS,
				'in_mean': float(s.in_mean.flatten()[0]), 'in_std': float(s.in_std.flatten()[0]),
				'masking': f'columns {self.mask_cols}/{N_COLS}', 'trained_with': args.streams + ('+fusion' if args.fusion else ''),
				'args': vars(args)}
		return {'state_dict': {k: v.detach().cpu() for k, v in s.enc.state_dict().items()}, 'meta': meta}

	def save_encoders(self, args, tag):
		if self.use_spat:
			save_atomic(self.spat_checkpoint(args), os.path.join(args.exp_dir, f'spat_encoder_{tag}.pth'))
		if self.use_spec:
			save_atomic(self.spec_checkpoint(args), os.path.join(args.exp_dir, f'spec_encoder_{tag}.pth'))
		if self.use_spat and self.use_spec and tag in ('best', 'final'):
			save_atomic({'model': {k: v.detach().cpu() for k, v in self.state_dict().items()}, 'args': vars(args)},
						os.path.join(args.exp_dir, f'joint_{tag}.pth'))


# ---------------------------------------------------------------- stats
@torch.no_grad()
def input_stats(model, stft_mod, loader, device, nbatch=50):
	acc = {}
	for i, sig in enumerate(loader):
		if i >= nbatch:
			break
		X = model.raw_inputs(shared_stft(stft_mod, sig.to(device)))
		for k, v in X.items():
			v = v.double()
			s, s2, n = acc.get(k, (0, 0, 0))
			acc[k] = (s + v.sum(dim=(0, 2, 3)), s2 + (v ** 2).sum(dim=(0, 2, 3)), n + v[:, 0].numel())
	out = {}
	for k, (s, s2, n) in acc.items():
		mean = s / n
		out[k] = (mean.float(), (s2 / n - mean ** 2).clamp_min(1e-12).sqrt().float())
	return out


# ---------------------------------------------------------------- train
@torch.no_grad()
def evaluate(model, stft_mod, loader, device, fusion=True, seed=1234):
	model.eval()
	rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
		   np.random.get_state(), random.getstate())
	torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)  # same masks every evaluation
	tot, n = {}, 0
	for sig in loader:
		out = model(shared_stft(stft_mod, sig.to(device)), fusion=fusion)
		for k, v in out.items():
			tot[k] = tot.get(k, 0.0) + float(v) * sig.shape[0]
		n += sig.shape[0]
	torch.set_rng_state(rng[0])
	if torch.cuda.is_available():
		torch.cuda.set_rng_state_all(rng[1])
	np.random.set_state(rng[2]); random.setstate(rng[3])
	model.train()
	return {k: v / n for k, v in tot.items()}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--tar', required=True)
	ap.add_argument('--exp-dir', required=True)
	ap.add_argument('--streams', choices=['spec', 'spat', 'joint'], required=True)
	ap.add_argument('--fusion', action='store_true', help='Pilot B: time-column fusion (needs --streams joint)')
	ap.add_argument('--fusion-drop', type=float, default=0.3)
	ap.add_argument('--mask-cols', type=int, default=7, help='masked columns of 16 (fallback 6)')
	ap.add_argument('--num-train', type=int, default=50000, help='50000 = the ipd16 runs (masking ablation)')
	ap.add_argument('--num-val', type=int, default=None)
	ap.add_argument('--val-from-train', type=int, default=None, help='tar without a preval split: val = pretrain clips from this index on (paired set: 48000 = its last 500 scenes, as swap_pretrain)')
	ap.add_argument('--bs', type=int, default=128)
	ap.add_argument('--lr', type=float, default=4e-4)
	ap.add_argument('--nepoch', type=int, default=10)
	ap.add_argument('--recon-lambda', type=float, default=10.0)
	ap.add_argument('--classif-lambda', type=float, default=1.0)
	ap.add_argument('--lr-patience', type=int, default=2)
	ap.add_argument('--seed', type=int, default=1)
	ap.add_argument('--ckpt-every', type=int, default=200)
	ap.add_argument('--log-every', type=int, default=20)
	ap.add_argument('--max-steps', type=int, default=None, help='stop after this many steps (smoke tests)')
	ap.add_argument('--num-workers', type=int, default=4)
	ap.add_argument('--snapshot-epochs', type=int, nargs='*', default=[2, 5])
	ap.add_argument('--device', default='cuda')
	args = ap.parse_args()
	os.makedirs(args.exp_dir, exist_ok=True)
	device = args.device
	torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

	ds_train = TarMicSigDataset(args.tar, 'pretrain', args.num_train)
	if args.val_from_train is None:
		ds_val = TarMicSigDataset(args.tar, 'preval', args.num_val)
	else:  # tar with one split (the paired set): val = pretrain clips [val_from_train:], disjoint from --num-train
		assert args.num_train <= args.val_from_train, 'train and val would overlap'
		ds_val = TarMicSigDataset(args.tar, 'pretrain')
		ds_val.items = ds_val.items[args.val_from_train:][:args.num_val]
	print(f'[data] train {len(ds_train)}, val {len(ds_val)}', flush=True)
	kw = dict(num_workers=args.num_workers, pin_memory=device == 'cuda', persistent_workers=args.num_workers > 0)
	val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.bs, shuffle=False, **kw)
	steps_per_epoch = len(ds_train) // args.bs

	stft_mod = at_module.STFT(win_len=WIN_LEN, win_shift_ratio=WIN_SHIFT_RATIO, nfft=NFFT).to(device)
	model = JointPretrain(args.streams, args.fusion, args.fusion_drop, args.mask_cols,
						  args.recon_lambda, args.classif_lambda).to(device)
	n_par = {k: sum(p.numel() for p in s.parameters()) / 1e6 for k, s in model.streams.items()}
	if model.fuse is not None:
		n_par['fusion'] = sum(p.numel() for p in model.fuse.parameters()) / 1e6
	print(f'[model] streams {list(model.streams)}, fusion {args.fusion} (drop {args.fusion_drop}), mask {args.mask_cols}/{N_COLS} '
		  f'columns, params (M) { {k: round(v, 1) for k, v in n_par.items()} }', flush=True)
	opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], args.lr, weight_decay=5e-8, betas=(0.95, 0.999))
	sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=args.lr_patience)

	latest = os.path.join(args.exp_dir, 'latest.pth')
	step, best_val = 0, float('inf')
	if os.path.isfile(latest):
		st = torch.load(latest, map_location='cpu', weights_only=False)
		model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
		step, best_val = st['step'], st['best_val']
		torch.set_rng_state(st['rng'][0]); np.random.set_state(st['rng'][1]); random.setstate(st['rng'][2])
		print(f'[resume] step {step} = epoch {step // steps_per_epoch}, batch {step % steps_per_epoch}/{steps_per_epoch}', flush=True)
	else:
		stat_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.bs, shuffle=True,
												  generator=torch.Generator().manual_seed(args.seed), num_workers=args.num_workers)
		for k, (mean, std) in input_stats(model, stft_mod, stat_loader, device).items():
			s = model.streams[k]
			if k == 'spec':  # one scalar, as mono SSAMBA's dataset_mean / dataset_std
				mean, std = mean.mean().view(1), std.mean().view(1)  # single channel: these are the channel stats
			s.in_mean.copy_(mean.view_as(s.in_mean)); s.in_std.copy_(std.view_as(s.in_std))
			print(f'[stats] {k} input mean {mean.tolist()} std {std.tolist()}', flush=True)
	with open(os.path.join(args.exp_dir, 'args.json'), 'w') as f:
		json.dump(vars(args), f, indent=1)

	def save_latest():
		save_atomic({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
					 'step': step, 'best_val': best_val,
					 'rng': (torch.get_rng_state(), np.random.get_state(), random.getstate())}, latest)

	logf = open(os.path.join(args.exp_dir, 'log.jsonl'), 'a')
	total_steps = steps_per_epoch * args.nepoch
	if args.max_steps:
		total_steps = min(total_steps, args.max_steps)
	model.train()
	t0, acc = time.time(), {}
	while step < total_steps:
		epoch, start_b = step // steps_per_epoch, step % steps_per_epoch
		perm = torch.randperm(len(ds_train), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch))
		idx = perm[start_b * args.bs: steps_per_epoch * args.bs].tolist()
		loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_train, idx), batch_size=args.bs, shuffle=False,
											 drop_last=True, num_workers=args.num_workers, pin_memory=device == 'cuda')
		for sig in loader:
			out = model(shared_stft(stft_mod, sig.to(device, non_blocking=True)))
			if not torch.isfinite(out['loss']):
				raise RuntimeError(f'non-finite loss at step {step}: { {k: float(v) for k, v in out.items()} }')
			opt.zero_grad(set_to_none=True)
			out['loss'].backward()
			opt.step()
			step += 1
			for k, v in out.items():
				acc[k] = acc.get(k, 0.0) + float(v)
			if step % args.log_every == 0:
				rec = {'step': step, 'epoch': step / steps_per_epoch, 'lr': opt.param_groups[0]['lr'],
					   's_per_step': (time.time() - t0) / args.log_every, **{k: v / args.log_every for k, v in acc.items()}}
				if model.fuse is not None:
					rec['gate_abs'] = {d: float(m.gate.abs().mean()) for d, m in model.fuse.items()}
				print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + '\n'); logf.flush()
				acc, t0 = {}, time.time()
			if step % steps_per_epoch == 0 or step == total_steps:
				val = evaluate(model, stft_mod, val_loader, device, fusion=True)
				rec = {'step': step, 'epoch': step / steps_per_epoch, 'val': val}
				if model.fuse is not None:
					rec['val_fusion_off'] = evaluate(model, stft_mod, val_loader, device, fusion=False)
				print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + '\n'); logf.flush()
				sched.step(val['loss'])
				ep = step // steps_per_epoch
				if step % steps_per_epoch == 0 and ep in args.snapshot_epochs:
					model.save_encoders(args, f'epoch{ep}')
				if val['loss'] < best_val:
					best_val = val['loss']
					model.save_encoders(args, 'best')
				save_latest()
			elif step % args.ckpt_every == 0:
				save_latest()
			if step >= total_steps:
				break
	model.save_encoders(args, 'final')
	save_latest()
	print(f'[done] {step} steps, best val loss {best_val:.4f}', flush=True)


if __name__ == '__main__':
	main()
