""" V1 room teacher (Agent Handoff 23 Nov; V1a / V1b): a two-channel RIR encoder, self-supervised masked reconstruction.

Data: the saved RIRs of the data-v2 renders, MicSig/simu_paired_pretrain_v2.tar.rirs.npz (12,500 scenes) and
simu_unpaired8_pretrain.tar.rirs.npz (25,000), rir (S, 2, 13868) fp16 peak-normalised (0.87 s at 16 kHz).
Held out (never trained on; the probes' test set): paired scenes [12000, 12500) (= the swap runs' validation scenes) and
unpaired scenes [22500, 25000): 3,000 rooms.
Input (phase kept, so inter-channel delay / TDOA survives): STFT per channel (Hann 512, hop 128, no centring, zero-padded
to 112 frames, Nyquist bin dropped -> 256 bins), power-law compressed complex X |X|^-0.7 (magnitude^0.3, phase kept),
real + imaginary of both channels = 4 maps (256 x 112), one global scale (std over 2,048 training RIRs).
Model (MAE, He et al. 2022): patches 16 bins x 4 frames x 4 maps (256 values) -> 16 x 28 = 448 tokens; encoder
ViT d 384, 8 layers, 6 heads, pre-norm, learned positions; 75% of tokens masked at random; decoder d 256, 2 layers;
loss MSE on the masked patches (unnormalised targets: per-patch normalisation would erase the decay / level cues that
carry T60 and DRR). Augmentation: channel swap with p 0.5 (flips the TDOA sign; a label-free symmetry).
Embedding (what V1 uses): the encoder on the full unmasked input, final LayerNorm, mean over the 448 tokens -> 384-d.
Probes (the ceiling for the spatial student): closed-form ridge on the frozen z-scored embeddings, penalty by 5-fold CV
on the training rooms, test MAE on the 3,000 held-out rooms for TDOA (samples), T60 (EDC, s), DRR (dB), C50 (dB);
reference rows: predict-the-mean, the same encoder at random init, and TDOA from the cross-correlation peak of the two
direct-path RIRs (the label's own definition, so it shows the label precision).
Result of this design (--arch v1, 11-23, exp/rir_encoder/rirmae_v2): FAILED as a teacher. Val MSE 0.98 -> 0.96 in 100
epochs; probes TDOA 2.87 (= mean predictor), T60 0.034 (random init 0.015), DRR 1.14, C50 0.43. Causes: most masked
patches are diffuse late tail (a random draw: unpredictable), and each patch held both channels, so masking never asked
for the inter-channel relation (TDOA / level difference).
--arch v2 (the handoff's other option: early RIR + coarse late-tail envelope, per-channel tokens):
  early: first 256 ms (4096 samples), STFT Hann 256 / hop 64 (16 / 4 ms), 128 bins x 64 frames, compressed complex re + im
         per channel; patches 16 bins x 4 frames x (re, im) = 128 values; 8 x 16 = 128 tokens PER CHANNEL;
  late:  whole RIR, STFT 512 / hop 256 (16 ms), power summed into 16 log-spaced bands, log10, 56 frames; patches
         16 bands x 4 frames = 64 values; 14 tokens per channel (a smooth, predictable decay envelope);
  284 tokens with learned positions (channel identity included), separate patch embeddings and output heads, one
  encoder / decoder; 75% of tokens masked at random (one channel's patch can be hidden while the other's is visible);
  loss = mean of the early and late masked MSEs. Same encoder size, embedding (mean over tokens), probes and outputs.
v2 result (11-23): TDOA still at the mean predictor (2.86); T60 0.0128 / DRR 1.03 / C50 0.32 (10-20% better than random init).
--arch a1 (your decision 11-23, "A' design 1"): fix the objective, the input and the pooling.
  input:  early 256 ms as the student's ipd16 (Hann 512 / hop 256, no centring, bins 1..256, / mean |L|; log|L|, ILD,
          cos IPD, sin IPD; per-feature z-norm), patches 16 bins x 4 frames = 64 tokens of 256 values (L part = log|L|,
          64 values; R part = ILD, cos, sin, 192 values) + v2's late log-energy envelope tokens (2 ch x 14).
  pooling (the teacher embedding): concat[attention pool over the early tokens (one learned query, 1 head), mean over
          the late tokens] = 768-d; plain mean over all tokens also reported.
  objective, per step 50 / 50, equal weights:
    mae:   v2's random masking (75% of the 92 tokens), reconstruct all masked tokens (keeps T60 / DRR / C50);
    cross: the encoder sees everything; a small decoder gets, per early token, ONLY the left features (log|L|) plus
           the pooled embedding, and must rebuild the right-dependent features (ILD, cos / sin IPD; the IPD terms
           weighted by sqrt(|L||R|) as in the swap loss). Given the left channel, the missing information is the
           inter-channel relation (TDOA, ILD), and its only route is the pooled embedding V1 uses. (As literally
           specified, with all right-channel information also hidden from the ENCODER, nothing in the encoder's input
           would carry the inter-channel phase, so it could not be learned; hence the bottleneck reading.)
  extra report: Spearman between embedding cosine distance (z-scored) and oracle-B Euclidean distance on 10k random
  held-out room pairs (ceiling dir from rir_ceiling.py).
--arch a1u (A′-uniform ablation, Yotam 11-25: is A′'s early / late split + late envelope needed?): A′'s ipd16 features and
  normalisation over the WHOLE RIR (Hann 512 / hop 256, not centred, padded to 14592 -> 56 frames x 256 bins), patches
  16 bins x 4 frames -> 224 tokens, no envelope tokens. Frames past rir_len are masked (key padding, pooling, targets);
  MAE masks 75% of each RIR's valid tokens. Objective / decoders / 50-50 / budget / probes as a1; embedding = one
  attention pool over the valid tokens (384-d), mean pool also reported. Report vs A′ / v2: code/rir_compare.py.
Usage (GPU): python rir_encoder.py --out <dir> [--epochs 100] [--arch v1|v2|a1|a1u]   (--epochs 0: random-init embeddings + probes only)
Output: <dir>/rir_encoder.pth, emb_paired_v2.npy (12500, 384) fp32, emb_unpaired8.npy (25000, 384), emb_stats.npz
(mean / std per dim over the paired TRAINING scenes [0, 12000): the V1 z-score), probes.json, train_log.jsonl.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_DIR = '/scratch200/yotams2/SAC-SAR-SSL/data/MicSig'
SETS = (('paired_v2', 'simu_paired_pretrain_v2.tar', 12000), ('unpaired8', 'simu_unpaired8_pretrain.tar', 22500))
FS, NFFT, HOP, NFR, LPAD = 16000, 512, 128, 112, 512 + 111 * 128
PF, PT, NCH = 16, 4, 4
ROWS, COLS = 256 // PF, NFR // PT


def load_lens():
	""" true RIR lengths (N,) int64, rows in load_sets() order (A′-uniform's frame masks). """
	return torch.from_numpy(np.concatenate([np.load(os.path.join(D_DIR, tar + '.rirs.npz'))['rir_len'] for _, tar, _ in SETS]))


def load_sets():
	""" -> rir (N, 2, L) fp16 tensor (CPU), labels dict of (N,) float64, is_test (N,) bool, set_slices. """
	rirs, lab, test, slices, n0 = [], {k: [] for k in ('TDOA', 'T60', 'DRR', 'C50')}, [], {}, 0
	for name, tar, n_train in SETS:
		z = np.load(os.path.join(D_DIR, tar + '.rirs.npz'))
		r = z['rir']
		S = r.shape[0]
		m = np.load(os.path.join(D_DIR, tar + '.meta.npz'), allow_pickle=True)
		first = np.searchsorted(m['scene'], np.arange(S))  # meta rows are per clip, sorted by scene
		assert (m['scene'][first] == np.arange(S)).all()
		lab['TDOA'].append(np.asarray(m['TDOA'], np.float64).reshape(len(m['scene']), -1)[first, 0] * FS)
		lab['T60'].append(np.asarray(m['T60_edc'], np.float64).reshape(len(m['scene']), -1)[first, 0])
		lab['DRR'].append(np.asarray(m['DRR'], np.float64).reshape(len(m['scene']), -1)[first, 0])
		lab['C50'].append(np.asarray(m['C50'], np.float64).reshape(len(m['scene']), -1)[first, 0])
		rirs.append(torch.from_numpy(r))
		test.append(np.arange(S) >= n_train)
		slices[name] = (n0, n0 + S)
		n0 += S
		print(f'[data] {name}: {S} rooms, rir {tuple(r.shape)}, test rooms {int((np.arange(S) >= n_train).sum())}', flush=True)
	return torch.cat(rirs), {k: np.concatenate(v) for k, v in lab.items()}, np.concatenate(test), slices


def features(rir, scale=None):
	""" rir (B, 2, L) float on GPU -> (B, 4, 256, 112) compressed complex STFT maps (re0, im0, re1, im1). """
	B = rir.shape[0]
	x = F.pad(rir.float(), (0, LPAD - rir.shape[-1]))
	X = torch.stft(x.reshape(B * 2, -1), NFFT, HOP, window=torch.hann_window(NFFT, device=x.device), center=False,
				   return_complex=True)[:, :256]  # (2B, 256, 112)
	X = X * (X.abs() + 1e-8) ** -0.7
	out = torch.stack([X.real, X.imag], 1).reshape(B, 2, 2, 256, NFR).reshape(B, 4, 256, NFR)
	return out if scale is None else out / scale


def patchify(f):  # (B, 4, 256, 112) -> (B, 448, 256), frequency-major
	B = f.shape[0]
	return f.reshape(B, NCH, ROWS, PF, COLS, PT).permute(0, 2, 4, 1, 3, 5).reshape(B, ROWS * COLS, NCH * PF * PT)


class Block(nn.Module):
	def __init__(self, d, heads):
		super().__init__()
		self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
		self.att = nn.MultiheadAttention(d, heads, batch_first=True)
		self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

	def forward(self, x):
		h = self.n1(x)
		x = x + self.att(h, h, h, need_weights=False)[0]
		return x + self.mlp(self.n2(x))


class RIRMAE(nn.Module):
	def __init__(self, d=384, depth=8, heads=6, dd=256, ddepth=2, dheads=8, mask_ratio=0.75):
		super().__init__()
		P, N = NCH * PF * PT, ROWS * COLS
		self.mask_ratio, self.N = mask_ratio, N
		self.embed = nn.Linear(P, d)
		self.pos = nn.Parameter(torch.randn(1, N, d) * 0.02)
		self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
		self.norm = nn.LayerNorm(d)
		self.dec_in = nn.Linear(d, dd)
		self.mask_tok = nn.Parameter(torch.zeros(1, 1, dd))
		self.dpos = nn.Parameter(torch.randn(1, N, dd) * 0.02)
		self.dblocks = nn.ModuleList([Block(dd, dheads) for _ in range(ddepth)])
		self.dnorm = nn.LayerNorm(dd)
		self.head = nn.Linear(dd, P)
		self.register_buffer('scale', torch.ones(()))

	def encode(self, p, keep=None):
		x = self.embed(p) + self.pos
		if keep is not None:
			x = torch.gather(x, 1, keep[..., None].expand(-1, -1, x.shape[-1]))
		for b in self.blocks:
			x = b(x)
		return self.norm(x)

	def embedding(self, rir):
		""" rir (B, 2, L) -> (B, 384): the encoder on the unmasked input, mean over tokens. """
		return self.encode(patchify(features(rir, self.scale))).mean(1)

	def forward(self, rir):
		p = patchify(features(rir, self.scale))
		B, N = p.shape[:2]
		n_keep = int(round(N * (1 - self.mask_ratio)))
		order = torch.rand(B, N, device=p.device).argsort(1)
		keep, masked = order[:, :n_keep], order[:, n_keep:]
		z = self.dec_in(self.encode(p, keep))
		full = self.mask_tok.expand(B, N, -1).to(z.dtype).clone()
		full.scatter_(1, keep[..., None].expand(-1, -1, z.shape[-1]), z)
		y = full + self.dpos
		for b in self.dblocks:
			y = b(y)
		pred = self.head(self.dnorm(y))
		idx = masked[..., None].expand(-1, -1, p.shape[-1])
		return F.mse_loss(torch.gather(pred, 1, idx).float(), torch.gather(p, 1, idx).float())


E_LEN, E_FFT, E_HOP, E_FR = 4096, 256, 64, 64
L_FFT, L_HOP, L_FR, L_BANDS = 512, 256, 56, 16
_EDGES = None


def band_edges():
	global _EDGES
	if _EDGES is None:
		e = np.unique(np.round(np.geomspace(1, 257, 40)).astype(int))
		idx = np.linspace(0, len(e) - 1, L_BANDS + 1).round().astype(int)
		_EDGES = e[idx]
		_EDGES[0] = 0
		assert len(np.unique(_EDGES)) == L_BANDS + 1, _EDGES
	return _EDGES


def features_v2(rir):
	""" rir (B, 2, L) -> early (B, 2, 2, 128, 64) compressed complex (re, im), late (B, 2, 16, 56) log10 band power. """
	B = rir.shape[0]
	x = rir.float()
	xe = F.pad(x[..., :E_LEN], (0, E_FFT + (E_FR - 1) * E_HOP - E_LEN)).reshape(B * 2, -1)
	X = torch.stft(xe, E_FFT, E_HOP, window=torch.hann_window(E_FFT, device=x.device), center=False, return_complex=True)[:, :128]
	X = X * (X.abs() + 1e-8) ** -0.7
	early = torch.stack([X.real, X.imag], 1).reshape(B, 2, 2, 128, E_FR)
	xl = F.pad(x, (0, L_FFT + (L_FR - 1) * L_HOP - x.shape[-1])).reshape(B * 2, -1)
	P = torch.stft(xl, L_FFT, L_HOP, window=torch.hann_window(L_FFT, device=x.device), center=False, return_complex=True).abs() ** 2
	e = band_edges()
	bands = torch.stack([P[:, e[i]:e[i + 1]].sum(1) for i in range(L_BANDS)], 1)  # (2B, 16, 56)
	late = torch.log10(bands + 1e-10).reshape(B, 2, L_BANDS, L_FR)
	return early, late


def patchify_v2(early, late):
	B = early.shape[0]
	pe = early.reshape(B, 2, 2, 8, 16, 16, 4).permute(0, 1, 3, 5, 2, 4, 6).reshape(B, 2 * 8 * 16, 2 * 16 * 4)  # ch, row, col | re/im, bin, frame
	pl = late.reshape(B, 2, L_BANDS, 14, 4).permute(0, 1, 3, 2, 4).reshape(B, 2 * 14, L_BANDS * 4)
	return pe, pl


class RIRMAE2(nn.Module):
	NE, NL, PE, PL = 256, 28, 128, 64

	def __init__(self, d=384, depth=8, heads=6, dd=256, ddepth=2, dheads=8, mask_ratio=0.75):
		super().__init__()
		self.mask_ratio, self.N = mask_ratio, self.NE + self.NL
		self.embed_e, self.embed_l = nn.Linear(self.PE, d), nn.Linear(self.PL, d)
		self.pos = nn.Parameter(torch.randn(1, self.N, d) * 0.02)
		self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
		self.norm = nn.LayerNorm(d)
		self.dec_in = nn.Linear(d, dd)
		self.mask_tok = nn.Parameter(torch.zeros(1, 1, dd))
		self.dpos = nn.Parameter(torch.randn(1, self.N, dd) * 0.02)
		self.dblocks = nn.ModuleList([Block(dd, dheads) for _ in range(ddepth)])
		self.dnorm = nn.LayerNorm(dd)
		self.head_e, self.head_l = nn.Linear(dd, self.PE), nn.Linear(dd, self.PL)
		self.register_buffer('scale', torch.ones(()))  # early: global std
		self.register_buffer('late_mean', torch.zeros(()))
		self.register_buffer('late_std', torch.ones(()))

	def tokens(self, rir):
		early, late = features_v2(rir)
		return patchify_v2(early / self.scale, (late - self.late_mean) / self.late_std)

	def encode(self, pe, pl, keep=None):
		x = torch.cat([self.embed_e(pe), self.embed_l(pl)], 1) + self.pos
		if keep is not None:
			x = torch.gather(x, 1, keep[..., None].expand(-1, -1, x.shape[-1]))
		for b in self.blocks:
			x = b(x)
		return self.norm(x)

	def embedding(self, rir):
		return self.encode(*self.tokens(rir)).mean(1)

	def forward(self, rir):
		pe, pl = self.tokens(rir)
		B, N = pe.shape[0], self.N
		n_keep = int(round(N * (1 - self.mask_ratio)))
		order = torch.rand(B, N, device=pe.device).argsort(1)
		keep = order[:, :n_keep]
		z = self.dec_in(self.encode(pe, pl, keep))
		full = self.mask_tok.expand(B, N, -1).to(z.dtype).clone()
		full.scatter_(1, keep[..., None].expand(-1, -1, z.shape[-1]), z)
		y = full + self.dpos
		for b in self.dblocks:
			y = b(y)
		y = self.dnorm(y)
		masked = torch.ones(B, N, dtype=torch.bool, device=pe.device)
		masked.scatter_(1, keep, False)
		me, ml = masked[:, :self.NE], masked[:, self.NE:]
		le = ((self.head_e(y[:, :self.NE]).float() - pe.float()) ** 2).mean(-1)
		ll = ((self.head_l(y[:, self.NE:]).float() - pl.float()) ** 2).mean(-1)
		return 0.5 * ((le * me).sum() / me.sum().clamp_min(1) + (ll * ml).sum() / ml.sum().clamp_min(1))


I_FR = 16  # early frames (256 ms at hop 256)


def features_a1(rir):
	""" rir (B, 2, L) -> early ipd16 (B, 4, 256, 16), IPD weight (B, 1, 256, 16), late (B, 2, 16, 56). """
	x = rir.float()
	n = 512 + (I_FR - 1) * 256
	fr = F.pad(x[..., :n], (0, max(0, n - x.shape[-1]))).unfold(-1, 512, 256)  # (B, 2, 16, 512)
	X = torch.fft.rfft(fr * torch.hann_window(512, device=x.device), dim=-1)[..., 1:257]  # (B, 2, 16, 256)
	X = X / (X[:, 0].abs().mean(dim=(1, 2), keepdim=True)[:, None] + 1e-6)
	L, R = X[:, 0], X[:, 1]
	logL, logR = torch.log(L.abs() + 1e-3), torch.log(R.abs() + 1e-3)
	ipd = torch.angle(L * R.conj())
	early = torch.stack([logL, logR - logL, torch.cos(ipd), torch.sin(ipd)], 1).transpose(-1, -2)  # (B, 4, 256, 16)
	mag = (L.abs() * R.abs()).sqrt().transpose(-1, -2)[:, None]
	w = (mag / mag.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)).clamp(max=5.0)
	_, late = features_v2(rir)
	return early, w, late


def patchify_a1(early):  # (B, 4, 256, 16) -> (B, 64, 4 * 16 * 4): feature-major within a token
	B = early.shape[0]
	return early.reshape(B, 4, 16, 16, 4, 4).permute(0, 2, 4, 1, 3, 5).reshape(B, 64, 4 * 64)


class RIRMAE3(nn.Module):
	""" A' design 1: early ipd16 + late envelope tokens; random-masking MAE + cross-channel bottleneck; attention pool. """
	NE, NL, PE, PL = 64, 28, 256, 64

	def __init__(self, d=384, depth=8, heads=6, dd=256, ddepth=2, dheads=8, mask_ratio=0.75):
		super().__init__()
		self.mask_ratio, self.N, self.d = mask_ratio, self.NE + self.NL, d
		self.embed_e, self.embed_l = nn.Linear(self.PE, d), nn.Linear(self.PL, d)
		self.pos = nn.Parameter(torch.randn(1, self.N, d) * 0.02)
		self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
		self.norm = nn.LayerNorm(d)
		self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
		self.pool = nn.MultiheadAttention(d, 1, batch_first=True)
		# MAE decoder
		self.dec_in = nn.Linear(d, dd)
		self.mask_tok = nn.Parameter(torch.zeros(1, 1, dd))
		self.dpos = nn.Parameter(torch.randn(1, self.N, dd) * 0.02)
		self.dblocks = nn.ModuleList([Block(dd, dheads) for _ in range(ddepth)])
		self.dnorm = nn.LayerNorm(dd)
		self.head_e, self.head_l = nn.Linear(dd, self.PE), nn.Linear(dd, self.PL)
		# cross-channel decoder: left features + pooled embedding -> right-dependent features
		self.x_left, self.x_glob = nn.Linear(64, dd), nn.Linear(2 * d, dd)
		self.xpos = nn.Parameter(torch.randn(1, self.NE, dd) * 0.02)
		self.xblocks = nn.ModuleList([Block(dd, dheads) for _ in range(ddepth)])
		self.xnorm = nn.LayerNorm(dd)
		self.head_x = nn.Linear(dd, 192)
		self.register_buffer('e_mean', torch.zeros(1, 4, 1, 1))
		self.register_buffer('e_std', torch.ones(1, 4, 1, 1))
		self.register_buffer('late_mean', torch.zeros(()))
		self.register_buffer('late_std', torch.ones(()))
		self.register_buffer('scale', torch.ones(()))  # unused (kept for the shared code path)

	def tokens(self, rir):
		early, w, late = features_a1(rir)
		pe = patchify_a1((early - self.e_mean) / self.e_std)
		pw = patchify_a1(w.expand(-1, 4, -1, -1))[..., 2 * 64:]  # weights of the cos / sin values
		_, pl = patchify_v2(torch.zeros(rir.shape[0], 2, 2, 128, E_FR, device=rir.device), (late - self.late_mean) / self.late_std)
		return pe, pw, pl

	def encode(self, pe, pl, keep=None):
		x = torch.cat([self.embed_e(pe), self.embed_l(pl)], 1) + self.pos
		if keep is not None:
			x = torch.gather(x, 1, keep[..., None].expand(-1, -1, x.shape[-1]))
		for b in self.blocks:
			x = b(x)
		return self.norm(x)

	def pooled(self, h):
		a = self.pool(self.pool_q.expand(h.shape[0], -1, -1), h[:, :self.NE], h[:, :self.NE], need_weights=False)[0][:, 0]
		return torch.cat([a, h[:, self.NE:].mean(1)], -1)

	def embedding(self, rir):
		pe, _, pl = self.tokens(rir)
		return self.pooled(self.encode(pe, pl))

	def embedding_mean(self, rir):
		pe, _, pl = self.tokens(rir)
		return self.encode(pe, pl).mean(1)

	def loss_mae(self, pe, pl):
		B, N = pe.shape[0], self.N
		keep = torch.rand(B, N, device=pe.device).argsort(1)[:, :int(round(N * (1 - self.mask_ratio)))]
		z = self.dec_in(self.encode(pe, pl, keep))
		full = self.mask_tok.expand(B, N, -1).to(z.dtype).clone()
		full.scatter_(1, keep[..., None].expand(-1, -1, z.shape[-1]), z)
		y = full + self.dpos
		for b in self.dblocks:
			y = b(y)
		y = self.dnorm(y)
		masked = torch.ones(B, N, dtype=torch.bool, device=pe.device)
		masked.scatter_(1, keep, False)
		me, ml = masked[:, :self.NE], masked[:, self.NE:]
		le = ((self.head_e(y[:, :self.NE]).float() - pe.float()) ** 2).mean(-1)
		ll = ((self.head_l(y[:, self.NE:]).float() - pl.float()) ** 2).mean(-1)
		return 0.5 * ((le * me).sum() / me.sum().clamp_min(1) + (ll * ml).sum() / ml.sum().clamp_min(1))

	def loss_cross(self, pe, pw, pl):
		g = self.pooled(self.encode(pe, pl))
		y = self.x_left(pe[..., :64]) + self.x_glob(g)[:, None] + self.xpos
		for b in self.xblocks:
			y = b(y)
		pred = self.head_x(self.xnorm(y)).float()
		tgt = pe[..., 64:].float()
		e = (pred - tgt) ** 2  # (B, 64, 192): ILD 64 | cos 64 | sin 64
		w = torch.cat([torch.ones_like(pw[..., :64]), pw], -1).float()
		return (e * w).mean()

	def forward(self, rir, mode=None):
		pe, pw, pl = self.tokens(rir)
		if mode is None:
			mode = 'cross' if float(torch.rand(())) < 0.5 else 'mae'
		return self.loss_cross(pe, pw, pl) if mode == 'cross' else self.loss_mae(pe, pl)


# ---- A′-uniform (Yotam 11-25 ablation): one generic representation instead of A′'s early / late split ----
U_FR, U_COLS, U_N = 56, 14, 224  # whole RIR: Hann 512 / hop 256, not centred, padded to 512 + 55 * 256 = 14592 samples


class MBlock(Block):
	""" Block with a key-padding mask (True = ignore that token as a key). """
	def forward(self, x, kpm=None):
		h = self.n1(x)
		x = x + self.att(h, h, h, key_padding_mask=kpm, need_weights=False)[0]
		return x + self.mlp(self.n2(x))


def features_a1u(rir, lens):
	""" rir (B, 2, L), lens (B,) true RIR lengths -> ipd16 maps over the whole RIR (B, 4, 256, 56), IPD weight
		(B, 1, 256, 56), frame mask (B, 56) (frame f valid iff its window starts inside the RIR: f * 256 < len).
		Same features and per-RIR normalisation as A′'s early part (/ mean |L|), the mean taken over valid frames. """
	x = rir.float()
	n = 512 + (U_FR - 1) * 256
	fr = F.pad(x[..., :n], (0, max(0, n - x.shape[-1]))).unfold(-1, 512, 256)  # (B, 2, 56, 512)
	X = torch.fft.rfft(fr * torch.hann_window(512, device=x.device), dim=-1)[..., 1:257]  # (B, 2, 56, 256)
	fm = torch.arange(U_FR, device=x.device)[None] * 256 < lens[:, None].to(x.device)  # (B, 56)
	m = fm[:, :, None].float()
	X = X / ((X[:, 0].abs() * m).sum(dim=(1, 2)) / (m.sum(dim=(1, 2)) * 256)).clamp_min(1e-12)[:, None, None, None]
	L, R = X[:, 0], X[:, 1]
	logL, logR = torch.log(L.abs() + 1e-3), torch.log(R.abs() + 1e-3)
	ipd = torch.angle(L * R.conj())
	early = torch.stack([logL, logR - logL, torch.cos(ipd), torch.sin(ipd)], 1).transpose(-1, -2)  # (B, 4, 256, 56)
	mag = (L.abs() * R.abs()).sqrt().transpose(-1, -2)[:, None]  # (B, 1, 256, 56)
	mt = fm[:, None, None, :].float()
	w = (mag / ((mag * mt).sum(dim=(1, 2, 3), keepdim=True) / (mt.sum(dim=(1, 2, 3), keepdim=True) * 256)).clamp_min(1e-8)).clamp(max=5.0)
	return early, w, fm


def patchify_u(x):  # (B, C, 256, 56) -> (B, 224, C * 64): 16 bin rows x 14 frame columns, feature-major within a token
	B, C = x.shape[:2]
	return x.reshape(B, C, 16, 16, U_COLS, 4).permute(0, 2, 4, 1, 3, 5).reshape(B, U_N, C * 64)


class RIRMAE3U(nn.Module):
	""" A′-uniform: A′ design 1's objective (random-masking MAE 75% / cross-channel bottleneck, 50 / 50), decoders,
		encoder and budget, on ipd16 tokens of the WHOLE RIR (224 tokens, no envelope tokens, no split). Frames past the
		true RIR length are masked: not attended to (key padding), not pooled, not reconstruction targets (value mask),
		input values zeroed. MAE masks 75% of each RIR's VALID tokens. Embedding = one attention pool (1 learned query,
		1 head) over the valid tokens, 384-d; mean over valid tokens also reported. """
	N, P = U_N, 256

	def __init__(self, d=384, depth=8, heads=6, dd=256, ddepth=2, dheads=8, mask_ratio=0.75):
		super().__init__()
		self.mask_ratio, self.d = mask_ratio, d
		self.embed = nn.Linear(self.P, d)
		self.pos = nn.Parameter(torch.randn(1, self.N, d) * 0.02)
		self.blocks = nn.ModuleList([MBlock(d, heads) for _ in range(depth)])
		self.norm = nn.LayerNorm(d)
		self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
		self.pool = nn.MultiheadAttention(d, 1, batch_first=True)
		self.dec_in = nn.Linear(d, dd)
		self.mask_tok = nn.Parameter(torch.zeros(1, 1, dd))
		self.dpos = nn.Parameter(torch.randn(1, self.N, dd) * 0.02)
		self.dblocks = nn.ModuleList([MBlock(dd, dheads) for _ in range(ddepth)])
		self.dnorm = nn.LayerNorm(dd)
		self.head = nn.Linear(dd, self.P)
		self.x_left, self.x_glob = nn.Linear(64, dd), nn.Linear(d, dd)
		self.xpos = nn.Parameter(torch.randn(1, self.N, dd) * 0.02)
		self.xblocks = nn.ModuleList([MBlock(dd, dheads) for _ in range(ddepth)])
		self.xnorm = nn.LayerNorm(dd)
		self.head_x = nn.Linear(dd, 192)
		self.register_buffer('e_mean', torch.zeros(1, 4, 1, 1))
		self.register_buffer('e_std', torch.ones(1, 4, 1, 1))
		self.register_buffer('scale', torch.ones(()))  # unused (shared code path)

	def tokens(self, rir, lens):
		early, w, fm = features_a1u(rir, lens)
		vm = patchify_u(fm[:, None, None, :].expand(-1, 4, 256, -1).float())  # (B, 224, 256) value mask
		pe = patchify_u((early - self.e_mean) / self.e_std) * vm
		pw = patchify_u(w.expand(-1, 4, -1, -1))[..., 2 * 64:]  # weights of the cos / sin values
		valid = vm[..., 0] > 0  # token valid iff its first frame is valid
		return pe, pw, vm, valid

	def encode(self, pe, kpm):
		x = self.embed(pe) + self.pos
		for b in self.blocks:
			x = b(x, kpm)
		return self.norm(x)

	def pooled(self, h, valid):
		return self.pool(self.pool_q.expand(h.shape[0], -1, -1), h, h, key_padding_mask=~valid, need_weights=False)[0][:, 0]

	def embedding(self, rir, lens):
		pe, _, _, valid = self.tokens(rir, lens)
		return self.pooled(self.encode(pe, ~valid), valid)

	def embedding_mean(self, rir, lens):
		pe, _, _, valid = self.tokens(rir, lens)
		h, v = self.encode(pe, ~valid), valid[..., None].to(pe.dtype)
		return (h * v).sum(1) / v.sum(1)

	def loss_mae(self, pe, vm, valid):
		B = pe.shape[0]
		r = torch.rand(B, self.N, device=pe.device).masked_fill(~valid, 2.0).argsort(1).argsort(1)  # rank; invalid last
		n_keep = (valid.sum(1).float() * (1 - self.mask_ratio)).round().clamp_min(1)
		kept = r < n_keep[:, None]  # (B, N): 25% of the valid tokens, at random
		z = self.dec_in(self.encode(pe, ~kept))  # the encoder attends only to its kept tokens (= MAE's dropped tokens)
		y = torch.where(kept[..., None], z, self.mask_tok.to(z.dtype).expand(B, self.N, -1)) + self.dpos
		for b in self.dblocks:
			y = b(y, ~valid)
		tgt = (valid & ~kept)[..., None].float() * vm  # masked valid tokens, valid frames only (padding is never a target)
		e = (self.head(self.dnorm(y)).float() - pe.float()) ** 2
		return (e * tgt).sum() / tgt.sum().clamp_min(1)

	def loss_cross(self, pe, pw, vm, valid):
		g = self.pooled(self.encode(pe, ~valid), valid)
		y = self.x_left(pe[..., :64]) + self.x_glob(g)[:, None] + self.xpos
		for b in self.xblocks:
			y = b(y, ~valid)
		e = (self.head_x(self.xnorm(y)).float() - pe[..., 64:].float()) ** 2  # (B, 224, 192): ILD | cos | sin
		w = torch.cat([torch.ones_like(pw[..., :64]), pw], -1).float() * vm[..., 64:]
		return (e * w).sum() / vm[..., 64:].sum().clamp_min(1)

	def forward(self, rir, mode=None, lens=None):
		pe, pw, vm, valid = self.tokens(rir, lens)
		if mode is None:
			mode = 'cross' if float(torch.rand(())) < 0.5 else 'mae'
		return self.loss_cross(pe, pw, vm, valid) if mode == 'cross' else self.loss_mae(pe, vm, valid)


@torch.no_grad()
def embed_all(model, rir, dev, bs=512, fn='embedding', lens=None):
	model.eval()
	out = []
	for b in range(0, len(rir), bs):
		args = (rir[b:b + bs].to(dev),) + (() if lens is None else (lens[b:b + bs].to(dev),))  # lens: A′-uniform
		with torch.autocast('cuda', dtype=torch.bfloat16):
			out.append(getattr(model, fn)(*args).float().cpu())
	return torch.cat(out).numpy().astype(np.float64)


def ridge_cv(Xtr, ytr, Xte, lams=np.logspace(-3, 5, 33), k=5, seed=0):
	""" closed-form ridge with intercept on z-scored X, penalty by k-fold CV MAE; -> test predictions, lambda, edge. """
	mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
	Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
	folds = np.random.RandomState(seed).permutation(len(Xtr)) % k

	def fit_predict(A, y, B):
		ym = y.mean()
		U, s, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
		uy = U.T @ (y - ym)
		return [(B - A.mean(0)) @ (Vt.T @ ((s / (s ** 2 + l)) * uy)) + ym for l in lams]

	cv = np.zeros(len(lams))
	for f in range(k):
		tr, va = folds != f, folds == f
		for i, p in enumerate(fit_predict(Xtr[tr], ytr[tr], Xtr[va])):
			cv[i] += np.abs(p - ytr[va]).sum()
	i = int(np.argmin(cv))
	return fit_predict(Xtr, ytr, Xte)[i], float(lams[i]), i in (0, len(lams) - 1)


def xcorr_tdoa(rir_dp):
	""" TDOA (samples, ch1 relative to ch0, the meta's sign convention checked below) from the direct-path RIRs. """
	a, b = rir_dp[:, 0].astype(np.float64), rir_dp[:, 1].astype(np.float64)
	n = 2 * a.shape[1]
	c = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
	L = 64
	vals = np.concatenate([c[:, -L:], c[:, :L + 1]], 1)
	return np.argmax(vals, 1) - L


def probes(emb, lab, test, extra=None):
	res = {}
	tr, te = ~test, test
	for t, y in lab.items():
		p, lam, edge = ridge_cv(emb[tr], y[tr], emb[te])
		res[t] = dict(mae=float(np.abs(p - y[te]).mean()), mean_baseline=float(np.abs(y[tr].mean() - y[te]).mean()),
					  lam=lam, edge=edge, label_std_test=float(y[te].std()))
	return res


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', required=True)
	ap.add_argument('--epochs', type=int, default=100)
	ap.add_argument('--bs', type=int, default=256)
	ap.add_argument('--lr', type=float, default=5e-4)
	ap.add_argument('--wd', type=float, default=0.05)
	ap.add_argument('--warmup-frac', type=float, default=0.05)
	ap.add_argument('--seed', type=int, default=1)
	ap.add_argument('--arch', choices=['v1', 'v2', 'a1', 'a1u'], default='v1')
	ap.add_argument('--ceiling-dir', default='/scratch200/yotams2/SAC-SAR-SSL/exp/rir_encoder/ceiling')
	a = ap.parse_args()
	os.makedirs(a.out, exist_ok=True)
	torch.manual_seed(a.seed); np.random.seed(a.seed)
	dev = 'cuda'
	rir, lab, test, slices = load_sets()
	U = a.arch == 'a1u'
	lens = load_lens() if U else None
	assert lens is None or len(lens) == len(rir)
	tr_idx = np.where(~test)[0]
	model = {'v1': RIRMAE, 'v2': RIRMAE2, 'a1': RIRMAE3, 'a1u': RIRMAE3U}[a.arch]().to(dev)
	with torch.no_grad():
		si = torch.from_numpy(np.random.RandomState(0).choice(tr_idx, 2048, replace=False))
		smp = rir[si].to(dev)
		if U:  # per-feature z-norm over VALID frames only
			e, _, fm = features_a1u(smp, lens[si])
			m = fm[:, None, None, :].float()
			mu = (e * m).sum(dim=(0, 2, 3), keepdim=True) / (m.sum() * 256)
			model.e_mean.copy_(mu); model.e_std.copy_((((e - mu) ** 2 * m).sum(dim=(0, 2, 3), keepdim=True) / (m.sum() * 256)).sqrt())
			s = torch.ones(())
			print(f'[model] a1u: ipd16 over valid frames mean {model.e_mean.flatten().tolist()} std {model.e_std.flatten().tolist()}; '
				  f'valid frames per RIR {fm.sum(1).float().mean():.1f} (min {int(fm.sum(1).min())}, max {int(fm.sum(1).max())})', flush=True)
		elif a.arch == 'a1':
			e, _, l = features_a1(smp)
			model.e_mean.copy_(e.mean(dim=(0, 2, 3), keepdim=True)); model.e_std.copy_(e.std(dim=(0, 2, 3), keepdim=True))
			model.late_mean.fill_(float(l.mean())); model.late_std.fill_(float(l.std()))
			s = torch.ones(())
			print(f'[model] a1: early ipd16 mean {model.e_mean.flatten().tolist()} std {model.e_std.flatten().tolist()}', flush=True)
		elif a.arch == 'v1':
			s = features(smp).std()
		else:
			e, l = features_v2(smp)
			s = e.std()
			model.late_mean.fill_(float(l.mean())); model.late_std.fill_(float(l.std()))
			print(f'[model] v2: early std {float(s):.4f}; late log10 power mean {float(l.mean()):.3f} std {float(l.std()):.3f}', flush=True)
		model.scale.fill_(float(s))
	print(f'[model] {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters; input scale {float(s):.4f}', flush=True)
	report = {}
	# reference rows: random init, and TDOA from the direct-path cross-correlation
	t0 = time.time()
	report['random_init'] = probes(embed_all(model, rir, dev, lens=lens), lab, test)
	rdp = np.concatenate([np.load(os.path.join(D_DIR, tar + '.rirs.npz'))['rir_dp'] for _, tar, _ in SETS])
	xt = xcorr_tdoa(rdp)
	sign = 1 if np.abs(xt - lab['TDOA']).mean() <= np.abs(-xt - lab['TDOA']).mean() else -1
	report['xcorr_tdoa_direct_path'] = dict(mae_test=float(np.abs(sign * xt[test] - lab['TDOA'][test]).mean()), sign=sign)
	print(f"[ref] random init {json.dumps({t: round(v['mae'], 4) for t, v in report['random_init'].items()})}; "
		  f"mean baseline {json.dumps({t: round(v['mean_baseline'], 4) for t, v in report['random_init'].items()})}; "
		  f"xcorr TDOA (direct path) MAE {report['xcorr_tdoa_direct_path']['mae_test']:.3f} samples ({time.time() - t0:.0f} s)", flush=True)
	n_steps = a.epochs * (len(tr_idx) // a.bs)
	if n_steps > 0:
		opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
		wu = max(1, int(a.warmup_frac * n_steps))
		sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / wu if s < wu else
												  0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - wu) / max(1, n_steps - wu))))
		te_idx = torch.from_numpy(np.where(test)[0])
		log = open(os.path.join(a.out, 'train_log.jsonl'), 'a')
		step, t0 = 0, time.time()
		for ep in range(a.epochs):
			model.train()
			perm = torch.from_numpy(np.random.permutation(tr_idx))
			tot, nb = 0.0, 0
			for b in range(0, len(perm) - a.bs + 1, a.bs):
				x = rir[perm[b:b + a.bs]].to(dev, non_blocking=True)
				sw = torch.rand(x.shape[0], device=dev) < 0.5
				x = torch.where(sw[:, None, None], x.flip(1), x)
				with torch.autocast('cuda', dtype=torch.bfloat16):
					loss = model(x, lens=lens[perm[b:b + a.bs]].to(dev)) if U else model(x)
				opt.zero_grad(set_to_none=True)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				opt.step(); sched.step(); step += 1
				tot += loss.item(); nb += 1
			model.eval()
			with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
				torch.manual_seed(0)
				modes = ('mae', 'cross') if a.arch in ('a1', 'a1u') else (None,)
				kw = lambda b: {'lens': lens[te_idx[b:b + 512]].to(dev)} if U else {}
				vls = {m: float(np.mean([(model(rir[te_idx[b:b + 512]].to(dev), m, **kw(b)) if m else model(rir[te_idx[b:b + 512]].to(dev))).item()
										 for b in range(0, len(te_idx), 512)])) for m in modes}
				vl = float(np.mean(list(vls.values())))
			rec = dict(epoch=ep + 1, step=step, train=tot / nb, val=vl, lr=sched.get_last_lr()[0], min=(time.time() - t0) / 60)
			if a.arch in ('a1', 'a1u'):
				rec.update({f'val_{m}': v for m, v in vls.items()})
			log.write(json.dumps(rec) + '\n'); log.flush()
			if (ep + 1) % 5 == 0 or ep == 0 or ep + 1 == a.epochs:
				print(f'[train] {json.dumps(rec)}', flush=True)
			if (ep + 1) % 10 == 0 or ep + 1 == a.epochs:
				torch.save({'model': model.state_dict(), 'args': vars(a), 'epoch': ep + 1}, os.path.join(a.out, 'rir_encoder.pth'))
	emb = embed_all(model, rir, dev, lens=lens)
	report['trained' if n_steps > 0 else 'this_model'] = probes(emb, lab, test)
	for name, (lo, hi) in slices.items():
		np.save(os.path.join(a.out, f'emb_{name}.npy'), emb[lo:hi].astype(np.float32))
	lo, hi = slices['paired_v2']
	pe = emb[lo:hi][:SETS[0][2]]
	np.savez(os.path.join(a.out, 'emb_stats.npz'), mean=pe.mean(0).astype(np.float32), std=(pe.std(0) + 1e-6).astype(np.float32),
			 note='z-score for V1: statistics over the paired v2 training scenes [0, 12000)')
	if a.arch in ('a1', 'a1u'):
		em = embed_all(model, rir, dev, fn='embedding_mean', lens=lens)
		report['trained_meanpool'] = probes(em, lab, test)
		bf = [os.path.join(a.ceiling_dir, f'oracle_B_{n}.npy') for n, _, _ in SETS]
		if all(os.path.isfile(f) for f in bf):
			Bv = np.concatenate([np.load(f) for f in bf]).astype(np.float64)
			ti = np.where(test)[0]
			rs = np.random.RandomState(0)
			i, j = ti[rs.randint(0, len(ti), 10000)], ti[rs.randint(0, len(ti), 10000)]
			db = np.linalg.norm(Bv[i] - Bv[j], axis=1)
			rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)
			for key, E in ((('attn_pool' if U else 'concat_pool'), emb), ('mean_pool', em)):
				Z = (E - E[~test].mean(0)) / (E[~test].std(0) + 1e-6)
				Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
				de = 1 - (Zn[i] * Zn[j]).sum(1)
				report[f'spearman_{key}_vs_oracleB'] = float(np.corrcoef(rank(de), rank(db))[0, 1])
				print(f'[spearman] {key}: cosine distance (z-scored embedding) vs oracle-B distance, 10k held-out pairs: '
					  f"{report[f'spearman_{key}_vs_oracleB']:.3f}", flush=True)
	report['epochs'], report['n_train_rooms'], report['n_test_rooms'] = a.epochs, int((~test).sum()), int(test.sum())
	json.dump(report, open(os.path.join(a.out, 'probes.json'), 'w'), indent=1)
	key = 'trained' if n_steps > 0 else 'this_model'
	print('[probes] test MAE on held-out rooms (TDOA samples, T60 s, DRR dB, C50 dB)')
	for row in ('random_init', key) + (('trained_meanpool',) if 'trained_meanpool' in report else ()):
		print(f'  {row:12s} ' + '  '.join(f"{t} {v['mae']:.4f}{' (edge)' if v['edge'] else ''}" for t, v in report[row].items()))
	print(f"  {'mean':12s} " + '  '.join(f"{t} {v['mean_baseline']:.4f}" for t, v in report['random_init'].items()))
	print(f"[done] -> {a.out}", flush=True)


if __name__ == '__main__':
	main()
