"""
	Pilot C: two dedicated encoders trained by a shared swap-reconstruction objective (sprint 2026-11-16 -> 11-30,
	proposed and agreed 2026-11-20; see sac-ssamba STATUS / docs/log).

	Data: the paired set of data_generation/gen_simu_paired.py (scene x utterance, each scene with k utterances, each
	utterance in k scenes). For a target clip x(s, r) (utterance s in scene r):
		spectral encoder  <- x(s, r')   same utterance, another scene    (random among the k - 1)
		spatial encoder   <- x(s', r)   another utterance, same scene    (random among the k - 1)
		decoder           -> x(s, r)    the target, which neither encoder saw
	The spectral input carries the right source in the wrong scene, the spatial input the right scene with the wrong
	source, so the loss can only fall if the spectral encoder supplies the source and the spatial encoder the scene.

	Encoders: exactly Pilot A's (joint_pretrain.py, --streams joint, no fusion): spectral = 128-mel log-mel of the mid
	channel, spatial = ipd16, both from the shared SAR-SSL STFT, 16 time columns per clip. Saved in the same formats, so
	every Pilot A/B eval (spatial tasks, speech probes, concat, leakage) runs on them unchanged.

	Decoder (pretraining only): 256 learned queries (one per 16x16 ipd16 patch of the target) -> 4 Transformer decoder
	layers (d 512, 8 heads, pre-norm; self-attention over queries, cross-attention to the concatenated encoder tokens,
	each projected 768 -> 512 plus a learned stream embedding) -> Linear -> the target's standardised ipd16 patches
	(log|L|, ILD, cos IPD, sin IPD; spatial stream's input statistics). Loss = mean of the 4 per-channel MSEs; the IPD
	channels are weighted per bin by min(sqrt(|L||R|) / clip mean, 5), so phase in near-silent bins counts little.

	Objective: swap_lambda * swap + own_scale * (Pilot A's per-stream 10 * MPG + 1 * MPC on each encoder's own input).
	own_scale 0 = swap only. Validation also reports the swap loss with the spectral or the spatial input taken from
	another clip in the batch (rolled by one): how much the decoder relies on each stream, per channel.

	C2 (--ctr-lambda > 0, agreed 2026-11-21): + exact-pair InfoNCE per encoder on the MEAN-POOLED final tokens (exactly
	what the frozen probes read; no projection head, so the invariance must hold in the probed embedding):
		anchor = the target x(s, r), encoded by each encoder (gradient)
		spectral: positive x(s, r')  (its own input: same utterance); hard negative x(s', r) = the anchor's scene
		spatial:  positive x(s', r)  (its own input: same scene);     hard negative x(s, r') = the anchor's utterance
	so each hard negative shares exactly the anchor's WRONG factor: carrying it pulls the anchor toward the hard negative.
	(The first version anchored on each encoder's own input, whose hard negative then shared nothing with the anchor:
	hard-negative accuracy ~100% from step 10 in smoke 23717145. Fixed 2026-11-21 before any C2 training.)
	candidates = all positives + all hard negatives of the batch; candidates sharing the anchor's factor (utterance /
	scene) other than its positive are masked (false negatives). No gradient through the hard negatives. Logged per
	encoder: ctr_<k> (loss), ctr_<k>_acc (top-1 over all candidates), ctr_<k>_hard (positive scored above its hard negative).
	C2b (--align-lambda > 0, agreed 2026-11-21): + alignment 1 - cos(anchor, positive) per encoder on the same pooled
	pairs (does not saturate once the positive ranks first). Logged: align_<k>, emb_std_<k> (per-dimension std over the
	batch of the pooled anchor embeddings, before normalisation; a fall towards 0 = collapse).
	--grad-clip: clip the total gradient norm (0 = off). The norm before clipping is logged every step to grad_norm.txt,
	and per log line as grad_norm (mean), grad_norm_max and, with clipping on, grad_clip_frac.
	--grad-clip-median-mult M (agreed 2026-11-21 for C2b / C3'): clip at M x the median of the previous --grad-clip-window
	(200) raw norms (clipped steps included); no clipping until 20 norms exist; the window is saved in latest.pth.
	Logged per line: grad_clip_thr (threshold at the last step).
	C + SAC (--sac-lambda > 0, agreed 2026-11-22): + in-batch SAC (CWCL) on the SPATIAL encoder only, exactly as the
	ipd16_sac run (ssamba_spatial_pretrain.py): the 5 SAR-SSL spatial features (coherence low / mid / high, GCC-PHAT max,
	TDOA) of the spatial encoder's own input clip (view 2), from the same normalised band-limited STFT; z-norm statistics
	from sac_feature_stats (50 batches, computed once at the start, saved in latest.pth); a Linear-BN-ReLU-Linear(128)
	head on the mean-pooled unmasked spatial tokens; loss model.SARSSL._calc_single_feat_sim_loss (sigma 0.613, temp
	0.15 by default). Logged: sac. Off by default (the head is only built when on, so older checkpoints load unchanged).
	--spat-mpg-no-logL: the spatial encoder's own MPG target without the log|L| channel (input unchanged).
	Validation pairs: --val-pairing fixed (default) draws each validation target's partners from its own seed, so every
	validation pass sees the same triplets; Pilot C seed 1 ran with 'random' (new partners every pass).

	Outputs in --exp-dir as joint_pretrain.py: latest.pth, log.jsonl, args.json, spat_encoder_*.pth, spec_encoder_*.pth,
	joint_{best,final}.pth (encoders + decoder).
"""
import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.utils_module as at_module  # noqa: E402
import joint_pretrain as jp  # noqa: E402
from ssamba_spatial_pretrain import (NFFT, NT, WIN_LEN, WIN_SHIFT_RATIO, FRE_RANGE, TarMicSigDataset,  # noqa: E402
									 save_atomic, FS, SAC_FEATURES, sac_feature_stats)

CH_NAMES = ('logL', 'ILD', 'cosIPD', 'sinIPD')


# ---------------------------------------------------------------- data
class PairedDataset(torch.utils.data.Dataset):
	""" Target clips of scenes [scene_lo, scene_hi) of the paired tar; each item (3, nsample, 2) float32 =
		(target x(s, r), spectral input x(s, r'), spatial input x(s', r)). Partners are drawn per access (torch RNG:
		seeded per DataLoader worker). Partners may come from any scene of the tar. """
	def __init__(self, tar, scene_lo, scene_hi, max_items=None, fixed_seed=None):
		self.fixed_seed = fixed_seed  # None: partners from the torch RNG; int: from a per-target seed (fixed triplets)
		m = np.load(tar + '.meta.npz', allow_pickle=True)
		self.k, A = int(m['k']), m['assignment']
		assert (m['utt'] == A.ravel()).all()
		self.clips = TarMicSigDataset(tar, 'pretrain')
		assert len(self.clips) == A.size, (len(self.clips), A.size)
		self.utt = A.ravel()
		self.utt_clips = {}
		for c, u in enumerate(self.utt):
			self.utt_clips.setdefault(int(u), []).append(c)
		self.targets = list(range(scene_lo * self.k, scene_hi * self.k))[:max_items]

	def __len__(self):
		return len(self.targets)

	def partners(self, c):
		r, j = divmod(c, self.k)
		others = [x for x in self.utt_clips[int(self.utt[c])] if x // self.k != r]
		if self.fixed_seed is None:
			i_spec, jj = int(torch.randint(len(others), ())), int(torch.randint(self.k - 1, ()))
		else:
			rs = np.random.RandomState((self.fixed_seed * 1_000_003 + c) % 2**32)
			i_spec, jj = int(rs.randint(len(others))), int(rs.randint(self.k - 1))
		c_spec = others[i_spec]
		c_spat = r * self.k + (jj if jj < j else jj + 1)
		return c_spec, c_spat

	def __getitem__(self, i):
		""" -> (3, nsample, 2) signals, (3, 2) int64 [utterance, scene] of the target, spectral and spatial view. """
		c = self.targets[i]
		c_spec, c_spat = self.partners(c)
		ids = np.array([[int(self.utt[x]), x // self.k] for x in (c, c_spec, c_spat)], dtype=np.int64)
		return np.stack([self.clips[c], self.clips[c_spec], self.clips[c_spat]]), ids


# ---------------------------------------------------------------- model
class SwapDecoder(nn.Module):
	def __init__(self, n_queries, out_dim, d_enc=768, d=512, layers=4, heads=8):
		super().__init__()
		self.queries = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)
		self.proj = nn.ModuleDict({k: nn.Linear(d_enc, d) for k in ('spec', 'spat')})
		self.stream_emb = nn.ParameterDict({k: nn.Parameter(torch.zeros(1, 1, d)) for k in ('spec', 'spat')})
		layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
		self.dec = nn.TransformerDecoder(layer, layers)
		self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, out_dim)

	def forward(self, toks):
		mem = torch.cat([self.proj[k](toks[k].float()) + self.stream_emb[k] for k in ('spec', 'spat')], dim=1)
		q = self.queries.expand(mem.shape[0], -1, -1)
		return self.head(self.norm(self.dec(q, mem)))


class SwapPretrain(jp.JointPretrain):
	def __init__(self, mask_cols, recon_lambda, classif_lambda, swap_lambda, own_scale, use_checkpointing=True,
				 ctr_lambda=0.0, ctr_temp=0.1, spat_mpg_no_logL=False, align_lambda=0.0,
				 sac_lambda=0.0, sac_sigma=0.613, sac_temp=0.15):
		super().__init__('joint', False, 0.0, mask_cols, recon_lambda, classif_lambda, use_checkpointing)
		self.swap_lambda, self.own_scale = swap_lambda, own_scale
		self.ctr_lambda, self.ctr_temp, self.spat_mpg_no_logL = ctr_lambda, ctr_temp, spat_mpg_no_logL
		self.align_lambda = align_lambda
		self.P = 16 * 16  # values per patch per channel
		self.decoder = SwapDecoder(n_queries=16 * 16, out_dim=4 * self.P)
		self.sac_lambda, self.sac_sigma, self.sac_temp = sac_lambda, sac_sigma, sac_temp
		D = 768
		self.feat_sim_proj = nn.Sequential(nn.Linear(D, D), nn.BatchNorm1d(D), nn.ReLU(), nn.Linear(D, 128)) if sac_lambda > 0 else None
		self.feature_stats = {}  # SAC z-norm stats; read by SARSSL._calc_single_feat_sim_loss with the two below
		self.normalization_mode, self.factorized_sac = 'z_norm', False
		self._stft_spat = None

	def views(self, sig3, stft_mod, extra=False):
		""" sig3 (B, 3, nsample, 2) -> standardised spectral input (of view 1), spatial input (of view 2), target (of
			view 0, spatial statistics), the IPD weight of the target (B, 1, 256, 256) and, if extra, the contrastive
			inputs: positives {spec: view 0, spat: view 0}, hard negatives {spec: view 2, spat: view 1}. """
		B = sig3.shape[0]
		stft = jp.shared_stft(stft_mod, sig3.flatten(0, 1)).unflatten(0, (B, 3))  # each clip normalised by itself
		sp_in = lambda v: self.streams['spec'].norm_input(self.inputs.spec(stft[:, v]))
		sa_in = lambda v: self.streams['spat'].norm_input(self.inputs.spat(stft[:, v]))
		spec, spat, tgt = sp_in(1), sa_in(2), sa_in(0)
		self._stft_spat = stft[:, 2][:, :, FRE_RANGE] if self.sac_lambda > 0 else None  # SAC features of the spatial input
		mag = jp.fit_frames((stft[:, 0, :, FRE_RANGE].abs().prod(dim=1)).sqrt()[:, None])  # (B, 1, 256, 256)
		w = (mag / mag.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)).clamp(max=5.0)
		if not extra:
			return spec, spat, tgt, w, None, None
		return spec, spat, tgt, w, {'spec': sp_in(0), 'spat': tgt}, {'spec': sp_in(2), 'spat': sa_in(1)}  # anchors, hard negs

	def mpg_no_logL(self, Xn, out_final, cmask):
		""" spatial stream MPG as JointStream.losses('mpg') without the log|L| channel (patches are channel-major). """
		st = self.streams['spat']
		x = st.enc.v.norm_f(out_final)[:, 1:]
		patches = st.enc.unfold(Xn).transpose(1, 2)
		m = cmask[:, None, :].expand(-1, st.n_rows, -1).reshape(x.shape[0], -1)
		idx = m.float().argsort(dim=1, descending=True, stable=True)[:, :int(m[0].sum())]
		h = torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
		tgt = torch.gather(patches, 1, idx[..., None].expand(-1, -1, patches.shape[-1])).detach()
		return {'mpg': torch.mean((st.enc.gpredlayer(h).float()[..., self.P:] - tgt.float()[..., self.P:]) ** 2)}

	@staticmethod
	def pooled(h):
		""" what the frozen probes read: mean over the real tokens (cls dropped), L2-normalised for the InfoNCE. """
		return F.normalize(h[:, 1:].float().mean(dim=1), dim=-1)

	def ctr_losses(self, toks, Xpos, Xneg, ids):
		pos = self.encode(Xpos, None, fusion=False)
		with torch.no_grad():
			neg = self.encode(Xneg, None, fusion=False)
		B = ids.shape[0]
		ar = torch.arange(B, device=ids.device)
		out = {}
		for k, vp, vn, f in (('spec', 1, 2, 0), ('spat', 2, 1, 1)):  # anchor = view 0 (target); f: 0 utterance, 1 scene
			va = 0
			out[f'emb_std_{k}'] = pos[k][:, 1:].float().mean(dim=1).std(dim=0).mean().detach()
			a, p, h = self.pooled(pos[k]), self.pooled(toks[k]), self.pooled(neg[k])  # target, own input, hard negative
			out[f'align_{k}'] = (1 - (a * p).sum(dim=-1)).mean()
			logits = a @ torch.cat([p, h]).T / self.ctr_temp  # (B, 2B): positives, then hard negatives
			same = ids[:, va, f][:, None] == torch.cat([ids[:, vp, f], ids[:, vn, f]])[None, :]
			same[ar, ar] = False  # the anchor's own positive stays
			logits = logits.masked_fill(same, float('-inf'))
			out[f'ctr_{k}'] = F.cross_entropy(logits, ar)
			out[f'ctr_{k}_acc'] = (logits.argmax(dim=1) == ar).float().mean()
			out[f'ctr_{k}_hard'] = (logits[ar, ar] > logits[ar, B + ar]).float().mean()
		return out

	def patchify(self, X):
		return self.streams['spat'].enc.unfold(X).transpose(1, 2)  # (B, 256 patches, C * 256), channel-major

	def swap_losses(self, toks, tgt, w):
		pred = self.decoder(toks).float()
		T, W = self.patchify(tgt).float(), self.patchify(w).float()
		out = {}
		for c, name in enumerate(CH_NAMES):
			e = (pred[..., c * self.P:(c + 1) * self.P] - T[..., c * self.P:(c + 1) * self.P]) ** 2
			out[f'swap_{name}'] = (e * W).mean() if c >= 2 else e.mean()
		out['swap'] = sum(out[f'swap_{n}'] for n in CH_NAMES) / len(CH_NAMES)
		return out

	def forward(self, sig3, stft_mod, diag=False, ids=None):
		use_ctr = self.ctr_lambda > 0 or self.align_lambda > 0
		spec, spat, tgt, w, Xpos, Xneg = self.views(sig3, stft_mod, extra=use_ctr)
		Xn = {'spec': spec, 'spat': spat}
		B, dev = spec.shape[0], spec.device
		out, loss = {}, 0.0
		if self.own_scale > 0:  # Pilot A's objective on each encoder's own input
			for kind, lam in (('mpg', self.recon_lambda), ('mpc', self.classif_lambda)):
				cmask = jp.column_mask(B, self.mask_cols, device=dev)
				enc = self.encode(Xn, cmask, fusion=False)
				for k in Xn:
					if k == 'spat' and kind == 'mpg' and self.spat_mpg_no_logL:
						lo = self.mpg_no_logL(Xn[k], enc[k], cmask)
					else:
						lo = self.streams[k].losses(Xn[k], enc[k], cmask, kind)
					for name, v in lo.items():
						out[f'{k}_{name}'] = v
					loss = loss + self.own_scale * lam * out[f'{k}_{kind}']
		toks = self.encode(Xn, None, fusion=False)
		sw = self.swap_losses(toks, tgt, w)
		out.update(sw)
		loss = loss + self.swap_lambda * sw['swap']
		if self.sac_lambda > 0:  # as ssamba_spatial_pretrain.SSAMBASpatialPretrain.forward
			import model as at_model
			z_proj = self.feat_sim_proj(toks['spat'][:, 1:].float().mean(dim=1))
			with torch.no_grad():
				feats = at_module.spatial_features_from_stft(self._stft_spat, sample_rate=FS)
			out['sac'], _ = at_model.SARSSL._calc_single_feat_sim_loss(
				self, z_proj, feats, SAC_FEATURES, self.sac_temp, self.sac_sigma, 1e-8, False)
			loss = loss + self.sac_lambda * out['sac']
		if use_ctr:
			assert ids is not None, 'the contrastive term needs the view ids'
			ct = self.ctr_losses(toks, Xpos, Xneg, ids)
			out.update(ct)
			loss = loss + self.ctr_lambda * (ct['ctr_spec'] + ct['ctr_spat'])
			if self.align_lambda > 0:
				loss = loss + self.align_lambda * (ct['align_spec'] + ct['align_spat'])
		if diag:  # reliance on each stream: its tokens from another clip of the batch
			for k in ('spec', 'spat'):
				t2 = dict(toks); t2[k] = toks[k].roll(1, dims=0)
				out.update({f'{n}_shuf{k}': v for n, v in self.swap_losses(t2, tgt, w).items()})
		out['loss'] = loss
		return out


# ---------------------------------------------------------------- train
@torch.no_grad()
def evaluate(model, stft_mod, loader, device, seed=1234):
	model.eval()
	rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
		   np.random.get_state(), random.getstate())
	torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
	tot, n = {}, 0
	for sig3, ids in loader:
		out = model(sig3.to(device), stft_mod, diag=True, ids=ids.to(device))
		for k, v in out.items():
			tot[k] = tot.get(k, 0.0) + float(v) * sig3.shape[0]
		n += sig3.shape[0]
	torch.set_rng_state(rng[0])
	if torch.cuda.is_available():
		torch.cuda.set_rng_state_all(rng[1])
	np.random.set_state(rng[2]); random.setstate(rng[3])
	model.train()
	return {k: v / n for k, v in tot.items()}


class _First(torch.utils.data.Dataset):  # target view only, for the input statistics
	def __init__(self, ds):
		self.ds = ds

	def __len__(self):
		return len(self.ds)

	def __getitem__(self, i):
		return self.ds.clips[self.ds.targets[i]]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--tar', required=True, help='paired tar of gen_simu_paired.py (+ .meta.npz)')
	ap.add_argument('--exp-dir', required=True)
	ap.add_argument('--val-scenes', type=int, default=500, help='last N scenes = validation targets')
	ap.add_argument('--num-train', type=int, default=None)
	ap.add_argument('--num-val', type=int, default=None)
	ap.add_argument('--mask-cols', type=int, default=7)
	ap.add_argument('--bs', type=int, default=128)
	ap.add_argument('--lr', type=float, default=4e-4)
	ap.add_argument('--nepoch', type=int, default=10)
	ap.add_argument('--recon-lambda', type=float, default=10.0)
	ap.add_argument('--classif-lambda', type=float, default=1.0)
	ap.add_argument('--swap-lambda', type=float, default=10.0)
	ap.add_argument('--own-scale', type=float, default=0.1, help='weight of Pilot A per-stream losses (0 = swap only)')
	ap.add_argument('--ctr-lambda', type=float, default=0.0, help='C2: weight of each encoder\'s exact-pair InfoNCE (0 = Pilot C)')
	ap.add_argument('--ctr-temp', type=float, default=0.1)
	ap.add_argument('--align-lambda', type=float, default=0.0, help='C2b: weight of each encoder\'s 1 - cos(anchor, positive)')
	ap.add_argument('--grad-clip', type=float, default=0.0, help='clip the total gradient norm at this value (0 = off)')
	ap.add_argument('--grad-clip-median-mult', type=float, default=0.0,
					help='clip at this multiple of the running median gradient norm (0 = off); exclusive with --grad-clip')
	ap.add_argument('--grad-clip-window', type=int, default=200, help='steps in the running-median window')
	ap.add_argument('--sac-lambda', type=float, default=0.0, help='C + SAC: in-batch SAC on the spatial encoder (0 = off)')
	ap.add_argument('--sac-sigma', type=float, default=0.613)
	ap.add_argument('--sac-temp', type=float, default=0.15)
	ap.add_argument('--spat-mpg-no-logL', action='store_true', help='spatial own MPG target without the log|L| channel')
	ap.add_argument('--val-pairing', choices=['fixed', 'random'], default='fixed',
					help='fixed: same validation triplets every pass (Pilot C seed 1 ran with random)')
	ap.add_argument('--lr-patience', type=int, default=2)
	ap.add_argument('--seed', type=int, default=1)
	ap.add_argument('--ckpt-every', type=int, default=200)
	ap.add_argument('--log-every', type=int, default=20)
	ap.add_argument('--max-steps', type=int, default=None)
	ap.add_argument('--num-workers', type=int, default=4)
	ap.add_argument('--snapshot-epochs', type=int, nargs='*', default=[2, 5])
	ap.add_argument('--device', default='cuda')
	args = ap.parse_args()
	os.makedirs(args.exp_dir, exist_ok=True)
	device = args.device
	torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

	n_scenes = int(np.load(args.tar + '.meta.npz')['n_scenes'])
	ds_train = PairedDataset(args.tar, 0, n_scenes - args.val_scenes, args.num_train)
	ds_val = PairedDataset(args.tar, n_scenes - args.val_scenes, n_scenes, args.num_val,
						   fixed_seed=12345 if args.val_pairing == 'fixed' else None)
	print(f'[data] {n_scenes} scenes x {ds_train.k}: train {len(ds_train)} targets, val {len(ds_val)}', flush=True)
	kw = dict(num_workers=args.num_workers, pin_memory=device == 'cuda', persistent_workers=args.num_workers > 0)
	val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.bs, shuffle=False, **kw)
	steps_per_epoch = len(ds_train) // args.bs

	stft_mod = at_module.STFT(win_len=WIN_LEN, win_shift_ratio=WIN_SHIFT_RATIO, nfft=NFFT).to(device)
	model = SwapPretrain(args.mask_cols, args.recon_lambda, args.classif_lambda, args.swap_lambda, args.own_scale,
						 ctr_lambda=args.ctr_lambda, ctr_temp=args.ctr_temp, spat_mpg_no_logL=args.spat_mpg_no_logL,
						 align_lambda=args.align_lambda, sac_lambda=args.sac_lambda, sac_sigma=args.sac_sigma,
						 sac_temp=args.sac_temp).to(device)
	n_par = {k: sum(p.numel() for p in s.parameters()) / 1e6 for k, s in model.streams.items()}
	n_par['decoder'] = sum(p.numel() for p in model.decoder.parameters()) / 1e6
	print(f'[model] ctr x{args.ctr_lambda} (temp {args.ctr_temp}), align x{args.align_lambda}, grad clip {args.grad_clip}, '
		  f'(median x{args.grad_clip_median_mult} over {args.grad_clip_window}), '
		  f'spat MPG no logL {args.spat_mpg_no_logL}, val pairing {args.val_pairing}, '
		  f'SAC x{args.sac_lambda} (sigma {args.sac_sigma}, temp {args.sac_temp})', flush=True)
	print(f'[model] swap x{args.swap_lambda}, own x{args.own_scale} (MPG {args.recon_lambda}, MPC {args.classif_lambda}, '
		  f'mask {args.mask_cols}/{jp.N_COLS}), params (M) { {k: round(v, 1) for k, v in n_par.items()} }', flush=True)
	opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], args.lr, weight_decay=5e-8, betas=(0.95, 0.999))
	sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=args.lr_patience)
	ckargs = argparse.Namespace(**vars(args), streams='joint', fusion=False)  # what save_encoders records

	assert not (args.grad_clip > 0 and args.grad_clip_median_mult > 0), 'use --grad-clip or --grad-clip-median-mult, not both'
	gn_window = collections.deque(maxlen=args.grad_clip_window)  # raw gradient norms of the previous steps
	latest = os.path.join(args.exp_dir, 'latest.pth')
	step, best_val = 0, float('inf')
	if os.path.isfile(latest):
		st = torch.load(latest, map_location='cpu', weights_only=False)
		model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
		step, best_val = st['step'], st['best_val']
		torch.set_rng_state(st['rng'][0]); np.random.set_state(st['rng'][1]); random.setstate(st['rng'][2])
		if st.get('cuda_rng') is not None and torch.cuda.is_available():  # saved since 2026-11-21 (older states lack it)
			torch.cuda.set_rng_state_all(st['cuda_rng'])
		gn_window.extend(st.get('gn_window', []))
		model.feature_stats = st.get('feature_stats', {})
		assert args.sac_lambda == 0 or model.feature_stats, 'SAC on but latest.pth has no feature statistics'
		print(f'[resume] step {step} = epoch {step // steps_per_epoch}, batch {step % steps_per_epoch}/{steps_per_epoch}', flush=True)
	else:
		stat_loader = torch.utils.data.DataLoader(_First(ds_train), batch_size=args.bs, shuffle=True,
												  generator=torch.Generator().manual_seed(args.seed), num_workers=args.num_workers)
		for k, (mean, std) in jp.input_stats(model, stft_mod, stat_loader, device).items():
			s = model.streams[k]
			if k == 'spec':
				mean, std = mean.mean().view(1), std.mean().view(1)
			s.in_mean.copy_(mean.view_as(s.in_mean)); s.in_std.copy_(std.view_as(s.in_std))
			print(f'[stats] {k} input mean {mean.tolist()} std {std.tolist()}', flush=True)
		if args.sac_lambda > 0:
			model.feature_stats = sac_feature_stats(stft_mod, stat_loader, device)
			for k in SAC_FEATURES:
				print(f"[stats] SAC {k} mean {model.feature_stats[k]['mean']:.6f} std {model.feature_stats[k]['std']:.6f}", flush=True)
	with open(os.path.join(args.exp_dir, 'args.json'), 'w') as f:
		json.dump(vars(args), f, indent=1)

	def save_latest():
		save_atomic({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
					 'step': step, 'best_val': best_val,
					 'rng': (torch.get_rng_state(), np.random.get_state(), random.getstate()),
					 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
					 'gn_window': list(gn_window), 'feature_stats': model.feature_stats}, latest)

	logf = open(os.path.join(args.exp_dir, 'log.jsonl'), 'a')
	gnf = open(os.path.join(args.exp_dir, 'grad_norm.txt'), 'a')  # step <TAB> total gradient norm before clipping
	params = [p for p in model.parameters() if p.requires_grad]
	gmax, gclip, thr = 0.0, 0, float('inf')
	clipping = args.grad_clip > 0 or args.grad_clip_median_mult > 0
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
											 drop_last=True, num_workers=args.num_workers, pin_memory=device == 'cuda',
											 generator=torch.Generator().manual_seed(args.seed * 7919 + step))
		for sig3, ids in loader:
			out = model(sig3.to(device, non_blocking=True), stft_mod, ids=ids.to(device))
			if not torch.isfinite(out['loss']):
				raise RuntimeError(f'non-finite loss at step {step}: { {k: float(v) for k, v in out.items()} }')
			opt.zero_grad(set_to_none=True)
			out['loss'].backward()
			if args.grad_clip > 0:
				thr = args.grad_clip
			elif args.grad_clip_median_mult > 0 and len(gn_window) >= 20:
				thr = args.grad_clip_median_mult * float(np.median(gn_window))
			else:
				thr = float('inf')
			gn = float(torch.nn.utils.clip_grad_norm_(params, thr))
			gn_window.append(gn)
			if not np.isfinite(gn):
				raise RuntimeError(f'non-finite gradient norm at step {step}')
			opt.step()
			step += 1
			gnf.write(f'{step}\t{gn:.5g}\n')
			out['grad_norm'] = gn
			gmax, gclip = max(gmax, gn), gclip + int(gn > thr)
			for k, v in out.items():
				acc[k] = acc.get(k, 0.0) + float(v)
			if step % args.log_every == 0:
				rec = {'step': step, 'epoch': step / steps_per_epoch, 'lr': opt.param_groups[0]['lr'],
					   's_per_step': (time.time() - t0) / args.log_every, **{k: v / args.log_every for k, v in acc.items()}}
				rec['grad_norm_max'] = gmax
				if clipping:
					rec['grad_clip_frac'] = gclip / args.log_every
					rec['grad_clip_thr'] = thr
				gmax, gclip = 0.0, 0
				gnf.flush()
				if step == args.log_every and torch.cuda.is_available():
					rec['max_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
				print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + '\n'); logf.flush()
				acc, t0 = {}, time.time()
			if step % steps_per_epoch == 0 or step == total_steps:
				val = evaluate(model, stft_mod, val_loader, device)
				rec = {'step': step, 'epoch': step / steps_per_epoch, 'val': val}
				print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + '\n'); logf.flush()
				sched.step(val['loss'])
				ep = step // steps_per_epoch
				if step % steps_per_epoch == 0 and ep in args.snapshot_epochs:
					model.save_encoders(ckargs, f'epoch{ep}')
				if val['loss'] < best_val:
					best_val = val['loss']
					model.save_encoders(ckargs, 'best')
				save_latest()
			elif step % args.ckpt_every == 0:
				save_latest()
			if step >= total_steps:
				break
	model.save_encoders(ckargs, 'final')
	save_latest()
	print(f'[done] {step} steps, best val loss {best_val:.4f}', flush=True)


if __name__ == '__main__':
	main()
