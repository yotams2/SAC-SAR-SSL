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
	C2 + head (--ctr-head, agreed 2026-11-21 as the diagnostic if C2's SC2 / SID drop; triggered 11-23 by the SID drop):
	a small 2-layer head per encoder, Linear(768, 768)-GELU-Linear(768, 256), on the mean-pooled embedding before the
	normalisation, for the InfoNCE (and alignment) only: anchors, positives and hard negatives all pass through it. The
	probes still read the encoder's pooled embedding. Off by default (built only when on).
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
	V1 (Agent Handoff 23 Nov; privileged teachers, paired v2 only): targets from a dry-speech content teacher (mono
	LibriSpeech SSAMBA on the dry utterance: MicSig/v1_teacher_amba_s1.npy, (12500, 8, 16, 768), z-scored per dimension with
	training-set statistics, .stats.npz; not instance norm, which would remove speaker information) and a room teacher (the
	two-channel RIR MAE, rir_encoder.py: one 384-d embedding per paired-v2 scene, z-scored with emb_stats.npz).
	V1a (--lat-lambda > 0): direct cosine regression, one linear predictor per encoder: spectral tokens (unmasked swap pass,
	8 x 16, frequency-major as the teacher) -> the teacher tokens of the spectral input's utterance; spatial pooled embedding
	-> the RIR embedding of the spatial input's scene. Logged: lat_spec, lat_spat.
	V1b (--tsac-lambda-spec / --tsac-lambda-spat > 0): in-batch SAC (CWCL, model.SARSSL._calc_single_feat_sim_loss, the
	C + SAC implementation) per encoder with only the cue source swapped: the cues are the L2-normalised z-scored teacher
	embeddings (spectral: the content teacher mean-pooled over its 128 tokens, of the spectral input's utterance; spatial:
	the RIR embedding of the spatial input's scene), so the Gaussian soft weight exp(-|c_i - c_j|^2 / sigma^2) =
	exp(-(2 - 2 cos(c_i, c_j)) / sigma^2) is a function of the teachers' pairwise cosine similarity. No predictor head:
	the loss is on the encoder's own mean-pooled embedding (the one the probes read). Sigma / temperature per encoder from
	the ESS pipeline on the teacher cues (v1b_sigma_tau.py). Logged: tsac_<enc> + its uniformity / top-10 alignment /
	effective rank. Cues reduced (11-23 revision, --tsac-pca-spec / --tsac-pca-spat): centre -> PCA fitted on the training-set
	teacher embeddings -> k components (95% variance, or by probe preservation on the RIR side; v1b_pca.py), no whitening ->
	L2-normalise. Oracle teacher B (analysis row "oracle SAC", 11-23): --v1-rir-dir pointing at a dir with
	emb_paired_v2.npy = the z-scored simulator parameters (rir_ceiling.py) and identity emb_stats.npz, no PCA, and
	--tsac-spat-cue euclid (Euclidean distance on the 6 z-scored parameters). --grad-share-every N logs every N steps each loss term's share of each encoder's gradient.
	--spat-mpg-no-logL: the spatial encoder's own MPG target without the log|L| channel (input unchanged).
	V2 (--cka-lambda > 0, agreed 2026-11-23; post-sprint candidate): + independence penalty between the two encoders on
	the SAME clip (the target x(s, r), the C2 anchors; encoded with gradient, reused from C2's pass when C2 is on):
	debiased linear CKA (unbiased HSIC estimator, Song et al. 2012; as in Nguyen et al. 2021) between the mean-pooled
	embeddings, batch-level. Unbiased because at batch 128 vs 768 dims the plain estimator is biased upward. Penalty =
	max(CKA, 0) (the debiased estimate is ~0 under independence and can dip below). Logged: cka, emb_std_tgt_{spec,spat}.
	--lr-schedule cosine (agreed 2026-11-23 as the recipe for all full runs): linear warmup over --warmup-frac of the
	steps from 0, then cosine decay to --min-lr-ratio x lr at the last step; stepped per optimiser step. Default
	'plateau' = the ReduceLROnPlateau of every run so far (unchanged).
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
				 sac_lambda=0.0, sac_sigma=0.613, sac_temp=0.15, cka_lambda=0.0, ctr_head=False, lat_lambda=0.0, tsac=None, d_rir=384):
		super().__init__('joint', False, 0.0, mask_cols, recon_lambda, classif_lambda, use_checkpointing)
		self.swap_lambda, self.own_scale = swap_lambda, own_scale
		self.ctr_lambda, self.ctr_temp, self.spat_mpg_no_logL = ctr_lambda, ctr_temp, spat_mpg_no_logL
		self.align_lambda = align_lambda
		self.ctr_head = nn.ModuleDict({k: nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 256))
									   for k in ('spec', 'spat')}) if ctr_head else None
		self.cka_lambda = cka_lambda
		self.P = 16 * 16  # values per patch per channel
		self.decoder = SwapDecoder(n_queries=16 * 16, out_dim=4 * self.P)
		self.sac_lambda, self.sac_sigma, self.sac_temp = sac_lambda, sac_sigma, sac_temp
		D = 768
		self.feat_sim_proj = nn.Sequential(nn.Linear(D, D), nn.BatchNorm1d(D), nn.ReLU(), nn.Linear(D, 128)) if sac_lambda > 0 else None
		self.feature_stats = {}  # SAC z-norm stats; read by SARSSL._calc_single_feat_sim_loss with the two below
		self.normalization_mode, self.factorized_sac = 'z_norm', False
		self._stft_spat = None
		self.lat_lambda = lat_lambda
		self.tsac = tsac or {}  # {encoder: (lambda, sigma, temperature)}, V1b
		self.lat_pred = nn.ModuleDict({'spec': nn.Linear(D, D), 'spat': nn.Linear(D, d_rir)}) if lat_lambda > 0 else None
		self.v1 = None  # teacher targets (set_v1_targets; not in the state dict: reloaded from the run's args)

	@property
	def v1_on(self):
		return self.lat_lambda > 0 or any(v[0] > 0 for v in self.tsac.values())

	@staticmethod
	def pca_cues(x, pca, l2=True):
		""" V1b cues (11-23 revision): centre with the PCA's training-set mean, project on its k components (no
			whitening), L2-normalise. pca = npz from v1b_pca.py; None = the full embedding, L2-normalised. """
		if pca is None:
			return F.normalize(x, dim=-1) if l2 else x
		z = np.load(pca)
		mean = torch.tensor(z['mean'], dtype=torch.float32, device=x.device)
		comp = torch.tensor(z['components'], dtype=torch.float32, device=x.device)  # (k, d), rows = principal axes
		y = (x - mean) @ comp.T
		return F.normalize(y, dim=-1) if l2 else y

	def set_v1_targets(self, teacher, teacher_stats, rir_dir, device, chunk=500, pca_spec=None, pca_spat=None, spat_l2=True, cue_file_spat=None):
		""" teacher tokens (n_utt, 8, 16, 768) fp16 (CPU, RAM; V1a only), their z-score, the pooled + normalised
			content cues (n_utt, 768) and the z-scored RIR embeddings (n_scenes, d) + normalised cues (GPU). """
		st = np.load(teacher_stats)
		mean, std = (torch.tensor(st[k], dtype=torch.float32, device=device) for k in ('mean', 'std'))
		tok = np.load(teacher, mmap_mode='r')
		pooled = []
		for b in range(0, len(tok), chunk):
			x = torch.from_numpy(np.ascontiguousarray(tok[b:b + chunk])).to(device).float().flatten(1, 2)
			pooled.append(((x - mean) / std).mean(dim=1))
		pooled = torch.cat(pooled)
		rir = torch.tensor(np.load(os.path.join(rir_dir, 'emb_paired_v2.npy')), dtype=torch.float32, device=device)
		rs = np.load(os.path.join(rir_dir, 'emb_stats.npz'))
		rir = (rir - torch.tensor(rs['mean'], device=device)) / torch.tensor(rs['std'], device=device)
		cue_spat = (torch.tensor(np.load(cue_file_spat), dtype=torch.float32, device=device) if cue_file_spat
					else self.pca_cues(rir, pca_spat, l2=spat_l2))  # cue file: precomputed final cues (v1b_cue_check.py split cue)
		self.v1 = {'tok': torch.from_numpy(np.load(teacher)) if self.lat_lambda > 0 else None, 'tok_mean': mean, 'tok_std': std,
				   'cue_spec': self.pca_cues(pooled, pca_spec), 'rir': rir, 'cue_spat': cue_spat}
		return {'n_utt': len(pooled), 'n_scenes': len(rir), 'd_rir': rir.shape[1], 'k_spec': self.v1['cue_spec'].shape[1],
				'k_spat': self.v1['cue_spat'].shape[1],
				'cue_spec_mean_cos': float((self.v1['cue_spec'][:2000] @ self.v1['cue_spec'][:2000].T).mean()),
				'cue_spat_mean_cos': float((self.v1['cue_spat'][:2000] @ self.v1['cue_spat'][:2000].T).mean())}

	def v1_losses(self, toks, ids):
		""" V1a / V1b terms. ids (B, 3, 2) [utterance, scene] of (target, spectral input, spatial input). """
		import model as at_model
		assert self.v1 is not None, 'V1 on but set_v1_targets was not called'
		u, r = ids[:, 1, 0], ids[:, 2, 1]  # the spectral input's utterance (= the target's), the spatial input's scene (= the target's)
		e = {k: toks[k][:, 1:].float().mean(dim=1) for k in ('spec', 'spat')}  # what the probes read (before normalisation)
		out, loss = {}, 0.0
		if self.lat_lambda > 0:
			t = self.v1['tok'][u.cpu()].to(e['spec'].device, non_blocking=True).float().flatten(1, 2)  # (B, 128, 768)
			t = (t - self.v1['tok_mean']) / self.v1['tok_std']
			h = toks['spec'][:, 1:].float()
			assert h.shape[1] == t.shape[1], (h.shape, t.shape)
			out['lat_spec'] = (1 - F.cosine_similarity(self.lat_pred['spec'](h), t, dim=-1)).mean()
			out['lat_spat'] = (1 - F.cosine_similarity(self.lat_pred['spat'](e['spat']), self.v1['rir'][r], dim=-1)).mean()
			loss = loss + self.lat_lambda * (out['lat_spec'] + out['lat_spat'])
		for k, idx in (('spec', u), ('spat', r)):
			lam, sigma, temp = self.tsac.get(k, (0.0, None, None))
			if lam <= 0:
				continue
			c = self.v1[f'cue_{k}'][idx]
			feats = {f'_v1_{i}': c[:, i] for i in range(c.shape[1])}  # no feature_stats entry: used as they are
			l, met = at_model.SARSSL._calc_single_feat_sim_loss(self, e[k], feats, list(feats), temp, sigma, 1e-8, False)
			out[f'tsac_{k}'] = l
			for n, v in met['diagnostics']['Global'].items():
				out[f'tsac_{k}_{n}'] = float(v)
			loss = loss + lam * l
		return out, loss

	def term_weights(self):
		""" {term: (output key(s), weight)} of the loss, for the gradient-share diagnostic. """
		w = {'swap': (['swap'], self.swap_lambda)}
		if self.lat_lambda > 0:
			w['lat'] = (['lat_spec', 'lat_spat'], self.lat_lambda)
		for k, (lam, _, _) in self.tsac.items():
			if lam > 0:
				w[f'tsac_{k}'] = ([f'tsac_{k}'], lam)
		return w

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

	@staticmethod
	def debiased_cka(x, y):
		""" Debiased linear CKA between (n, d1) and (n, d2) (unbiased HSIC, Song et al. 2012). ~0 if independent. """
		x, y = x.float(), y.float()
		n = x.shape[0]
		def hsic(K, L):
			K, L = K.clone(), L.clone()
			K.fill_diagonal_(0); L.fill_diagonal_(0)
			one = torch.ones(n, 1, device=K.device)
			t = (K * L).sum() + (one.T @ K @ one) * (one.T @ L @ one) / ((n - 1) * (n - 2)) \
				- 2.0 / (n - 2) * (one.T @ K @ L @ one)
			return t.squeeze() / (n * (n - 3))
		K, L = x @ x.T, y @ y.T
		return hsic(K, L) / (hsic(K, K) * hsic(L, L)).clamp_min(1e-12).sqrt()

	def cka_loss(self, pos):
		""" V2: pos = both encoders on the target clip (with gradient). """
		e = {k: pos[k][:, 1:].float().mean(dim=1) for k in ('spec', 'spat')}
		out = {f'emb_std_tgt_{k}': v.std(dim=0).mean().detach() for k, v in e.items()}
		out['cka'] = self.debiased_cka(e['spec'], e['spat'])
		return out

	def ctr_losses(self, toks, Xpos, Xneg, ids, pos=None):
		if pos is None:
			pos = self.encode(Xpos, None, fusion=False)
		with torch.no_grad():
			neg = self.encode(Xneg, None, fusion=False)
		B = ids.shape[0]
		ar = torch.arange(B, device=ids.device)
		out = {}
		for k, vp, vn, f in (('spec', 1, 2, 0), ('spat', 2, 1, 1)):  # anchor = view 0 (target); f: 0 utterance, 1 scene
			va = 0
			out[f'emb_std_{k}'] = pos[k][:, 1:].float().mean(dim=1).std(dim=0).mean().detach()
			if self.ctr_head is None:
				a, p, h = self.pooled(pos[k]), self.pooled(toks[k]), self.pooled(neg[k])  # target, own input, hard negative
			else:  # C2 + head: the head on the pooled embedding, then normalised
				hd = lambda x: F.normalize(self.ctr_head[k](x[:, 1:].float().mean(dim=1)), dim=-1)
				a, p = hd(pos[k]), hd(toks[k])
				with torch.no_grad():
					h = hd(neg[k])
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
		use_cka = self.cka_lambda > 0
		spec, spat, tgt, w, Xpos, Xneg = self.views(sig3, stft_mod, extra=use_ctr or use_cka)
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
		if self.v1_on:
			assert ids is not None, 'V1 needs the view ids'
			v1o, v1l = self.v1_losses(toks, ids)
			out.update(v1o)
			loss = loss + v1l
		pos = None
		if use_cka:  # both encoders on the target clip; shared with C2's anchor pass
			pos = self.encode(Xpos, None, fusion=False)
			ck = self.cka_loss(pos)
			out.update(ck)
			loss = loss + self.cka_lambda * ck['cka'].clamp_min(0)
		if use_ctr:
			assert ids is not None, 'the contrastive term needs the view ids'
			ct = self.ctr_losses(toks, Xpos, Xneg, ids, pos=pos)
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
def grad_share(model, out):
	""" Each weighted loss term's gradient on each encoder vs the total: share = |g_term| / |g_total|, and the cosine
		between g_term and the rest of the gradient (g_total - g_term). Non-reentrant checkpointing, so autograd.grad
		works; the graph is retained for the step's own backward. """
	res = {}
	terms = {n: w * sum(out[k] for k in keys) for n, (keys, w) in model.term_weights().items()}
	for enc in ('spec', 'spat'):
		P = [p for p in model.streams[enc].parameters() if p.requires_grad]
		flat = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, P)])
		gt = flat(torch.autograd.grad(out['loss'], P, retain_graph=True, allow_unused=True))
		res[f'gs_{enc}_total_norm'] = float(gt.norm())
		for n, t in terms.items():
			if not t.requires_grad:
				continue
			g = flat(torch.autograd.grad(t, P, retain_graph=True, allow_unused=True))
			rest = gt - g
			res[f'gs_{enc}_{n}'] = float(g.norm() / gt.norm().clamp_min(1e-12))
			res[f'gs_{enc}_{n}_cos_rest'] = float((g @ rest) / (g.norm() * rest.norm()).clamp_min(1e-12))
	return res


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
	ap.add_argument('--ctr-head', action='store_true', help='C2 + head: 2-layer head per encoder before the InfoNCE (diagnostic)')
	ap.add_argument('--align-lambda', type=float, default=0.0, help='C2b: weight of each encoder\'s 1 - cos(anchor, positive)')
	ap.add_argument('--grad-clip', type=float, default=0.0, help='clip the total gradient norm at this value (0 = off)')
	ap.add_argument('--grad-clip-median-mult', type=float, default=0.0,
					help='clip at this multiple of the running median gradient norm (0 = off); exclusive with --grad-clip')
	ap.add_argument('--grad-clip-window', type=int, default=200, help='steps in the running-median window')
	ap.add_argument('--sac-lambda', type=float, default=0.0, help='C + SAC: in-batch SAC on the spatial encoder (0 = off)')
	ap.add_argument('--sac-sigma', type=float, default=0.613)
	ap.add_argument('--sac-temp', type=float, default=0.15)
	ap.add_argument('--cka-lambda', type=float, default=0.0, help='V2: debiased-CKA penalty between the encoders on the target (0 = off)')
	ap.add_argument('--lr-schedule', choices=['plateau', 'cosine'], default='plateau',
					help='plateau: ReduceLROnPlateau on val loss (all runs before 11-23); cosine: warmup + cosine decay per step')
	ap.add_argument('--warmup-frac', type=float, default=0.05, help='cosine: fraction of the steps for linear warmup')
	ap.add_argument('--min-lr-ratio', type=float, default=0.01, help='cosine: final lr / lr')
	ap.add_argument('--v1-teacher', default='/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/v1_teacher_amba_s1.npy')
	ap.add_argument('--v1-teacher-stats', default='/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/v1_teacher_amba_s1.stats.npz')
	ap.add_argument('--v1-rir-dir', default=None, help='V1: rir_encoder.py output dir (emb_paired_v2.npy, emb_stats.npz)')
	ap.add_argument('--lat-lambda', type=float, default=0.0, help='V1a: weight of the two cosine regressions (0 = off)')
	for _k in ('spec', 'spat'):
		ap.add_argument(f'--tsac-lambda-{_k}', type=float, default=0.0, help=f'V1b: teacher-cue SAC on the {_k} encoder (0 = off)')
		ap.add_argument(f'--tsac-sigma-{_k}', type=float, default=None)
		ap.add_argument(f'--tsac-temp-{_k}', type=float, default=None)
	ap.add_argument('--tsac-pca-spec', default=None, help='V1b: PCA npz (v1b_pca.py) for the spectral cues (None = full 768-d)')
	ap.add_argument('--tsac-pca-spat', default=None, help='V1b: PCA npz (v1b_pca.py) for the spatial cues (None = full 384-d)')
	ap.add_argument('--tsac-spat-cue', choices=['cosine', 'euclid'], default='cosine',
					help='V1b spatial cues: L2-normalised (cosine; learned teachers) or as they are (euclid; the oracle '
						 'parameter teacher B, like the z-normed handcrafted SAC cues)')
	ap.add_argument('--tsac-cue-file-spat', default=None, help='V1b: precomputed final spatial cues (n_scenes, k), used as they are')
	ap.add_argument('--grad-share-every', type=int, default=0, help='log each term\'s share of each encoder\'s gradient every N steps')
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
						 sac_temp=args.sac_temp, cka_lambda=args.cka_lambda, ctr_head=args.ctr_head, lat_lambda=args.lat_lambda,
						 tsac={'spec': (args.tsac_lambda_spec, args.tsac_sigma_spec, args.tsac_temp_spec),
							   'spat': (args.tsac_lambda_spat, args.tsac_sigma_spat, args.tsac_temp_spat)},
						 d_rir=(np.load(os.path.join(args.v1_rir_dir, 'emb_paired_v2.npy'), mmap_mode='r').shape[1]
								if args.v1_rir_dir else 384)).to(device)  # 11-24: A′ embeddings are 768-d (the RIR MAEs were 384)
	if model.v1_on:
		for k in ('spec', 'spat'):
			lam, sg, tp = model.tsac[k]
			assert lam <= 0 or (sg is not None and tp is not None), f'V1b {k}: give --tsac-sigma-{k} and --tsac-temp-{k}'
		assert args.v1_rir_dir, 'V1 needs --v1-rir-dir (rir_encoder.py output)'
		assert '_v2' in os.path.basename(args.tar), 'V1: the RIR embeddings are of the paired v2 scenes'
		info = model.set_v1_targets(args.v1_teacher, args.v1_teacher_stats, args.v1_rir_dir, device,
									pca_spec=args.tsac_pca_spec, pca_spat=args.tsac_pca_spat,
									spat_l2=args.tsac_spat_cue == 'cosine', cue_file_spat=args.tsac_cue_file_spat)
		assert info['n_scenes'] == n_scenes, (info, n_scenes)
		print(f'[v1] lat x{args.lat_lambda}; tsac {model.tsac}; targets {info}', flush=True)
	n_par = {k: sum(p.numel() for p in s.parameters()) / 1e6 for k, s in model.streams.items()}
	n_par['decoder'] = sum(p.numel() for p in model.decoder.parameters()) / 1e6
	print(f'[model] ctr x{args.ctr_lambda} (temp {args.ctr_temp}), align x{args.align_lambda}, grad clip {args.grad_clip}, '
		  f'(median x{args.grad_clip_median_mult} over {args.grad_clip_window}), '
		  f'spat MPG no logL {args.spat_mpg_no_logL}, val pairing {args.val_pairing}, '
		  f'SAC x{args.sac_lambda} (sigma {args.sac_sigma}, temp {args.sac_temp}), CKA x{args.cka_lambda}, '
		  f'lr schedule {args.lr_schedule}' + (f' (warmup {args.warmup_frac}, min ratio {args.min_lr_ratio})'
		                                       if args.lr_schedule == 'cosine' else ''), flush=True)
	print(f'[model] swap x{args.swap_lambda}, own x{args.own_scale} (MPG {args.recon_lambda}, MPC {args.classif_lambda}, '
		  f'mask {args.mask_cols}/{jp.N_COLS}), params (M) { {k: round(v, 1) for k, v in n_par.items()} }', flush=True)
	opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], args.lr, weight_decay=5e-8, betas=(0.95, 0.999))
	total_steps = steps_per_epoch * args.nepoch
	if args.max_steps:
		total_steps = min(total_steps, args.max_steps)
	if args.lr_schedule == 'cosine':
		n_sched = steps_per_epoch * args.nepoch  # the full run, independent of --max-steps (pieces / resumes)
		n_warm = max(1, int(round(args.warmup_frac * n_sched)))
		def lr_at(s):  # multiplier of args.lr after s optimiser steps
			if s < n_warm:
				return (s + 1) / n_warm
			t = min(1.0, (s - n_warm) / max(1, n_sched - n_warm))
			return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * t))
		sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
	else:
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
			if args.grad_share_every and step % args.grad_share_every == 0:
				gs = grad_share(model, out)
				print(json.dumps({'step': step, **gs}), flush=True)
				logf.write(json.dumps({'step': step, 'grad_share': gs}) + '\n'); logf.flush()
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
			if args.lr_schedule == 'cosine':
				sched.step()
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
				if args.lr_schedule == 'plateau':
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
