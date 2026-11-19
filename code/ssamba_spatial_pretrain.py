"""
	Pretraining of the spatial SSAMBA encoder on the SAR-SSL simulated pretraining set (MicSig/simu, WSJ0 sources).

	Single encoder, the "ipd16" input that won the random-init probe (job 23594658):
		SAR-SSL's STFT (STFTLearner: Hann 512, hop 256, divided by the mean |STFT| of mic 0, bins 1..256), turned into
		(log|L|, ILD, cos IPD, sin IPD) by ssamba_spatial.SSAMBASpatialDownstream.stft_frontend, 16x16 patches,
		256 x 256 per 4.112-s clip = 256 tokens. Each channel is then standardised with dataset statistics computed
		at the start of the run and saved in every checkpoint (the downstream wrapper applies the same ones).
	Objective (SSAMBA's own, as in sac-ssamba mono pretraining):
		recon_lambda * mpg + classif_lambda * mpc   (masked patch generation / classification, cluster masking)
		+ sac_lambda * SAC on the unmasked encoding (mean over patch tokens -> Linear-BN-ReLU-Linear(128)).
	SAC is the Interspeech one, computed with SAC-SAR-SSL's own code: the 5 spatial features (coherence low/mid/high,
	GCC-PHAT max, TDOA) from common.utils_module.spatial_features_from_stft on the same normalised STFT,
	z-norm with stats from calculate_feature_stats (50 batches), loss model.SARSSL._calc_single_feat_sim_loss.

	Data are read straight out of one uncompressed tar (scripts/stage_simu_pretrain.sbatch): /scratch200 is out of
	inodes, so the ~112k files are never unpacked. An index of member offsets is cached next to the tar.

	Resumable: latest.pth (full state) every --ckpt-every steps and at each epoch end; a requeued job continues from it.
	Outputs in --exp-dir: latest.pth, encoder_epoch<N>.pth (--snapshot-epochs), best_encoder.pth (lowest val loss), encoder_final.pth,
	log.jsonl. Encoder checkpoints hold {'state_dict': spatial-encoder weights with the prefix that
	ssamba_spatial.load_spatial_state_dict expects, 'meta': input settings + channel statistics}.
"""

import argparse
import io
import json
import math
import os
import random
import sys
import tarfile
import time
from types import SimpleNamespace

import numpy as np
import soundfile
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.utils_module as at_module  # noqa: E402
import ssamba_spatial  # noqa: E402

FS = 16000
WIN_LEN, NFFT, WIN_SHIFT_RATIO = 512, 512, 0.5
FRE_RANGE = range(1, NFFT // 2 + 1)  # STFTLearner with fre_used_ratio=1
T_SEC = 4.112
NT = int((T_SEC * FS - WIN_LEN * (1 - WIN_SHIFT_RATIO)) / (WIN_LEN * WIN_SHIFT_RATIO))  # 256, as in run_pretrain.py
SAC_FEATURES = ['coherence_low', 'coherence_mid', 'coherence_high', 'gcc_phat_max', 'tdoa']


# ---------------------------------------------------------------- data
def tar_index(tar_path):
	""" {member name: (data offset, size)} for every .wav in the tar, cached as <tar>.index.json. """
	cache = tar_path + '.index.json'
	if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(tar_path):
		with open(cache) as f:
			return {k: tuple(v) for k, v in json.load(f).items()}
	t0 = time.time()
	idx = {}
	with tarfile.open(tar_path, 'r:') as tf:
		for m in tf:
			if m.isfile() and m.name.endswith('.wav'):
				idx[m.name] = (m.offset_data, m.size)
	tmp = cache + f'.tmp{os.getpid()}'
	with open(tmp, 'w') as f:
		json.dump(idx, f)
	os.replace(tmp, cache)
	print(f'[data] indexed {len(idx)} wav in {tar_path} ({time.time() - t0:.0f} s)', flush=True)
	return idx


class TarMicSigDataset(torch.utils.data.Dataset):
	""" FixMicSigDataset (load_anno=False) over one split of the tar: returns the (nsample, nch) float32 signal. """
	def __init__(self, tar_path, split, dataset_sz=None):
		idx = tar_index(tar_path)
		names = [n for n in idx if n.startswith(split + '/') and not n.endswith('_dp.wav')]
		names.sort(key=lambda n: int(os.path.basename(n)[:-4]))
		if dataset_sz is not None:
			names = names[:dataset_sz]
		if not names:
			raise RuntimeError(f'no {split}/*.wav in {tar_path}')
		self.tar_path, self.items = tar_path, [idx[n] for n in names]
		self._fh = None  # one handle per worker process, opened lazily

	def __len__(self):
		return len(self.items)

	def __getitem__(self, i):
		if self._fh is None:
			self._fh = open(self.tar_path, 'rb')
		off, size = self.items[i]
		self._fh.seek(off)
		sig, fs = soundfile.read(io.BytesIO(self._fh.read(size)), dtype='float32')
		assert fs == FS, f'fs {fs}'
		return sig


def stft_reim(stft_mod, sig, eps=1e-6):
	""" STFTLearner.data_preprocess for 2 mics: (nb, nsample, 2) -> raw STFT (nb, 2, 257, nt) and the normalised,
		band-limited (nb, 2, 256, nt, 2) that SAR-SSL models (and the downstream wrapper) receive.
	"""
	stft = stft_mod(signal=sig).permute(0, 3, 1, 2)  # (nb, nch, nf, nt)
	mag = torch.abs(stft[:, 0:1])
	mean_value = mag.reshape(mag.shape[0], -1).mean(dim=1)[:, None, None, None]
	stft_n = stft / (mean_value + eps)
	reim = torch.view_as_real(stft_n)[:, :, FRE_RANGE]  # (nb, 2, 256, nt, 2)
	return stft, reim


# ---------------------------------------------------------------- model
class SSAMBASpatialPretrain(nn.Module):
	def __init__(self, mask_patch, recon_lambda, classif_lambda, sac_lambda, sac_sigma, sac_temp, use_checkpointing=True,
				 sac_form='cwcl', sac_queue_size=1024, sac_queue_warmup=390):
		super().__init__()
		# Same module the downstream uses, so the input path is identical; only its encoder and frontend are used.
		self.ds = ssamba_spatial.SSAMBASpatialDownstream(
			ckpt_path=None, fill='none', input_mode='ipd', patch='16x16', n_stft_frames=NT,
			clip_seconds=T_SEC, use_checkpointing=use_checkpointing)
		self.encoder = self.ds.encoder
		for name, p in self.encoder.named_parameters():
			if name.split('.')[0] in ['cpredlayer', 'gpredlayer', 'mask_embed']:
				p.requires_grad = True  # the wrapper freezes the pretraining heads; here they train
			# (p_input_fdim / p_input_tdim are integer Parameters with requires_grad=False: leave them alone)
		del self.ds.mlp_head
		D = self.encoder.original_embedding_dim
		self.register_buffer('in_mean', torch.zeros(1, 4, 1, 1))
		self.register_buffer('in_std', torch.ones(1, 4, 1, 1))
		self.mask_patch, self.recon_lambda, self.classif_lambda = mask_patch, recon_lambda, classif_lambda
		self.sac_lambda, self.sac_sigma, self.sac_temp = sac_lambda, sac_sigma, sac_temp
		self.feat_sim_proj = nn.Sequential(nn.Linear(D, D), nn.BatchNorm1d(D), nn.ReLU(), nn.Linear(D, 128)) if sac_lambda > 0 else None
		self.feature_stats = {}
		self.normalization_mode, self.factorized_sac = 'z_norm', False  # read by SARSSL._calc_single_feat_sim_loss
		# SAC form (model.SARSSL._calc_single_feat_sim_loss reads these): 'cwcl' (original), 'sigmoid' (SigLIP-style,
		# learnable logit scale starting at 1/temp and bias set from the first batch), 'queue' (CWCL + cross-batch memory
		# of sac_queue_size detached embeddings and cues, used after sac_queue_warmup training steps).
		self.sac_form = sac_form
		if sac_form == 'sigmoid':
			self.sac_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / sac_temp)))
			self.sac_logit_bias = nn.Parameter(torch.tensor(0.0))
			self.register_buffer('sac_bias_set', torch.tensor(False))
		elif sac_form == 'queue':
			self.sac_queue_warmup = sac_queue_warmup
			self.register_buffer('sac_queue_z', torch.zeros(sac_queue_size, 128))
			self.register_buffer('sac_queue_c', torch.zeros(sac_queue_size, len(SAC_FEATURES)))
			self.register_buffer('sac_queue_ptr', torch.tensor(0))
			self.register_buffer('sac_queue_n', torch.tensor(0))
			self.register_buffer('sac_queue_calls', torch.tensor(0))
		else:
			assert sac_form == 'cwcl', sac_form

	def ipd(self, reim):
		X, _ = self.ds.stft_frontend(reim)  # (nb, 4, 256, 256)
		return X

	def forward(self, reim):
		X = (self.ipd(reim) - self.in_mean) / self.in_std
		out = {}
		loss = X.new_zeros(())
		if self.recon_lambda > 0:
			out['mpg'] = self.encoder.mpg(X, mask_patch=self.mask_patch, cluster=True)
			loss = loss + self.recon_lambda * out['mpg']
		if self.classif_lambda > 0:
			out['acc'], out['mpc'] = self.encoder.mpc(X, mask_patch=self.mask_patch, cluster=True)
			loss = loss + self.classif_lambda * out['mpc']
		if self.sac_lambda > 0:
			import model as at_model
			h = self.encoder._encode_with_mamba(X)
			z_proj = self.feat_sim_proj(h[:, 1:].mean(dim=1))
			with torch.no_grad():
				feats = at_module.spatial_features_from_stft(torch.view_as_complex(reim.contiguous()), sample_rate=FS)
			out['sac'], _ = at_model.SARSSL._calc_single_feat_sim_loss(
				self, z_proj, feats, SAC_FEATURES, self.sac_temp, self.sac_sigma, 1e-8, False)
			loss = loss + self.sac_lambda * out['sac']
			if self.sac_form == 'sigmoid':  # logged, to see whether the learnable scale / bias move
				out['sac_t'], out['sac_b'] = self.sac_logit_scale.exp().detach(), self.sac_logit_bias.detach()
		out['loss'] = loss
		return out

	def encoder_checkpoint(self, args):
		sd = {ssamba_spatial.SPATIAL_PREFIX + k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}
		meta = {'input': 'ipd', 'patch': '16x16', 'n_frames': NT, 'n_freq': 256,
				'in_mean': self.in_mean.flatten().tolist(), 'in_std': self.in_std.flatten().tolist(),
				'args': vars(args)}
		return {'state_dict': sd, 'meta': meta}


# ---------------------------------------------------------------- stats
@torch.no_grad()
def input_stats(model, stft_mod, loader, device, nbatch=50):
	s = torch.zeros(4, device=device, dtype=torch.float64)
	s2, n = torch.zeros_like(s), 0
	for i, sig in enumerate(loader):
		if i >= nbatch:
			break
		_, reim = stft_reim(stft_mod, sig.to(device))
		X = model.ipd(reim).double()
		s += X.sum(dim=(0, 2, 3)); s2 += (X ** 2).sum(dim=(0, 2, 3)); n += X[:, 0].numel()
	mean = s / n
	std = (s2 / n - mean ** 2).clamp_min(1e-12).sqrt()
	return mean.float(), std.float()


@torch.no_grad()
def sac_feature_stats(stft_mod, loader, device, nbatch=50):
	""" run_pretrain.calculate_feature_stats: raw (unnormalised, all-bin) STFT, max_ipd_freq 1500, 50 batches. """
	coll = {}
	for i, sig in enumerate(loader):
		if i >= nbatch:
			break
		stft, _ = stft_reim(stft_mod, sig.to(device))
		feats = at_module.spatial_features_from_stft(stft, FS, max_ipd_freq=1500.0)
		for k, v in feats.items():
			if isinstance(v, torch.Tensor) and v.dim() == 1 and not torch.is_complex(v):
				coll.setdefault(k, []).append(v)
	stats = {}
	for k, vs in coll.items():
		v = torch.cat(vs)
		stats[k] = {'mean': v.mean().item(), 'std': v.std().item()}
	return stats


# ---------------------------------------------------------------- train
def save_atomic(obj, path):
	tmp = path + '.tmp'
	torch.save(obj, tmp)
	os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, stft_mod, loader, device, seed=1234):
	model.eval()
	rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
				 np.random.get_state(), random.getstate())
	torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)  # same masks every evaluation
	tot, n = {}, 0
	for sig in loader:
		_, reim = stft_reim(stft_mod, sig.to(device))
		out = model(reim)
		for k, v in out.items():
			tot[k] = tot.get(k, 0.0) + float(v) * sig.shape[0]
		n += sig.shape[0]
	torch.set_rng_state(rng_state[0])
	if torch.cuda.is_available():
		torch.cuda.set_rng_state_all(rng_state[1])
	np.random.set_state(rng_state[2]); random.setstate(rng_state[3])
	model.train()
	return {k: v / n for k, v in tot.items()}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--tar', required=True)
	ap.add_argument('--exp-dir', required=True)
	ap.add_argument('--num-train', type=int, default=50000)
	ap.add_argument('--num-val', type=int, default=None)
	ap.add_argument('--bs', type=int, default=128)
	ap.add_argument('--lr', type=float, default=4e-4)
	ap.add_argument('--nepoch', type=int, default=30)
	ap.add_argument('--mask-patch', type=int, default=150, help='of 256 patches (mono SSAMBA: 300 of 512)')
	ap.add_argument('--recon-lambda', type=float, default=10.0)
	ap.add_argument('--classif-lambda', type=float, default=1.0)
	ap.add_argument('--sac-lambda', type=float, default=0.0)
	ap.add_argument('--sac-sigma', type=float, default=0.613)
	ap.add_argument('--sac-temp', type=float, default=0.15)
	ap.add_argument('--sac-form', default='cwcl', choices=['cwcl', 'sigmoid', 'queue'], help='SAC loss form (model.py)')
	ap.add_argument('--sac-queue-size', type=int, default=1024, help='queue: memory of past embeddings (8 batches of 128)')
	ap.add_argument('--sac-queue-warmup', type=int, default=390, help='queue: training steps before the memory is used (1 epoch)')
	ap.add_argument('--lr-patience', type=int, default=2)
	ap.add_argument('--seed', type=int, default=1)
	ap.add_argument('--ckpt-every', type=int, default=200)
	ap.add_argument('--log-every', type=int, default=20)
	ap.add_argument('--max-steps', type=int, default=None, help='stop after this many steps (smoke tests)')
	ap.add_argument('--num-workers', type=int, default=4)
	ap.add_argument('--snapshot-epochs', type=int, nargs='*', default=[2, 5, 10, 20])
	ap.add_argument('--device', default='cuda', help='cpu only for code checks')
	args = ap.parse_args()
	os.makedirs(args.exp_dir, exist_ok=True)
	device = args.device
	torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

	ds_train = TarMicSigDataset(args.tar, 'pretrain', args.num_train)
	ds_val = TarMicSigDataset(args.tar, 'preval', args.num_val)
	print(f'[data] train {len(ds_train)}, val {len(ds_val)}', flush=True)
	kw = dict(num_workers=args.num_workers, pin_memory=device == 'cuda', persistent_workers=args.num_workers > 0)
	val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.bs, shuffle=False, **kw)
	steps_per_epoch = len(ds_train) // args.bs

	stft_mod = at_module.STFT(win_len=WIN_LEN, win_shift_ratio=WIN_SHIFT_RATIO, nfft=NFFT).to(device)
	model = SSAMBASpatialPretrain(args.mask_patch, args.recon_lambda, args.classif_lambda, args.sac_lambda,
								  args.sac_sigma, args.sac_temp, sac_form=args.sac_form,
								  sac_queue_size=args.sac_queue_size, sac_queue_warmup=args.sac_queue_warmup).to(device)
	opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], args.lr, weight_decay=5e-8, betas=(0.95, 0.999))
	sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=args.lr_patience)

	latest = os.path.join(args.exp_dir, 'latest.pth')
	step, best_val = 0, float('inf')
	if os.path.isfile(latest):
		st = torch.load(latest, map_location='cpu', weights_only=False)
		model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
		model.feature_stats, step, best_val = st['feature_stats'], st['step'], st['best_val']
		torch.set_rng_state(st['rng'][0]); np.random.set_state(st['rng'][1]); random.setstate(st['rng'][2])
		print(f'[resume] step {step} = epoch {step // steps_per_epoch}, batch {step % steps_per_epoch}/{steps_per_epoch}', flush=True)
	else:
		stat_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.bs, shuffle=True,
												  generator=torch.Generator().manual_seed(args.seed), num_workers=args.num_workers)
		mean, std = input_stats(model, stft_mod, stat_loader, device)
		model.in_mean.copy_(mean.view(1, 4, 1, 1)); model.in_std.copy_(std.view(1, 4, 1, 1))
		print(f'[stats] input channel mean {mean.tolist()} std {std.tolist()}', flush=True)
		if args.sac_lambda > 0:
			model.feature_stats = sac_feature_stats(stft_mod, stat_loader, device)
			for k in SAC_FEATURES:
				print(f"[stats] {k} mean {model.feature_stats[k]['mean']:.6f} std {model.feature_stats[k]['std']:.6f}", flush=True)
	with open(os.path.join(args.exp_dir, 'args.json'), 'w') as f:
		json.dump(vars(args), f, indent=1)

	def save_latest():
		save_atomic({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
					 'feature_stats': model.feature_stats, 'step': step, 'best_val': best_val,
					 'rng': (torch.get_rng_state(), np.random.get_state(), random.getstate())}, latest)

	logf = open(os.path.join(args.exp_dir, 'log.jsonl'), 'a')
	total_steps = steps_per_epoch * args.nepoch
	if args.max_steps:
		total_steps = min(total_steps, args.max_steps)
	model.train()
	t0, acc = time.time(), {}
	while step < total_steps:
		epoch, start_b = step // steps_per_epoch, step % steps_per_epoch
		# epoch order fixed by (seed, epoch), so a resume continues the same permutation
		perm = torch.randperm(len(ds_train), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch))
		idx = perm[start_b * args.bs: steps_per_epoch * args.bs].tolist()
		loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_train, idx), batch_size=args.bs, shuffle=False,
											 drop_last=True, num_workers=args.num_workers, pin_memory=device == 'cuda')
		for sig in loader:
			_, reim = stft_reim(stft_mod, sig.to(device, non_blocking=True))
			out = model(reim)
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
				print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + '\n'); logf.flush()
				acc, t0 = {}, time.time()
			if step % steps_per_epoch == 0 or step == total_steps:
				val = evaluate(model, stft_mod, val_loader, device)
				ep = step // steps_per_epoch
				rec = {'step': step, 'epoch': step / steps_per_epoch, 'val': val}
				print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + '\n'); logf.flush()
				sched.step(val['loss'])
				ck = model.encoder_checkpoint(args)
				if ep in args.snapshot_epochs:  # ~0.4 GB each: only the epochs used for early reads
					save_atomic(ck, os.path.join(args.exp_dir, f'encoder_epoch{ep}.pth'))
				if val['loss'] < best_val:
					best_val = val['loss']
					save_atomic(ck, os.path.join(args.exp_dir, 'best_encoder.pth'))
				save_latest()
			elif step % args.ckpt_every == 0:
				save_latest()
			if step >= total_steps:
				break
	save_atomic(model.encoder_checkpoint(args), os.path.join(args.exp_dir, 'encoder_final.pth'))
	save_latest()
	print(f'[done] {step} steps, best val loss {best_val:.4f}', flush=True)


if __name__ == '__main__':
	main()
