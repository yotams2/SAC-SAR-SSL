"""
	Spatial SSAMBA encoder (binaural dual-encoder SAC, ../sac-ssamba/src/binaural_dual_stream_sac) as the upstream of
	run_downstream.py. Selected with `--upstream ssamba_spatial --ssamba-ckpt <audio_model.final.pth>`.

	Only `dual_encoder.spatial_encoder` is used; the spectral encoder, decoder and SAC heads are not built.
	The input is made the way the SSAMBA pretraining dataloader makes it (binaural_dataloader.wav2stft), not with
	SAR-SSL's STFT, because the encoder only ever saw that representation:
		- clips peak-normalised to 0.9 (same gain on both channels, so level differences between them stay),
		- STFT n_fft 512, hop 160, no window (torch.stft default, as in pretraining), bins 0..255,
		  stacked as (Re_L, Im_L, Re_R, Im_R), 1024 frames.
	Pretraining clips are 10 s (1001 real frames + 23 zero frames). SAR-SSL clips are 1.04 s (TDOA) or 4.112 s, so
	`fill` decides how the rest of the 10 s is filled:
		tile (default): repeat the clip up to 10 s, so the encoder sees the same amount of signal as in pretraining;
		zero: pad with zero frames.
	Either way only patches over real frames are mean-pooled (`token='all'`); `token='cls'` uses the CLS token.
	The head is the SAR-SSL one (model.SARSSL, downstream_head='mlp'): LayerNorm + Linear on the pooled embedding.
"""

import math
import os
import sys

import torch
import torch.nn as nn

import learner as at_learner

SAC_SSAMBA_SRC = os.environ.get('SAC_SSAMBA_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'sac-ssamba', 'src'))
SPATIAL_PREFIX = 'dual_encoder.spatial_encoder.'


def load_spatial_state_dict(ckpt_path):
	""" Spatial-encoder weights from a BinauralSSAMBASACModel checkpoint (audio_model.*.pth, best_audio_model.pth or
		latest.pth), with the prefix removed. Fails if the checkpoint holds none.
	"""
	sd = torch.load(ckpt_path, map_location='cpu')
	if isinstance(sd, dict) and 'state_dict' in sd:  # latest.pth (full training state)
		sd = sd['state_dict']
	out = {}
	for k, v in sd.items():
		if k.startswith('dp.module.'):
			continue  # duplicate copy from BinauralSSAMBASACModelParallel
		if k.startswith('module.'):
			k = k[len('module.'):]
		if k.startswith(SPATIAL_PREFIX):
			out[k[len(SPATIAL_PREFIX):]] = v
	if not out:
		raise RuntimeError(f'[ssamba] no {SPATIAL_PREFIX}* weights in {ckpt_path}')
	return out


class SSAMBASpatialDownstream(nn.Module):
	def __init__(self, ckpt_path=None, frozen=False, fill='tile', token='all', dlabel=1,
				 n_fft=512, hop=160, n_freq=256, n_frames=1024, clip_seconds=10.0, fs=16000, peak=0.9,
				 use_checkpointing=False):
		super().__init__()
		assert fill in ['tile', 'zero'], f'unknown fill {fill}'
		assert token in ['all', 'cls'], f'unknown token {token}'
		if SAC_SSAMBA_SRC not in sys.path:
			sys.path.append(SAC_SSAMBA_SRC)  # appended: this repo's modules (model, learner, dataset, common) keep priority
		from models.both_models import AMBAModel

		self.fill, self.token, self.frozen = fill, token, frozen
		self.n_fft, self.hop, self.n_freq, self.n_frames, self.peak = n_fft, hop, n_freq, n_frames, peak
		self.clip_samples = int(round(clip_seconds * fs))
		self.fshape = self.tshape = 16

		vision_mamba_config = {} if torch.cuda.is_available() else {'fused_add_norm': False}
		self.encoder = AMBAModel(fshape=16, tshape=16, fstride=16, tstride=16, input_fdim=n_freq, input_tdim=n_frames,
								 model_size='base', in_chans=4, pretrain_stage=True, vision_mamba_config=vision_mamba_config)
		self.encoder.use_checkpointing = use_checkpointing
		self.encoder.v.use_checkpointing = use_checkpointing
		# pretraining-only parts of AMBAModel, not used here (kept so the checkpoint loads strictly)
		for name, p in self.encoder.named_parameters():
			if name.split('.')[0] in ['cpredlayer', 'gpredlayer', 'mask_embed']:
				p.requires_grad = False

		if ckpt_path is None or ckpt_path == 'random':
			print('[ssamba] WARNING: no checkpoint, the spatial encoder stays RANDOMLY INITIALISED', flush=True)
		else:
			if not os.path.isfile(ckpt_path):
				raise FileNotFoundError(f'[ssamba] checkpoint not found: {ckpt_path}')
			sd = load_spatial_state_dict(ckpt_path)
			self.encoder.load_state_dict(sd, strict=True)
			print(f'[ssamba] loaded {len(sd)} spatial-encoder tensors from {ckpt_path}', flush=True)

		if frozen:
			for p in self.encoder.parameters():
				p.requires_grad = False

		embed_dim = self.encoder.original_embedding_dim
		self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, dlabel))

	def train(self, mode=True):
		super().train(mode)
		if self.frozen:
			self.encoder.eval()  # linear evaluation: the encoder is a fixed feature extractor
		return self

	def frontend(self, x):
		""" x: (nb, nsample, nch) waveform. Returns X (nb, 4, n_freq, n_frames) and the number of real frames.
		"""
		w = x.permute(0, 2, 1)[:, :2].float()  # (nb, 2, nsample)
		if self.peak is not None:
			w = w / w.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8) * self.peak
		n = w.shape[-1]
		if self.fill == 'tile' and n < self.clip_samples:
			w = w.repeat(1, 1, math.ceil(self.clip_samples / n))[..., :self.clip_samples]
		elif n > self.clip_samples:
			w = w[..., :self.clip_samples]
		nb = w.shape[0]
		with torch.autocast(device_type=w.device.type, enabled=False):
			spec = torch.stft(w.reshape(nb * 2, -1), n_fft=self.n_fft, hop_length=self.hop, return_complex=True)
		spec = spec.reshape(nb, 2, spec.shape[-2], spec.shape[-1])[:, :, :self.n_freq]  # (nb, 2, F, T)
		X = torch.stack([spec.real, spec.imag], dim=2).reshape(nb, 4, spec.shape[2], spec.shape[3])  # Re_L, Im_L, Re_R, Im_R
		n_real = min(X.shape[-1], self.n_frames)
		if X.shape[-1] < self.n_frames:
			X = nn.functional.pad(X, (0, self.n_frames - X.shape[-1]))
		return X[..., :self.n_frames], n_real

	def forward(self, x):
		""" x: (nb, nsample, nch) waveform (WaveLearner passes it through unchanged).
			Returns pred (nb, dlabel), pooled embedding (nb, D), encoder tokens (nb, 1 + npatch, D).
		"""
		X, n_real = self.frontend(x)
		h = self.encoder._encode_with_mamba(X)  # (nb, 1 + nfp*ntp, D), CLS first
		if self.token == 'cls':
			embed = h[:, 0]
		else:
			nfp, ntp = self.n_freq // self.fshape, self.n_frames // self.tshape
			ntp_real = min(ntp, math.ceil(n_real / self.tshape))
			patches = h[:, 1:].reshape(h.shape[0], nfp, ntp, h.shape[-1])  # patch order is frequency-major
			embed = patches[:, :, :ntp_real].mean(dim=(1, 2))
		return self.mlp_head(embed), embed, h


class WaveLearner(at_learner.STFTLearner):
	""" STFTLearner that hands the raw waveform to the model, which makes its own input (SSAMBASpatialDownstream).
		Targets, loss, metric, training loop and checkpoints are STFTLearner's.
	"""
	def data_preprocess(self, mic_sig_batch=None, gt_batch=None, eps=1e-6):
		data = []
		if mic_sig_batch is not None:
			data += [mic_sig_batch.to(self.device)]
		if gt_batch is not None:
			data += [self.get_tar_batch(gt_batch=gt_batch[self.task].to(self.device))]
		return data
