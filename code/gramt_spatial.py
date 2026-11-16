"""
	GRAM-T (labhamlet/gramt-binaural-frame, Yuksel et al. 2025) as the upstream of run_downstream.py, the external
	reference for the spatial SSAMBA encoder on SAR-SSL's tasks. Selected with `--upstream gramt [--gramt-model <id>]`.

	The encoder is sac-ssamba's shared wrapper (src/gramt_wrapper/gramt_encoder.py), the one behind our GRAM-T SC2 / SID
	numbers. It follows the official HF extractor: resample 16 -> 32 kHz, one RMS gain for both ears (-14 dBFS, so level
	differences between the ears stay), 2-channel 128-bin log-mel (n_fft 1024, hop 320), per-2-s-chunk LayerNorm, and
	non-overlapping 2 s chunks. Unlike HF `get_audio_representation` it does not force no_grad, so finetune works.
	Note the input has no phase: only level/spectral differences between the ears reach the model.

	Embedding: GRAM's recommended "raw" strategy, last layer, tokens [T_tokens, 8 freq patches x 768], averaged over the
	time tokens that cover real audio (chunk padding excluded): the same pooling as GRAMTClassifier (SC2).
	TDOA clips (1.04 s) fill 13 of the 25 tokens of one chunk; 4.112 s clips use 3 chunks (~52 of 75 tokens).
	Head: the SAR-SSL one, LayerNorm + Linear (as in ssamba_spatial.py).
	finetune and lineareval only: no scratchLOW (a faithful random GRAM-T would need its own init code).
"""

import os
import sys

import torch
import torch.nn as nn

SAC_SSAMBA_SRC = os.environ.get('SAC_SSAMBA_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'sac-ssamba', 'src'))


class GRAMTSpatialDownstream(nn.Module):
	def __init__(self, model_id='labhamlet/gramt-binaural-frame', frozen=False, dlabel=1, fs=16000):
		super().__init__()
		if SAC_SSAMBA_SRC not in sys.path:
			sys.path.append(SAC_SSAMBA_SRC)  # appended: this repo's modules keep priority
		from gramt_wrapper import GRAMTEncoder

		self.frozen = frozen
		self.encoder = GRAMTEncoder(model_id=model_id, input_sample_rate=fs)
		print(f'[gramt] loaded {model_id}', flush=True)
		if frozen:
			for p in self.encoder.parameters():
				p.requires_grad = False
		self.mlp_head = nn.Sequential(nn.LayerNorm(self.encoder.output_dim), nn.Linear(self.encoder.output_dim, dlabel))
		print(f'[gramt] in_channels {self.encoder.in_channels}, grid {self.encoder.grid_f}x{self.encoder.grid_t} per 2 s, '
			  f'embedding {self.encoder.output_dim}, frozen {frozen}', flush=True)

	def train(self, mode=True):
		super().train(mode)
		if self.frozen:
			self.encoder.eval()
		return self

	def forward(self, x):
		""" x: (nb, nsample, nch) waveform (WaveLearner passes it through unchanged).
			Returns pred (nb, dlabel), pooled embedding (nb, D), last-layer tokens (nb, T_tokens, D).
		"""
		w = x.permute(0, 2, 1)[:, :2].float()  # (nb, 2, nsample): L, R
		with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
			hidden, n_tokens = self.encoder(w)
		h = hidden[-1]
		mask = torch.zeros(h.shape[:2], device=h.device, dtype=h.dtype)
		for i, n in enumerate(n_tokens):
			mask[i, :n] = 1.0
		embed = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
		return self.mlp_head(embed), embed, h
