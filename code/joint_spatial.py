""" Frozen two-stream upstream for the SAR-SSL downstream tasks (run_downstream.py --upstream joint).

Both encoders of a joint_pretrain.py run (spectral log-mel + spatial ipd16), fed from SAR-SSL's own STFT (STFTLearner:
Hann 512 / hop 256, / mean |L|, bins 1..256), exactly as in pretraining. Each stream is mean-pooled over its real
patches (as SSAMBASpatialDownstream), the two vectors are concatenated [2 * D] -> LayerNorm -> Linear head.
	mode 'concat': Pilot A-concat. ckpt = '<spec_encoder.pth>,<spat_encoder.pth>' (two separately trained encoders,
	               each with its own input statistics), run independently.
	mode 'fusion': Pilot B with fusion on. ckpt = <joint_{best,final}.pth> (both encoders + fusion), run jointly.
	mode 'spec':   spectral encoder alone (leakage probe: spatial tasks from the spectral stream). ckpt = <spec_encoder.pth>;
	               head on its D-dim mean only.
Frozen only (lineareval): this compares representations, the head is the only trained part.
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_pretrain as jp  # noqa: E402
import ssamba_spatial  # noqa: E402


class JointSpatialDownstream(nn.Module):
	def __init__(self, ckpt, mode, dlabel=1):
		super().__init__()
		assert mode in ('concat', 'fusion', 'spec'), mode
		self.mode = mode
		self.names = ('spec',) if mode == 'spec' else ('spec', 'spat')
		self.model = jp.JointPretrain('spec' if mode == 'spec' else 'joint', mode == 'fusion', 0.3, 7, 10.0, 1.0,
									  use_checkpointing=False)
		if mode == 'fusion':
			st = torch.load(ckpt, map_location='cpu', weights_only=False)
			assert st['args']['fusion'], f'{ckpt} was not trained with fusion'
			self.model.load_state_dict(st['model'], strict=True)
			print(f'[joint] fusion model from {ckpt}', flush=True)
		else:
			spec_path, spat_path = (ckpt, None) if mode == 'spec' else ckpt.split(',')
			spec = torch.load(spec_path, map_location='cpu', weights_only=False)
			assert spec['meta']['input'] == 'logmel_mid_sarssl_stft', spec['meta']['input']
			s = self.model.streams['spec']
			s.enc.load_state_dict(spec['state_dict'], strict=True)
			s.in_mean.fill_(spec['meta']['in_mean']); s.in_std.fill_(spec['meta']['in_std'])
		if mode == 'concat':
			t = self.model.streams['spat']
			t.enc.load_state_dict(ssamba_spatial.load_spatial_state_dict(spat_path), strict=True)
			meta = ssamba_spatial.load_meta(spat_path)
			t.in_mean.copy_(torch.tensor(meta['in_mean']).view_as(t.in_mean))
			t.in_std.copy_(torch.tensor(meta['in_std']).view_as(t.in_std))
			print(f'[joint] concat of {spec_path} + {spat_path}', flush=True)
		for p in self.model.parameters():
			p.requires_grad = False
		D = 768
		self.mlp_head = nn.Sequential(nn.LayerNorm(len(self.names) * D), nn.Linear(len(self.names) * D, dlabel))

	def train(self, mode=True):
		super().train(mode)
		self.model.eval()  # frozen feature extractor
		return self

	def stft_n(self, x):
		""" STFTLearner's (nb, 2, 256, nt, 2) (bins 1..256) -> the complex (nb, 2, 257, nt) of jp.shared_stft. The DC
			bin is dropped by STFTLearner; it carries no weight in the mel filterbank (checked in joint_spatial_check.py)
			and the spatial stream never uses it, so it is set to 0. """
		x = x.float()
		z = torch.complex(x[..., 0], x[..., 1])
		return nn.functional.pad(z, (0, 0, 1, 0))

	@torch.no_grad()
	def patches(self, x):
		""" -> {stream: (nb, rows, ntr, D) patch tokens over real time columns}, CLS dropped. """
		stft_n = self.stft_n(x)
		n_real = min(stft_n.shape[-1], jp.NT)
		X = self.model.raw_inputs(stft_n)
		Xn = {}
		for k, v in X.items():
			v = self.model.streams[k].norm_input(v)
			v[..., n_real:] = 0.0  # frames past the clip are 0 after standardising, as SSAMBASpatialDownstream
			Xn[k] = v
		h = self.model.encode(Xn, None, fusion=self.mode == 'fusion')
		ntr = min(jp.N_COLS, math.ceil(n_real / 16))
		out = {}
		for k in self.names:
			rows = self.model.streams[k].n_rows
			out[k] = h[k][:, 1:].reshape(h[k].shape[0], rows, jp.N_COLS, -1)[:, :, :ntr].float()  # frequency-major
		return out

	def embed(self, x):
		p = self.patches(x)
		return torch.cat([p[k].mean(dim=(1, 2)) for k in self.names], dim=-1)

	def forward(self, x):
		embed = self.embed(x)
		return self.mlp_head(embed), embed, None
