""" Checks for the SAC forms in model.SARSSL._calc_single_feat_sim_loss (cwcl / sigmoid / queue). GPU or CPU.
Usage: python sac_forms_check.py <path to an unmodified model.py (main branch)>
"""
import importlib.util
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as new_model  # noqa: E402

spec = importlib.util.spec_from_file_location('model_main', sys.argv[1])
old_model = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_model)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
KEYS = ['a', 'b', 'c', 'd', 'e']
B, T, S = 128, 0.15, 0.613


class Host(nn.Module):  # stands in for SSAMBASpatialPretrain: the attributes the loss reads
	def __init__(self, form, Q=256, warm=2):
		super().__init__()
		self.feature_stats = {k: {'mean': 0.0, 'std': 1.0} for k in KEYS}
		self.normalization_mode, self.factorized_sac, self.sac_form = 'z_norm', False, form
		if form == 'sigmoid':
			self.sac_logit_scale = nn.Parameter(torch.tensor(math.log(1 / T)))
			self.sac_logit_bias = nn.Parameter(torch.tensor(0.0))
			self.register_buffer('sac_bias_set', torch.tensor(False))
		if form == 'queue':
			self.sac_queue_warmup = warm
			for n, v in (('sac_queue_z', torch.zeros(Q, 128)), ('sac_queue_c', torch.zeros(Q, 5)), ('sac_queue_ptr', torch.tensor(0)),
						 ('sac_queue_n', torch.tensor(0)), ('sac_queue_calls', torch.tensor(0))):
				self.register_buffer(n, v)


def batch():
	z = torch.randn(B, 128, device=dev, requires_grad=True)
	f = {k: torch.randn(B, device=dev) for k in KEYS}
	return z, f


def loss(mod, host, z, f):
	return mod.SARSSL._calc_single_feat_sim_loss(host, z, f, KEYS, T, S, 1e-8, False)[0]


# 1. cwcl == the unchanged main-branch function
z, f = batch()
h = Host('cwcl').to(dev)
l_new, l_old = loss(new_model, h, z, f), loss(old_model, h, z, f)
print(f'[check] cwcl new {l_new.item():.6f} vs main {l_old.item():.6f}')
assert torch.allclose(l_new, l_old)

# 2. queue: == cwcl before warm-up / with an empty queue; uses the memory after; untouched in eval
hq = Host('queue', Q=256, warm=2).to(dev).train()
l0 = loss(new_model, hq, z, f)
print(f'[check] queue call 1 (no memory yet) {l0.item():.6f}  == cwcl: {torch.allclose(l0, l_old)}')
assert torch.allclose(l0, l_old)
z2, f2 = batch(); loss(new_model, hq, z2, f2)
z3, f3 = batch(); l3 = loss(new_model, hq, z3, f3)
l3_plain = loss(old_model, h, z3, f3)
print(f'[check] queue call 3 (memory {int(hq.sac_queue_n)}/256, calls {int(hq.sac_queue_calls)}): {l3.item():.4f} vs batch-only {l3_plain.item():.4f}')
assert int(hq.sac_queue_n) == 256 and not torch.allclose(l3, l3_plain)
assert l3.item() > l3_plain.item()  # more candidates in the softmax -> larger CE at random init
l3.backward(); assert torch.isfinite(z3.grad).all() and z3.grad.abs().sum() > 0
hq.eval(); n_before, calls = int(hq.sac_queue_n), int(hq.sac_queue_calls)
le = loss(new_model, hq, z3, f3)
assert torch.allclose(le, l3_plain) and int(hq.sac_queue_calls) == calls, 'eval must use the plain batch and leave the memory'
print('[check] queue eval == plain batch, memory untouched')

# 3. sigmoid: bias init = logit(mean target), finite, grads reach z, scale and bias
hs = Host('sigmoid').to(dev).train()
z4, f4 = batch()
ls = loss(new_model, hs, z4, f4)
cues = torch.stack([torch.clamp(f4[k] / 3, -1, 1) for k in KEYS], 1)
w = torch.exp(-(torch.cdist(cues, cues) / S) ** 2); m = w[~torch.eye(B, dtype=torch.bool, device=dev)].mean()
print(f'[check] sigmoid loss {ls.item():.4f}; bias {hs.sac_logit_bias.item():.4f} vs logit(mean w) {math.log(m / (1 - m)):.4f}; '
	  f'scale {hs.sac_logit_scale.exp().item():.3f}')
assert abs(hs.sac_logit_bias.item() - math.log(m / (1 - m))) < 1e-4 and bool(hs.sac_bias_set)
ls.backward()
assert torch.isfinite(z4.grad).all() and hs.sac_logit_scale.grad is not None and hs.sac_logit_bias.grad is not None
# per-pair gradient scale comparable to cwcl (same order of magnitude)
z5 = z4.detach().clone().requires_grad_(True); loss(old_model, h, z5, f4).backward()
r = z4.grad.norm().item() / z5.grad.norm().item()
print(f'[check] |grad z| sigmoid / cwcl = {r:.2f}')
assert 0.01 < r < 100
print('[check] all OK')
