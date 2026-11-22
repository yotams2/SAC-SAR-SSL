""" CPU check for joint_spatial.PerStreamHead (--joint-head-norm per_stream). Usage: python joint_head_check.py """
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_spatial import PerStreamHead  # noqa: E402

torch.manual_seed(0)
D = 768
h = PerStreamHead(2, D, 1)
j = nn.Sequential(nn.LayerNorm(2 * D), nn.Linear(2 * D, 1))
n_h, n_j = (sum(p.numel() for p in m.parameters()) for m in (h, j))
assert n_h == n_j == 2 * 768 * 2 + 2 * 768 + 1, (n_h, n_j)
spec, spat = torch.randn(64, D), torch.randn(64, D) * 0.05 + 0.3
x = torch.cat([spec, spat], -1)
# each half normalised exactly as the single-encoder head's LayerNorm(D)
ref = torch.cat([nn.functional.layer_norm(spec, (D,)), nn.functional.layer_norm(spat, (D,))], -1)
got = torch.cat([ln(c) for ln, c in zip(h.norms, x.split(D, -1))], -1)
assert torch.allclose(got, ref, atol=1e-5)
# the spatial half's normalised value no longer depends on the spectral half's scale (it did with one joint LayerNorm)
x2 = torch.cat([spec * 100, spat], -1)
d_ps = (h.norms[1](x2[:, D:]) - h.norms[1](x[:, D:])).abs().max().item()
d_j = (j[0](x2)[:, D:] - j[0](x)[:, D:]).abs().max().item()
assert d_ps == 0.0 and d_j > 1e-2, (d_ps, d_j)
print(f'joint LayerNorm: spatial-half change when the spectral half is x100 = {d_j:.3f}; per_stream = {d_ps}')
print('head output shape', tuple(h(x).shape))
print('PASS')
