""" Check 2 (agreed 2026-11-21): a random-init spectral encoder checkpoint in the Pilot A format, for the leakage baseline
    (random-init spectral encoder on T60 / DRR, frozen). Input standardisation copied from a trained spectral checkpoint,
    so only the weights differ. Usage: python make_random_spec.py <trained spec_encoder.pth> <out.pth> [seed] """
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_pretrain as jp  # noqa: E402
from ssamba_spatial_pretrain import save_atomic  # noqa: E402

ref_path, out = sys.argv[1:3]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.manual_seed(seed)
model = jp.JointPretrain('spec', False, 0.0, 7, 10.0, 1.0, use_checkpointing=False)
ref = torch.load(ref_path, map_location='cpu', weights_only=False)['meta']
s = model.streams['spec']
s.in_mean.fill_(ref['in_mean']); s.in_std.fill_(ref['in_std'])
ck = model.spec_checkpoint(argparse.Namespace(streams='spec', fusion=False, note=f'random init, seed {seed}'))
save_atomic(ck, out)
print(f'wrote {out} (random init seed {seed}, input stats of {ref_path})')
