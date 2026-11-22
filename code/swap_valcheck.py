""" Check 1 (agreed 2026-11-21): was Pilot C's epoch-10 validation dip after the resume real or an evaluation artefact?

    Re-evaluates saved full checkpoints (encoders + decoder, joint_<tag>.pth) in a fresh process, with the validation set
    exactly as in training, under both pairings:
      random (Pilot C seed 1's): partners drawn per access by persistent DataLoader workers, so each pass sees new
              triplets and the first pass of a process repeats the first pass of every other process; run twice here
      fixed  (the new default): partners from a per-target seed, the same every pass
    and prints them next to the values logged during training. If the logged epoch-9 value reproduces only under the same
    pass index, the dip is a pairing artefact; if epoch 9 and final differ under the fixed pairing, it is real.

    Usage: python swap_valcheck.py <exp dir> [tags ...]   (default tags: best final; best = epoch 9 for Pilot C seed 1)
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.utils_module as at_module  # noqa: E402
import swap_pretrain as sp  # noqa: E402
from ssamba_spatial_pretrain import NFFT, WIN_LEN, WIN_SHIFT_RATIO  # noqa: E402

KEYS = ['loss', 'swap', 'swap_logL', 'swap_ILD', 'swap_cosIPD', 'swap_sinIPD', 'swap_shufspec', 'swap_shufspat',
        'spec_acc', 'spat_acc']


def main():
    exp, tags = sys.argv[1], sys.argv[2:] or ['best', 'final']
    a = json.load(open(os.path.join(exp, 'args.json')))
    dev = 'cuda'
    n_scenes = int(np.load(a['tar'] + '.meta.npz')['n_scenes'])
    lo = n_scenes - a['val_scenes']
    stft_mod = at_module.STFT(win_len=WIN_LEN, win_shift_ratio=WIN_SHIFT_RATIO, nfft=NFFT).to(dev)
    logged = [json.loads(line) for line in open(os.path.join(exp, 'log.jsonl')) if '"val"' in line]
    print('logged validation (training run):')
    for r in logged[-3:]:
        print('  epoch %4.1f  ' % r['epoch'] + '  '.join('%s %.4f' % (k, r['val'][k]) for k in KEYS if k in r['val']))
    for tag in tags:
        model = sp.SwapPretrain(a['mask_cols'], a['recon_lambda'], a['classif_lambda'], a['swap_lambda'],
                                a['own_scale']).to(dev)
        model.load_state_dict(torch.load(os.path.join(exp, f'joint_{tag}.pth'), map_location='cpu')['model'])
        for pairing in ('random', 'fixed'):
            ds = sp.PairedDataset(a['tar'], lo, n_scenes, a.get('num_val'),
                                  fixed_seed=12345 if pairing == 'fixed' else None)
            loader = torch.utils.data.DataLoader(ds, batch_size=a['bs'], shuffle=False, num_workers=5,
                                                 persistent_workers=True, pin_memory=True)
            for p in range(2 if pairing == 'random' else 1):
                v = sp.evaluate(model, stft_mod, loader, dev)
                print(f'{tag:6s} {pairing:6s} pass {p + 1}: ' + '  '.join('%s %.4f' % (k, v[k]) for k in KEYS if k in v),
                      flush=True)


if __name__ == '__main__':
    main()
