"""Checks code/ssamba_spatial.py on a GPU before a downstream run (sbatch scripts/check_ssamba_spatial.sbatch <ckpt>).

1. the checkpoint's spatial encoder loads strictly;
2. the wrapper's batched front-end equals sac-ssamba's wav2stft (the pretraining input) on a real binaural clip;
3. forward + backward at the two SAR-SSL clip lengths (1.04 s TDOA, 4.112 s others), both fills, frozen and not;
4. time per training batch (bs 8) and per eval batch (bs 16), to size the downstream jobs.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import ssamba_spatial  # noqa: E402

ckpt = sys.argv[1]
wav = sys.argv[2] if len(sys.argv) > 2 else None
dev = 'cuda'
fs = 16000

net = ssamba_spatial.SSAMBASpatialDownstream(ckpt_path=ckpt).to(dev)
n_train = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
print(f'trainable params {n_train:.1f} M')

# 2. front-end vs the pretraining dataloader
from binaural_dual_stream_sac.binaural_dataloader import wav2stft  # on sys.path via ssamba_spatial
import soundfile as sf
x, sr = sf.read(wav) if wav else (np.random.randn(160000, 2) * 0.1, fs)
x = x[:160000] / np.abs(x[:160000]).max() * 0.9  # pretraining clips are 10 s with peak 0.9
ref = wav2stft(torch.tensor(x.T, dtype=torch.float32))
got, n_real = net.frontend(torch.tensor(x, dtype=torch.float32, device=dev)[None])
err = (got[0].cpu() - ref).abs().max().item() / ref.abs().max().item()
print(f'front-end vs wav2stft: shape {tuple(got.shape)} vs {tuple(ref.shape)}, n_real {n_real}, max rel err {err:.2e}')
assert got.shape[1:] == ref.shape and err < 1e-4, 'front-end differs from the pretraining input'

# 3./4.
for T in [1.04, 4.112]:
    for fill in ['tile', 'zero']:
        for frozen in [False, True]:
            m = ssamba_spatial.SSAMBASpatialDownstream(ckpt_path=ckpt, frozen=frozen, fill=fill).to(dev).train()
            opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)
            xb = torch.randn(8, int(T * fs), 2, device=dev) * 0.1
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(3):
                pred, emb, h = m(xb)
                loss = pred.pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize(); t_train = (time.time() - t0) / 3
            m.eval()
            with torch.no_grad():
                xe = torch.randn(16, int(T * fs), 2, device=dev) * 0.1
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(3):
                    pe, ee, _ = m(xe)
                torch.cuda.synchronize(); t_eval = (time.time() - t0) / 3
            assert torch.isfinite(pred).all() and torch.isfinite(pe).all()
            print(f'T {T} fill {fill} frozen {frozen}: pred {tuple(pred.shape)} embed {tuple(emb.shape)} tokens {tuple(h.shape)}; '
                  f'train step bs8 {t_train:.2f} s, eval bs16 {t_eval:.2f} s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')
            del m, opt
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
print('ALL CHECKS PASSED')
