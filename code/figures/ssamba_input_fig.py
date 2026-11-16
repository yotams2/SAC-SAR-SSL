"""Spatial SSAMBA input: before (raw STFT Re/Im, 10 s tiled) vs now (ipd16), with patch rectangles.
Reads one clip out of the pretraining tar (numpy only, mirrors ssamba_spatial.py frontend / stft_frontend)."""
import io, json, sys
import numpy as np, soundfile as sf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TAR = '/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_pretrain.tar'
name = sys.argv[1] if len(sys.argv) > 1 else 'pretest/0.wav'
out = sys.argv[2]
off, size = json.load(open(TAR + '.index.json'))[name]
with open(TAR, 'rb') as f:
    f.seek(off); sig, fs = sf.read(io.BytesIO(f.read(size)), dtype='float32')
sig = sig[:, :2]; n = sig.shape[0]; FS = 16000

def stft(x, nfft, hop, win):
    x = np.pad(x, nfft // 2, mode='reflect')  # torch center=True
    nfr = 1 + (len(x) - nfft) // hop
    fr = np.stack([x[i * hop:i * hop + nfft] for i in range(nfr)], 1) * win[:, None]
    return np.fft.rfft(fr, axis=0)

# ---- before: wave input (peak 0.9, tile to 10 s, no window, hop 160, bins 0..255, 1024 frames)
w = sig / np.abs(sig).max() * 0.9
w10 = np.tile(w, (int(np.ceil(160000 / n)), 1))[:160000]
S_old = [stft(w10[:, c], 512, 160, np.ones(512))[:256] for c in range(2)]
real_old = S_old[0].shape[1]  # 1001
old = np.zeros((256, 1024)); old[:, :real_old] = np.log(np.abs(S_old[0]) + 1e-3)
old[:, real_old:] = np.nan

# ---- now: ipd16 (Hann 512, hop 256, /mean|STFT mic0|, bins 1..256, 256 frames, 4 spatial channels)
hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(512) / 512)
S = [stft(sig[:, c], 512, 256, hann) for c in range(2)]
m = np.abs(S[0]).mean()
L, R = S[0][1:257, :256] / m, S[1][1:257, :256] / m
ipd = np.angle(L * np.conj(R))
logL = np.log(np.abs(L) + 1e-3); ild = np.log(np.abs(R) + 1e-3) - logL
act = logL > np.percentile(logL, 65)  # speech-dominated bins; the rest is greyed out in the spatial panels
msk = lambda a: np.where(act, a, np.nan)
chans = [(logL, 'log|L|', 'magma'), (msk(ild), 'ILD  (grey = low-energy bins)', 'RdBu_r'),
         (msk(np.cos(ipd)), 'cos IPD', 'twilight'), (msk(np.sin(ipd)), 'sin IPD', 'twilight')]

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})
fig = plt.figure(figsize=(13, 9.6), dpi=130)
gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1, 1], hspace=0.75, wspace=0.18)
grid_kw = dict(color='white', lw=0.3, alpha=0.35)
hi = '#00e5ff'; hi2 = '#ffd400'

# row 1: before
ax = fig.add_subplot(gs[0, :])
cm = plt.get_cmap('magma').copy(); cm.set_bad('#d9d9d9')
ax.imshow(old, origin='lower', aspect='auto', cmap=cm, extent=[0, 1024, 0, 256],
          vmin=np.nanpercentile(old, 5), vmax=np.nanpercentile(old, 99.5), interpolation='nearest')
for x in range(0, 1025, 16): ax.axvline(x, **grid_kw)
for y in range(0, 257, 16): ax.axhline(y, **grid_kw)
rep = n / 160
k = 1
while k * rep < real_old:
    ax.axvline(k * rep, color=hi2, lw=1.4, ls='--'); k += 1
ax.text(rep / 2, 236, 'real clip (4.1 s)', ha='center', va='top', color='white', fontsize=9)
ax.text(rep * 1.5, 236, 'tiled copy', ha='center', va='top', color=hi2, fontsize=9)
ax.text((2 * rep + real_old) / 2, 236, 'copy, cut at 10 s', ha='center', va='top', color=hi2, fontsize=9)
ax.text((real_old + 1024) / 2, 128, 'pad', ha='center', va='center', fontsize=8, color='#444')
ax.add_patch(Rectangle((160, 96), 16, 16, fill=False, ec=hi, lw=2.2))
ax.annotate('one 16x16 patch\n= 16 bins x 0.16 s', xy=(176, 104), xytext=(300, 200), color='white', fontsize=9,
            arrowprops=dict(arrowstyle='->', color=hi, lw=1.3))
ax.set_xlim(0, 1024); ax.set_ylim(0, 256)
ax.set_xticks(np.arange(0, 1025, 100)); ax.set_xticklabels([f'{t * 0.01:.0f}' for t in np.arange(0, 1025, 100)])
ax.set_yticks([0, 64, 128, 192, 256]); ax.set_yticklabels(['0', '2', '4', '6', '8'])
ax.set_xlabel('time (s)'); ax.set_ylabel('kHz')
ax.set_title('BEFORE (to 11-11): raw STFT, 4 ch = Re/Im of L and R (shown: log|L|), no window, hop 10 ms, '
             'clip tiled to 10 s  ->  256 x 1024, 16 x 64 = 1024 tokens', loc='left')

# row 2: now, the 4 channels
nt = 256
for i, (img, title, cmap) in enumerate(chans):
    ax = fig.add_subplot(gs[1, i])
    lo, hi_ = (np.percentile(img, 2), np.percentile(img, 99.5)) if i == 0 else \
        ((-np.nanpercentile(np.abs(img), 98),) * 1 + (np.nanpercentile(np.abs(img), 98),)) if i == 1 else (-1, 1)
    cmap = plt.get_cmap(cmap).copy(); cmap.set_bad('#cfcfcf')
    ax.imshow(img, origin='lower', aspect='auto', cmap=cmap, extent=[0, nt, 0, 256], vmin=lo, vmax=hi_,
              interpolation='nearest')
    for x in range(0, nt + 1, 16): ax.axvline(x, **grid_kw)
    for y in range(0, 257, 16): ax.axhline(y, **grid_kw)
    ax.add_patch(Rectangle((64, 96), 16, 16, fill=False, ec=hi, lw=2.0))
    ax.set_title(title)
    ax.set_xticks([0, 62.5, 125, 187.5, 250]); ax.set_xticklabels(['0', '1', '2', '3', '4'])
    ax.set_yticks([0, 64, 128, 192, 256]); ax.set_yticklabels(['0', '2', '4', '6', '8'] if i == 0 else [])
    ax.set_xlabel('time (s)')
    if i == 0: ax.set_ylabel('kHz')
fig.axes[1].text(0, 1.22, 'NOW (ipd16, from 11-12): SAR-SSL STFT (Hann 512, hop 16 ms), 4 explicit spatial channels, '
         'z-normed, real 4.1 s clip only  ->  256 x 256, 16 x 16 = 256 tokens (same 16x16 patch, cyan)',
         fontsize=11, ha='left', transform=fig.axes[1].transAxes)

# row 3: patch shapes tested in the input probe (11-12), on log|L|
probe = [('16 x 16 patch: ipd16 (chosen), reim16, old input', 'sq'), ('full-band frame token, 256 x 1: ipdframe, reimframe', 'fr')]
for j, (title, kind) in enumerate(probe):
    ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
    ax.imshow(logL, origin='lower', aspect='auto', cmap='magma', extent=[0, nt, 0, 256],
              vmin=np.percentile(logL, 2), vmax=np.percentile(logL, 99.5), interpolation='nearest')
    if kind == 'sq':
        for x in range(0, nt + 1, 16): ax.axvline(x, **grid_kw)
        for y in range(0, 257, 16): ax.axhline(y, **grid_kw)
        for (x, y) in [(64, 96), (128, 32), (192, 176)]:
            ax.add_patch(Rectangle((x, y), 16, 16, fill=False, ec=hi, lw=2.0))
    else:
        for x in range(0, nt + 1, 8): ax.axvline(x, color='white', lw=0.25, alpha=0.25)
        for x in [64, 128, 192]:
            ax.add_patch(Rectangle((x, 0), 1, 256, fill=False, ec=hi2, lw=2.0))
    ax.set_title(title)
    ax.set_xticks([0, 62.5, 125, 187.5, 250]); ax.set_xticklabels(['0', '1', '2', '3', '4'])
    ax.set_yticks([0, 64, 128, 192, 256]); ax.set_yticklabels(['0', '2', '4', '6', '8'] if j == 0 else [])
    ax.set_xlabel('time (s)')
    if j == 0: ax.set_ylabel('kHz')
fig.axes[5].text(0, 1.22, 'Patch shapes compared in the input probe (random init, TDOA / T60 test MAE): '
         'ipd16 0.66 / 0.133, ipdframe 0.65 / 0.185, reim16 2.93 / 0.212, old Re/Im 2.92 / 0.199; SAR-SSL ref 0.64 / 0.107',
         fontsize=11, ha='left', transform=fig.axes[5].transAxes)
fig.savefig(out, bbox_inches='tight')
print('saved', out, 'n', n, 'real_old', real_old)
