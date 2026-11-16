"""Two infographics for the Notion report: (A) SAR-SSL vs SSAMBA front end on raw Re/Im; (B) the naive dual-encoder failure."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED = '#1f2328', '#59636e'
GOOD, GOOD_BG = '#1a7f37', '#dafbe1'
BAD, BAD_BG = '#cf222e', '#ffebe9'
NEU_BG, KEY_BG = '#f1f3f5', '#fff8c5'
BLUE_BG = '#ddf4ff'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})


def box(ax, x, y, w, h, text, fc=NEU_BG, ec='#d0d7de', fs=10, weight='normal', color=INK, lw=1.2, ha='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012', fc=fc, ec=ec, lw=lw))
    tx = x + w / 2 if ha == 'center' else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fs, fontweight=weight, color=color, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=12, lw=1.4, color=color))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    return fig, ax


# ------------------------------------------------------------------ A
fig, ax = canvas(12, 6.4)
ax.text(0.02, 0.955, 'Same raw input, different front end: why Re/Im works for SAR-SSL but not for SSAMBA (random init, R8)',
        fontsize=14, fontweight='bold', color=INK, va='center')
ax.text(0.02, 0.905, 'TDOA lives in the phase DIFFERENCE between the ears at each time-frequency bin: '
        'Re(L·R*) = Re_L·Re_R + Im_L·Im_R, a product of the two ears.',
        fontsize=10.5, color=MUTED, va='center')

cols = [0.02, 0.215, 0.475, 0.665, 0.84]
wid = [0.165, 0.23, 0.16, 0.145, 0.14]
rows = {'sar': 0.64, 'ssa': 0.37}
h = 0.17
ax.text(0.02, rows['sar'] + h + 0.03, 'SAR-SSL', fontsize=12, fontweight='bold', color=GOOD)
ax.text(0.02, rows['ssa'] + h + 0.03, 'SSAMBA (Re/Im, linear 16×16 patches)', fontsize=12, fontweight='bold', color=BAD)

stages = {
    'sar': ['STFT, 4 channels\nRe_L  Im_L  Re_R  Im_R',
            'CNN stem, per bin\n1×1 conv 4→64 + ReLU\nmixes L and R at the SAME bin\n→ can form L·R* (phase diff.)',
            'Frame tokens\n(256×1, full band)',
            '3 Conformer\nlayers, d 256',
            'TDOA MAE\n0.64'],
    'ssa': ['STFT, 4 channels\nRe_L  Im_L  Re_R  Im_R',
            'Linear 16×16 patch embedding\n1024 values → 768, no nonlinearity\nL and R never multiplied per bin\n→ sees random-phase "noise"',
            '16×16 patches\n(256 tokens)',
            '24 Mamba\nlayers, d 768',
            'TDOA MAE\n2.93'],
}
for r, y in rows.items():
    for i, (x, w, t) in enumerate(zip(cols, wid, stages[r])):
        if i == 1:
            fc, ec = (GOOD_BG, GOOD) if r == 'sar' else (BAD_BG, BAD)
            box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=9.5, lw=1.8)
        elif i == 4:
            fc, ec, c = (GOOD_BG, GOOD, GOOD) if r == 'sar' else (BAD_BG, BAD, BAD)
            box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=12, weight='bold', color=c, lw=1.8)
        else:
            box(ax, x, y, w, h, t, fs=9.5)
        if i < 4:
            arrow(ax, x + w + 0.004, y + h / 2, cols[i + 1] - 0.004, y + h / 2)
ax.text(0.91, rows['ssa'] - 0.03, '≈ predict-the-mean (3.07)', ha='center', fontsize=8.5, color=BAD)

# evidence strip
ax.text(0.02, 0.255, 'Input probe (random init, TDOA test MAE): changing only the front end of SSAMBA',
        fontsize=10.5, fontweight='bold', color=INK)
ev = [('Re/Im\n16×16 linear', '2.93', BAD_BG, BAD),
      ('Re/Im\nframe tokens', '2.31', BAD_BG, BAD),
      ("Re/Im + SAR-SSL's\nCNN stem", '0.99', KEY_BG, '#9a6700'),
      ('phase/level diff.\ngiven (ipd16)', '0.66', GOOD_BG, GOOD),
      ('SAR-SSL\nreference', '0.64', NEU_BG, MUTED)]
ex, ew, eh, ey = 0.02, 0.17, 0.15, 0.06
for i, (lab, val, fc, c) in enumerate(ev):
    x = ex + i * (ew + 0.025)
    box(ax, x, ey, ew, eh, '', fc=fc, ec=c, lw=1.4)
    ax.text(x + ew / 2, ey + eh * 0.68, lab, ha='center', va='center', fontsize=9, color=INK, linespacing=1.25)
    ax.text(x + ew / 2, ey + eh * 0.22, val, ha='center', va='center', fontsize=12.5, fontweight='bold', color=c)
    if i < len(ev) - 1:
        arrow(ax, x + ew + 0.003, ey + eh / 2, x + ew + 0.022, ey + eh / 2)
ax.text(0.02, 0.02, 'The stem alone recovers most of the gap; giving the phase difference directly closes it. '
        'Remaining 0.99 vs 0.64 (Mamba vs Conformer backbone) is untested.', fontsize=9, color=MUTED)
fig.savefig('fig7_sarssl_vs_ssamba.png', facecolor='white')

# ------------------------------------------------------------------ B
fig, ax = canvas(12, 7.0)
ax.text(0.02, 0.957, 'Reusing the dual-encoder + CCSR setup with SSAMBA backbones fails on both streams',
        fontsize=14, fontweight='bold', color=INK, va='center')
ax.text(0.02, 0.91, 'Both encoders got the raw complex STFT and a linear 16×16 patch embedding. '
        'Neither cue they need is a weighted sum of Re/Im values.', fontsize=10.5, color=MUTED, va='center')

# pipeline
box(ax, 0.02, 0.6, 0.17, 0.2, 'Binaural clip\n→ STFT\nRe/Im of L and R\n(4 channels)', fs=9.5)
box(ax, 0.26, 0.71, 0.2, 0.12, 'Spectral encoder\nSSAMBA, linear 16×16', fs=9.5)
box(ax, 0.26, 0.57, 0.2, 0.12, 'Spatial encoder\nSSAMBA, linear 16×16', fs=9.5)
arrow(ax, 0.194, 0.73, 0.256, 0.77); arrow(ax, 0.194, 0.67, 0.256, 0.63)
box(ax, 0.53, 0.6, 0.16, 0.2, 'CCSR\n(cross-channel\nreconstruction of\nthe complex STFT)\n+ SAC', fs=9.5)
arrow(ax, 0.464, 0.77, 0.526, 0.74); arrow(ax, 0.464, 0.63, 0.526, 0.66)
box(ax, 0.76, 0.6, 0.22, 0.2, 'Root cause (common)\nThe useful cues are QUADRATIC\nin Re/Im; a linear patch\nembedding of raw values\ncannot form them', fc=KEY_BG, ec='#9a6700', fs=9.5, lw=1.6)

# two failure columns
cw, ch, cy = 0.465, 0.42, 0.105
for j, (title, need, ev, fix) in enumerate([
    ('Spectral stream (speech tasks)',
     'Needs magnitude |X|² = Re² + Im², then log compression.\nPer-bin phase is random, so raw Re/Im looks like noise.',
     ['Masked-patch classification accuracy: ~5% (log-mel 29%)',
      'Linear probe, SC2: 9.5 (log-mel 18.4; GRAM-T 71)',
      'Fine-tuned SID: binaural 41.5 vs mono SSAMBA 60.5',
      'Removing CCSR did not help an STFT encoder'],
     'Fix: log-mel input. Necessary but not sufficient (still far below GRAM-T).'),
    ('Spatial stream (SAR-SSL tasks)',
     'Needs inter-ear phase/level difference L·R* per bin.\nSAR-SSL forms it with a CNN stem; SSAMBA has none.',
     ['TDOA fine-tuned 2.39 (SAR-SSL 0.29-0.48)',
      'TDOA frozen 2.92 ≈ predict-the-mean 3.07',
      'Random init 2.86 (SAR-SSL from scratch 0.50)',
      'lr sweep: 2.89-2.93 at every lr'],
     'Fix: ipd16 input (log|L|, ILD, cos/sin IPD) → TDOA 0.23, beats SAR-SSL.')]):
    x = 0.02 + j * (cw + 0.03)
    box(ax, x, cy, cw, ch, '', fc='white', ec=BAD, lw=1.8)
    ax.text(x + 0.015, cy + ch - 0.04, title, fontsize=12, fontweight='bold', color=BAD, va='center')
    ax.text(x + 0.015, cy + ch - 0.11, need, fontsize=9.5, color=INK, va='center', linespacing=1.35)
    for k, line in enumerate(ev):
        ax.text(x + 0.015, cy + ch - 0.19 - k * 0.042, '✗  ' + line, fontsize=9.5, color=INK, va='center')
    box(ax, x + 0.01, cy + 0.012, cw - 0.02, 0.055, fix, fc=GOOD_BG, ec=GOOD, fs=9.5, color=INK, ha='left')
ax.text(0.02, 0.012, 'Conclusion: the dual-encoder + CCSR recipe cannot be reused with SSAMBA as is. Each stream needs an input\n'
        'in which its cue is (close to) linear: log-mel for the spectral stream, IPD/ILD channels for the spatial stream.', fontsize=10, fontweight='bold', va='bottom', linespacing=1.4, color=INK)
fig.savefig('fig8_naive_dual_encoder.png', facecolor='white')
print('ok')
