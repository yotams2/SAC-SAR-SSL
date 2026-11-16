import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
groups = ['Fine-tuned', 'Frozen (linear)', 'Random init']
before = [2.39, 2.92, 2.86]          # old Re/Im input, old binaural checkpoint (11-11)
nosac  = [0.26, 2.43, np.nan]
sac    = [0.23, 1.31, np.nan]
rand_new = [np.nan, np.nan, 1.04]    # ipd16, 256 frames, same setup as the pretrained arms
x = np.arange(3); w = 0.2
fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=140)
c = ['#9aa0a6', '#4c78a8', '#e45756', '#72b7b2']
for i, (v, lab) in enumerate([(before, 'Before: old input, old binaural ckpt (11-11)'),
                              (nosac, 'ipd16 pretrained, no SAC (30 ep)'),
                              (sac, 'ipd16 pretrained, + SAC (30 ep)'),
                              (rand_new, 'ipd16, random init')]):
    xs = x + (i - 1.5) * w
    ax.bar(xs, v, w, color=c[i], label=lab)
    for xi, vi in zip(xs, v):
        if np.isfinite(vi): ax.text(xi, vi + 0.04, f'{vi:.2f}', ha='center', va='bottom', fontsize=8)
ax.axhspan(0.29, 0.48, xmin=0.02, xmax=0.32, color='#59a14f', alpha=0.25)
ax.text(-0.42, 0.52, 'SAR-SSL fine-tuned 0.29-0.48', fontsize=8, color='#2f6b2a')
ax.plot([0.62, 1.38], [1.1, 1.1], color='#59a14f', lw=1.2, ls='--'); ax.plot([0.62, 1.38], [2.1, 2.1], color='#59a14f', lw=1.2, ls='--')
ax.text(1.0, 1.73, 'SAR-SSL frozen\n1.1-2.1', ha='center', fontsize=8, color='#2f6b2a')
ax.plot([1.62, 2.38], [0.50, 0.50], color='#59a14f', lw=1.2, ls='--'); ax.text(2.0, 0.55, 'SAR-SSL scratch 0.50', ha='center', fontsize=8, color='#2f6b2a')
ax.axhline(3.07, color='k', lw=0.8, ls=':'); ax.text(2.45, 3.1, 'predict the mean 3.07', ha='right', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel('TDOA test MAE (lower is better)')
ax.set_ylim(0, 3.4); ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.savefig('spatial_tdoa.png', bbox_inches='tight')
