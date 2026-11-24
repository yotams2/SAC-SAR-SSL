""" V1b SID safety check (analysis only, CPU, requested 2026-11-24).

Spectral-side V1b soft weights on simulated paired-v2 TRAINING batches, with the chosen sigma (selected.json; the
weights do not depend on tau). Cues built exactly as _swap_pretrain_v1.set_v1_targets: teacher tokens (n_utt, 8, 16,
768) -> per-dim z-score (training-set stats) -> mean over the 128 tokens -> centre + project on the PCA components
(pca_spec.npz, no whitening) -> L2. Weights as model.SARSSL._calc_single_feat_sim_loss: w_ij = exp(-(|c_i - c_j| /
sigma)^2), diagonal masked, row-normalised.

Per anchor, the normalised mass on: the exact pair (another batch item with the same utterance = identical cue),
other same-speaker items, different-speaker items; plus the chance share (fraction of batch items in each class) and
ESS = 1 / sum_j w_norm_ij^2. Also a speaker nearest-neighbour check of the cue itself (leave-one-out, 1-NN by cosine).
Batches: random targets from the training clips [0, (n_scenes - val_scenes) * k), batch size 128, as the trainer.

Usage: python v1b_sid_check.py --sel <v1b_sigma_tau dir> --out <json> [--n-batches 400]
"""
import argparse
import json
import os

import numpy as np

D = '/scratch200/yotams2/SAC-SAR-SSL'


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sel', default=f'{D}/exp/v1b_sigma_tau_A1')
	ap.add_argument('--teacher', default=f'{D}/data/MicSig/v1_teacher_amba_s1.npy')
	ap.add_argument('--stats', default=f'{D}/data/MicSig/v1_teacher_amba_s1.stats.npz')
	ap.add_argument('--tar', default=f'{D}/data/MicSig/simu_paired_pretrain_v2.tar')
	ap.add_argument('--val-scenes', type=int, default=500)
	ap.add_argument('--bs', type=int, default=128)
	ap.add_argument('--n-batches', type=int, default=400)
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	sel = json.load(open(os.path.join(a.sel, 'selected.json')))['selected']['spec']
	sigma, temp = float(sel['sigma']), float(sel['temp'])
	st = np.load(a.stats)
	mean, std = st['mean'].astype(np.float32), st['std'].astype(np.float32)
	tok = np.load(a.teacher, mmap_mode='r')
	pooled = np.concatenate([((np.asarray(tok[b:b + 500], dtype=np.float32).reshape(-1, 128, 768) - mean) / std).mean(1)
							 for b in range(0, len(tok), 500)])
	z = np.load(os.path.join(a.sel, 'pca', 'pca_spec.npz'))
	cue = (pooled - z['mean']) @ z['components'].T
	cue /= np.linalg.norm(cue, axis=1, keepdims=True)
	print(f'[cue] {cue.shape}, sigma {sigma:.4f}, temp {temp} (temp does not enter the weights)', flush=True)

	m = np.load(a.tar + '.meta.npz', allow_pickle=True)
	k, n_sc = int(m['k']), int(m['n_scenes'])
	utt, spk = m['utt'].astype(np.int64), m['spk'].astype(str)
	u2s = {}
	for u, s in zip(utt, spk):
		assert u2s.setdefault(int(u), s) == s, 'one speaker per utterance'
	n_train = (n_sc - a.val_scenes) * k
	print(f'[data] {n_train} training clips, {len(u2s)} utterances, {len(set(u2s.values()))} speakers', flush=True)

	rng = np.random.default_rng(0)
	acc = {n: [] for n in ('m_exact', 'm_same', 'm_diff', 'c_exact', 'c_same', 'c_diff', 'ess', 'w_max')}
	has_exact = []
	for _ in range(a.n_batches):
		cl = rng.choice(n_train, a.bs, replace=False)
		u = utt[cl]; s = np.array([u2s[int(x)] for x in u])
		c = cue[u]
		d = np.sqrt(np.clip(2 - 2 * c @ c.T, 0, None))  # unit vectors: |ci - cj| = sqrt(2 - 2 cos)
		w = np.exp(-(d / sigma) ** 2)
		np.fill_diagonal(w, 0)
		w /= w.sum(1, keepdims=True) + 1e-8
		off = ~np.eye(a.bs, dtype=bool)
		ex = (u[:, None] == u[None]) & off
		same = (s[:, None] == s[None]) & off & ~ex
		diff = (s[:, None] != s[None])
		for n, msk in (('exact', ex), ('same', same), ('diff', diff)):
			acc[f'm_{n}'].append((w * msk).sum(1))
			acc[f'c_{n}'].append(msk.sum(1) / (a.bs - 1))
		acc['ess'].append(1 / (w ** 2).sum(1))
		acc['w_max'].append(w.max(1))
		has_exact.append(ex.any(1))
	acc = {n: np.concatenate(v) for n, v in acc.items()}
	he = np.concatenate(has_exact)
	res = {'sigma': sigma, 'temp': temp, 'bs': a.bs, 'n_anchors': int(len(he)),
		   'mean_mass': {n: float(acc[f'm_{n}'].mean()) for n in ('exact', 'same', 'diff')},
		   'chance_share': {n: float(acc[f'c_{n}'].mean()) for n in ('exact', 'same', 'diff')},
		   'mass_per_item_over_chance': {n: float(acc[f'm_{n}'].sum() / max(acc[f'c_{n}'].sum(), 1e-12))
										 for n in ('exact', 'same', 'diff')},
		   'anchors_with_exact_pair': float(he.mean()),
		   'exact_mass_when_present': float(acc['m_exact'][he].mean()) if he.any() else None,
		   'exact_mass_when_present_p90': float(np.percentile(acc['m_exact'][he], 90)) if he.any() else None,
		   'frac_anchors_exact_gt_0.8': float((acc['m_exact'] > 0.8).mean()),
		   'anchors_with_same_spk': float((acc['c_same'] > 0).mean()),
		   'same_mass_when_present': float(acc['m_same'][acc['c_same'] > 0].mean()),
		   'same_chance_when_present': float(acc['c_same'][acc['c_same'] > 0].mean()),
		   'ess': {'mean': float(acc['ess'].mean()), 'median': float(np.median(acc['ess'])),
				   'p10': float(np.percentile(acc['ess'], 10)), 'p90': float(np.percentile(acc['ess'], 90))},
		   'w_max_mean': float(acc['w_max'].mean())}

	# speaker 1-NN of the cue itself (leave-one-out over training utterances, cosine), vs chance
	tr_u = np.unique(utt[:n_train])
	sub = rng.choice(tr_u, min(4000, len(tr_u)), replace=False)
	cs = cue[sub]; sp = np.array([u2s[int(x)] for x in sub])
	sim = cs @ cs.T; np.fill_diagonal(sim, -np.inf)
	nn = sim.argmax(1)
	_, cnt = np.unique(sp, return_counts=True)
	res['spk_1nn_acc'] = float((sp[nn] == sp).mean())
	res['spk_1nn_chance'] = float(((cnt / cnt.sum()) ** 2).sum())
	res['spk_1nn_n'] = int(len(sub))
	json.dump(res, open(a.out, 'w'), indent=1)
	print(json.dumps(res, indent=1))


if __name__ == '__main__':
	main()
