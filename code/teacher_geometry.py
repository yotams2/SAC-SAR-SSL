""" Teacher representation analysis (Yotam, 28 Nov), steps 1-3. CPU only, cached embeddings only (nothing recomputed).

Teachers (exactly as the V1b trainer builds its cues, _swap_pretrain_v1.set_v1_targets):
  A' (room):   rir_encoder/a1_design1 emb_*.npy (768-d = attention-pool half + late-mean half), z-scored with emb_stats.npz
               ("before") -> centre + project on pca_spat.npz (k = 768, a rotation) -> L2 ("after" = the V1b cue).
  speech:      v1_teacher_amba_s1.npy tokens (n_utt, 8, 16, 768) -> per-dim z-score (.stats.npz) -> mean over the 128
               tokens ("before") -> centre + pca_spec.npz (k = 108) -> L2 ("after" = the V1b cue).
  oracle B:    the 6 z-scored simulator parameters (TDOA, T60, DRR, C50, volume, source-mic distance), Euclidean (reference).
  HC:          v1b_sigma_tau_HC/hc_cues.npy, the 5 hand-crafted C + SAC spatial cues per scene, Euclidean (reference).
Scenes (as the cue check, v1b_cue_check.py):
  'cue' set (A', B):  fit on paired v2 [0, 12000) + unpaired8 [0, 22500); score on paired v2 [12000, 12500) + unpaired8
                      [22500, 25000) (3,000 held-out rooms).
  'paired' set (A', B, HC; the set HC exists on): fit on paired v2 [0, 12000), score on [12000, 12500).
  speech: utterances of the paired v2 set; held out = the 1,884 utterances used by the held-out scenes [12000, 12500),
          fit on the other 10,616 (speaker labels from the tar meta; WSJ0 read speech, so no SC2 labels exist here).
Step 1: singular-value spectrum, effective rank exp(H(s / sum s)) (Roy & Vetterli), participation ratio (sum l)^2 / sum l^2
        (l = s^2), share of dims with variance < 1% of the median, alignment / uniformity (Wang & Isola 2020, alpha = 2,
        t = 2) on the L2-normalised centred cue. Positives for alignment: spatial teachers = each held-out scene and its
        nearest held-out scene in oracle-B space (the acoustically closest room; the cues are scene-level, so there is
        no augmentation pair); speech = a random held-out utterance of the same speaker.
Step 2: ridge probes, the standard protocol of rir_encoder.ridge_cv (z-scored X, intercept, 33 lambdas 1e-3..1e5, 5-fold
        CV MAE); SID = multi-output ridge on one-hot speakers, lambda by CV accuracy, argmax.
Step 3: per label, the probe's direction(s) mapped back to cue space (w / sd), orthonormalised; for held-out pairs the
        squared cue distance is split into the in-subspace part per label, the joint span of all labels, and the residual.
        Mean share per pair (and the ratio of sums); chance share of a random m-dim subspace = m / d. Spearman of the
        in-subspace distance with the true label distance (|delta label|; SID: 1[different speaker]).
Writes <out>/geometry.json, sv_spectrum.png, distance_shares.png, and <cache>/heldout_cues.npz (held-out paired-v2 cues
for step 4). Usage: python teacher_geometry.py --out <results dir> --cache <scratch dir>
"""
import argparse
import json
import os
import time

import numpy as np

X = '/scratch200/yotams2/SAC-SAR-SSL'
RIR = f'{X}/exp/rir_encoder/a1_design1'
CEIL = f'{X}/exp/rir_encoder/ceiling'
PCA = f'{X}/exp/v1b_sigma_tau_A1/pca'
HC = f'{X}/exp/v1b_sigma_tau_HC/hc_cues'
TEACH = f'{X}/data/MicSig/v1_teacher_amba_s1'
META = f'{X}/data/MicSig/simu_paired_pretrain_v2.tar.meta.npz'
LABELS = ['TDOA', 'T60', 'DRR', 'C50', 'volume', 'src_mic_dist']
REVERB = ['T60', 'DRR', 'C50']
NP_TR, NU_TR = 12000, 22500
LAMS = np.logspace(-3, 5, 33)


def l2(x):
	return x / np.linalg.norm(x, axis=1, keepdims=True)


def rank(v):
	return np.argsort(np.argsort(v, kind='stable'), kind='stable').astype(np.float64)


def spearman(a, b):
	return float(np.corrcoef(rank(a), rank(b))[0, 1])


# ---------------------------------------------------------------- ridge (rir_encoder.ridge_cv, + the fitted weights)
def _svd_fit(A, Y):
	am = A.mean(0)
	U, s, Vt = np.linalg.svd(A - am, full_matrices=False)
	return am, Y.mean(0), U, s, Vt


def ridge_cv(Xtr, ytr, Xte, k=5, seed=0, classify=False):
	""" -> test predictions, weights in the ORIGINAL X space (d,) or (d, C), lambda, edge. classify: ytr = int labels,
		one-hot multi-output, CV accuracy. """
	mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
	Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
	if classify:
		cls = np.unique(ytr)
		Y = (ytr[:, None] == cls[None]).astype(np.float64)
	else:
		Y = ytr[:, None].astype(np.float64)
	folds = np.random.RandomState(seed).permutation(len(Ztr)) % k

	def weights(A, Yt):
		am, ym, U, s, Vt = _svd_fit(A, Yt)
		uy = U.T @ (Yt - ym)
		return am, ym, [Vt.T @ ((s / (s ** 2 + l))[:, None] * uy) for l in LAMS]

	cv = np.zeros(len(LAMS))
	for f in range(k):
		tr, va = folds != f, folds == f
		am, ym, Ws = weights(Ztr[tr], Y[tr])
		for i, W in enumerate(Ws):
			p = (Ztr[va] - am) @ W + ym
			cv[i] += -(p.argmax(1) == np.searchsorted(cls, ytr[va])).sum() if classify else np.abs(p[:, 0] - Y[va, 0]).sum()
	i = int(np.argmin(cv))
	am, ym, Ws = weights(Ztr, Y)
	W = Ws[i]
	pred = (Zte - am) @ W + ym
	W_orig = W / sd[:, None]  # prediction = ((x - mu) / sd) . w  ->  direction in cue space = w / sd
	if classify:
		return cls[pred.argmax(1)], W_orig, float(LAMS[i]), i in (0, len(LAMS) - 1)
	return pred[:, 0], W_orig[:, 0], float(LAMS[i]), i in (0, len(LAMS) - 1)


def orth(V, tol=1e-6):
	""" orthonormal basis of the column span of V (d, m). """
	V = V.reshape(V.shape[0], -1)
	U, s, _ = np.linalg.svd(V, full_matrices=False)
	return U[:, s > tol * s.max()]


def energy_basis(W, frac=0.9):
	""" the top singular directions of W (d, C) holding `frac` of its squared norm. """
	U, s, _ = np.linalg.svd(W - W.mean(1, keepdims=True), full_matrices=False)
	e = np.cumsum(s ** 2) / (s ** 2).sum()
	return U[:, :int(np.searchsorted(e, frac) + 1)]


# ---------------------------------------------------------------- step 1
def spread(Xtr, Xte, pos):
	""" Xtr fit rows (spectrum, variances), Xte held-out rows (alignment / uniformity), pos (n_te,) positive partner. """
	C = Xtr - Xtr.mean(0)
	s = np.linalg.svd(C, compute_uv=False)
	p = s / s.sum()
	lam = s ** 2
	var = C.var(0)
	U = l2(Xte - Xtr.mean(0))
	rs = np.random.RandomState(0)
	i, j = rs.randint(0, len(U), 20000), rs.randint(0, len(U), 20000)
	keep = i != j
	sq = ((U[i[keep]] - U[j[keep]]) ** 2).sum(1)
	ok = pos >= 0
	align = float(((U[ok] - U[pos[ok]]) ** 2).sum(1).mean())
	return {'dim': int(Xtr.shape[1]), 'effective_rank': float(np.exp(-(p * np.log(p + 1e-300)).sum())),
			'participation_ratio': float(lam.sum() ** 2 / (lam ** 2).sum()),
			'top1_var_share': float(lam[0] / lam.sum()), 'top10_var_share': float(lam[:10].sum() / lam.sum()),
			'dims_var_below_1pct_median': float((var < 0.01 * np.median(var)).mean()),
			'alignment': align, 'alignment_random_pairs': float(sq.mean()),
			'alignment_ratio': align / float(sq.mean()),
			'uniformity': float(np.log(np.exp(-2 * sq).mean())), 'sv_normalised': (s / s[0]).tolist()}


# ---------------------------------------------------------------- step 3
def decompose(Xte, bases, pairs, labdist):
	""" Xte held-out cue rows; bases {name: (d, m) orthonormal}; pairs (i, j); labdist {label: (n_pairs,)}. """
	i, j = pairs
	d = Xte[i] - Xte[j]
	D = (d ** 2).sum(1)
	ok = D > 1e-12
	out = {'n_pairs': int(ok.sum()), 'dim': int(Xte.shape[1])}
	for name, Q in bases.items():
		part = ((d @ Q) ** 2).sum(1)
		r = {'m': int(Q.shape[1]), 'chance_share': Q.shape[1] / Xte.shape[1],
			 'mean_share': float((part[ok] / D[ok]).mean()), 'sum_share': float(part[ok].sum() / D[ok].sum())}
		lab = name[len('label:'):] if name.startswith('label:') else None
		if lab in labdist:
			r['spearman_in_subspace_vs_label'] = spearman(np.sqrt(part[ok]), labdist[lab][ok])
			r['spearman_full_distance_vs_label'] = spearman(np.sqrt(D[ok]), labdist[lab][ok])
			r['spearman_residual_vs_label'] = spearman(np.sqrt(np.maximum(D - part, 0)[ok]), labdist[lab][ok])
		out[name] = r
	return out


def spatial_geometry(Xtr, Xte, Btr_nat, Bte_nat, pairs):
	""" ridge directions for the 6 labels in cue space, then the distance split on held-out pairs, for two choices of
		direction: 'readout' = the probe weight w / sd (as specified), 'pattern' = Cov(X) w (the encoding pattern,
		Haufe et al. 2014: the direction along which the cue moves with the label; the readout can point into
		low-variance dims that carry little distance). """
	probes, dirs = {}, {}
	for c, lab in enumerate(LABELS):
		p, w, lam, edge = ridge_cv(Xtr, Btr_nat[:, c], Xte)
		y = Bte_nat[:, c]
		probes[lab] = {'mae': float(np.abs(p - y).mean()), 'mean_baseline': float(np.abs(Btr_nat[:, c].mean() - y).mean()),
					   'r2': float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 'lam': lam, 'edge': edge}
		dirs[lab] = w
	C = np.cov(Xtr.T)
	var = np.diag(C)
	low = var < 0.01 * np.median(var)
	i, j = pairs
	labdist = {lab: np.abs(Bte_nat[i, c] - Bte_nat[j, c]) for c, lab in enumerate(LABELS)}
	geo = {}
	for kind, D in (('readout', dirs), ('pattern', {k: C @ v for k, v in dirs.items()})):
		bases = {f'label:{lab}': orth(D[lab][:, None]) for lab in LABELS}
		bases['span:reverb(T60,DRR,C50)'] = orth(np.stack([D[l] for l in REVERB], 1))
		bases['span:all6'] = orth(np.stack([D[l] for l in LABELS], 1))
		g = decompose(Xte, bases, pairs, labdist)
		g['residual_mean_share'] = 1 - g['span:all6']['mean_share']
		g['residual_sum_share'] = 1 - g['span:all6']['sum_share']
		g['dir_weight2_on_lowvar_dims'] = {lab: float((D[lab][low] ** 2).sum() / (D[lab] ** 2).sum()) for lab in LABELS}
		g['lowvar_dims_frac'] = float(low.mean())
		geo[kind] = g
	return probes, geo


def auc(score, pos):
	""" P(score of a positive > score of a negative), ties half. """
	r = rank(score) + 1
	n1, n0 = pos.sum(), (~pos).sum()
	return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def all_pairs(n):
	i, j = np.triu_indices(n, 1)
	return i, j


def random_pairs(n, m, seed=0):
	rs = np.random.RandomState(seed)
	i, j = rs.randint(0, n, m), rs.randint(0, n, m)
	keep = i != j
	return i[keep], j[keep]


def nn_partner(B):
	""" index of each row's nearest other row (Euclidean, z-scored oracle-B space). """
	G = (B ** 2).sum(1)
	D = G[:, None] + G[None] - 2 * B @ B.T
	np.fill_diagonal(D, np.inf)
	return D.argmin(1)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--out', required=True)
	ap.add_argument('--cache', required=True)
	a = ap.parse_args()
	os.makedirs(a.out, exist_ok=True)
	os.makedirs(a.cache, exist_ok=True)
	t0 = time.time()
	log = lambda s: print(f'[geom {time.time() - t0:5.0f}s] {s}', flush=True)
	# ---- spatial teachers
	st = np.load(f'{RIR}/emb_stats.npz')
	pz = np.load(f'{PCA}/pca_spat.npz')
	Ez = {n: (np.load(f'{RIR}/emb_{n}.npy').astype(np.float64) - st['mean']) / st['std'] for n in ('paired_v2', 'unpaired8')}
	Ac = {n: l2((v - pz['mean']) @ pz['components'].T) for n, v in Ez.items()}
	bst = np.load(f'{CEIL}/oracle_B_stats.npz')
	Bz = {n: np.load(f'{CEIL}/oracle_B_{n}.npy').astype(np.float64) for n in ('paired_v2', 'unpaired8')}
	Bn = {n: v * bst['std'] + bst['mean'] for n, v in Bz.items()}
	assert list(bst['names']) == LABELS, bst['names']
	hc = np.load(f'{HC}.npy').astype(np.float64)
	# cue set (paired + unpaired) and paired set
	cat = lambda d, tr: np.concatenate([d['paired_v2'][:NP_TR] if tr else d['paired_v2'][NP_TR:],
										d['unpaired8'][:NU_TR] if tr else d['unpaired8'][NU_TR:]])
	sets = {'cue': {'A_before': (cat(Ez, 1), cat(Ez, 0)), 'A_after': (cat(Ac, 1), cat(Ac, 0)), 'B': (cat(Bz, 1), cat(Bz, 0)),
					'_Bnat': (cat(Bn, 1), cat(Bn, 0))},
			'paired': {'A_before': (Ez['paired_v2'][:NP_TR], Ez['paired_v2'][NP_TR:]),
					   'A_after': (Ac['paired_v2'][:NP_TR], Ac['paired_v2'][NP_TR:]),
					   'B': (Bz['paired_v2'][:NP_TR], Bz['paired_v2'][NP_TR:]), 'HC': (hc[:NP_TR], hc[NP_TR:]),
					   '_Bnat': (Bn['paired_v2'][:NP_TR], Bn['paired_v2'][NP_TR:])}}
	log('spatial teachers loaded: ' + ', '.join(f'{s}:{k} {v[0].shape}/{v[1].shape}' for s, d in sets.items() for k, v in d.items()))
	# ---- speech teacher (pooled z-scored tokens; cached after the first run)
	pooled_f = f'{a.cache}/speech_pooled_z.npy'
	if os.path.exists(pooled_f):
		S = np.load(pooled_f).astype(np.float64)
	else:
		ts = np.load(f'{TEACH}.stats.npz')
		tok = np.load(f'{TEACH}.npy', mmap_mode='r')
		S = np.zeros((len(tok), tok.shape[-1]))
		for b in range(0, len(tok), 250):
			x = np.asarray(tok[b:b + 250], dtype=np.float32).reshape(-1, 128, tok.shape[-1])
			S[b:b + 250] = ((x - ts['mean']) / ts['std']).mean(1)
			if b % 2500 == 0:
				log(f'speech pooled {b}/{len(tok)}')
		np.save(pooled_f, S.astype(np.float32))
	ps = np.load(f'{PCA}/pca_spec.npz')
	Sc = l2((S - ps['mean']) @ ps['components'].T)
	m = np.load(META, allow_pickle=True)
	spk_of = {}
	for u, s_ in zip(m['utt'], m['spk']):
		spk_of[int(u)] = str(s_)
	spk_names = np.array([spk_of[u] for u in range(len(S))])
	spk = np.unique(spk_names, return_inverse=True)[1]
	te_utt = np.zeros(len(S), bool)
	te_utt[np.unique(m['assignment'][NP_TR:])] = True
	speech = {'S_before': (S[~te_utt], S[te_utt]), 'S_after': (Sc[~te_utt], Sc[te_utt])}
	ytr, yte = spk[~te_utt], spk[te_utt]
	log(f'speech teacher: {S.shape}, cue k {Sc.shape[1]}, {te_utt.sum()} held-out utts, {len(np.unique(spk))} speakers, '
		f'{len(np.setdiff1d(yte, ytr))} held-out speakers unseen in fit')
	res = {'protocol': __doc__.split('Writes')[0], 'step1': {}, 'step2': {}, 'step3': {}}
	# ---- step 1
	for sname, d in sets.items():
		pos = nn_partner(d['B'][1])
		for t, (tr, te) in d.items():
			if not t.startswith('_'):
				res['step1'][f'{sname}/{t}'] = spread(tr, te, pos)
	rs = np.random.RandomState(1)
	pos = np.full(len(yte), -1)
	for c in np.unique(yte):
		idx = np.where(yte == c)[0]
		if len(idx) > 1:
			pos[idx] = rs.permutation(idx) if len(idx) > 2 else idx[::-1]
			bad = pos[idx] == idx
			while bad.any():  # no self-pairs
				pos[idx] = rs.permutation(idx)
				bad = pos[idx] == idx
	for t, (tr, te) in speech.items():
		res['step1'][f'speech/{t}'] = spread(tr, te, pos)
	log('step 1 done')
	# ---- steps 2 + 3, spatial
	ref = json.load(open(f'{RIR}/probes.json'))
	res['step2']['random_init_reference_A_cue_set'] = ref.get('random_init')
	res['step2']['trained_reference_A_cue_set_rir_encoder'] = ref.get('trained')
	for sname, d in sets.items():
		Bt, Bh = d['_Bnat']
		pairs = random_pairs(len(Bh), 10000) if sname == 'cue' else all_pairs(len(Bh))
		for t in ('A_before', 'A_after', 'B', 'HC'):
			if t not in d:
				continue
			probes, geo = spatial_geometry(d[t][0], d[t][1], Bt, Bh, pairs)
			res['step2'][f'{sname}/{t}'] = probes
			if t != 'A_before':
				res['step3'][f'{sname}/{t}'] = geo
			log(f'steps 2-3 {sname}/{t}: ' + ', '.join(f'{k} {v["mae"]:.3f}' for k, v in probes.items()) +
				(f' | residual share readout {geo["readout"]["residual_mean_share"]:.4f} pattern {geo["pattern"]["residual_mean_share"]:.4f}' if t != 'A_before' else ''))
	# ---- steps 2 + 3, speech (SID)
	pairs = random_pairs(len(yte), 100000)
	same = [np.where(yte == c)[0] for c in np.unique(yte)]
	si = np.concatenate([np.repeat(g, len(g)) for g in same]); sj = np.concatenate([np.tile(g, len(g)) for g in same])
	keep = si != sj
	pairs = (np.concatenate([pairs[0], si[keep]]), np.concatenate([pairs[1], sj[keep]]))  # + all same-speaker pairs
	for t, (tr, te) in speech.items():
		p, W, lam, edge = ridge_cv(tr, ytr, te, classify=True)
		chance = float((yte == np.bincount(ytr).argmax()).mean())
		res['step2'][f'speech/{t}'] = {'SID': {'acc': float((p == yte).mean()), 'majority_chance': chance,
											   'n_classes': int(len(np.unique(ytr))), 'lam': lam, 'edge': edge}}
		log(f'step 2 speech/{t}: SID acc {(p == yte).mean():.3f} (chance {chance:.3f}, lam {lam:g})')
		if t == 'S_after':
			i, j = pairs
			Wc = W - W.mean(1, keepdims=True)
			Cs = np.cov(tr.T)
			lab = (yte[i] != yte[j]).astype(np.float64)
			geo = {}
			for kind, M in (('readout', Wc), ('pattern', Cs @ Wc)):
				bases = {'label:SID': orth(M), 'label:SID_energy90': energy_basis(M, 0.9), 'label:SID_energy50': energy_basis(M, 0.5)}
				g = decompose(te, bases, pairs, {'SID': lab, 'SID_energy90': lab, 'SID_energy50': lab})
				dd = te[i] - te[j]
				for n, Q in bases.items():
					g[n]['auc_different_vs_same_speaker'] = auc(np.sqrt(((dd @ Q) ** 2).sum(1)), lab > 0)
				g['auc_full_distance'] = auc(np.sqrt((dd ** 2).sum(1)), lab > 0)
				g['residual_mean_share_vs_energy90'] = 1 - g['label:SID_energy90']['mean_share']
				geo[kind] = g
			geo['same_speaker_pair_frac'] = float(1 - lab.mean())
			res['step3']['speech/S_after'] = geo
			log(f"step 3 speech: SID energy90 share readout {geo['readout']['label:SID_energy90']['mean_share']:.3f} "
				f"pattern {geo['pattern']['label:SID_energy90']['mean_share']:.3f} (chance {geo['pattern']['label:SID_energy90']['chance_share']:.3f})")
	log('steps 2-3 done')
	# ---- held-out paired-v2 cues for step 4 (per scene / per utterance; step 4 maps clips to them)
	np.savez(f'{a.cache}/heldout_cues.npz', scenes=np.arange(NP_TR, 12500), A=Ac['paired_v2'][NP_TR:], HC=hc[NP_TR:],
			 B=Bz['paired_v2'][NP_TR:], S_all=Sc, assignment=m['assignment'])
	json.dump(res, open(f'{a.out}/geometry.json', 'w'), indent=1)
	plots(res, a.out)
	log(f'-> {a.out}/geometry.json')


def plots(res, out):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	names = [('paired/A_before', "A′ before (z-scored 768)"), ('paired/A_after', "A′ after (V1b cue)"),
			 ('speech/S_before', 'speech before (pooled z 768)'), ('speech/S_after', 'speech after (V1b cue, k 108)'),
			 ('paired/B', 'oracle B (6)'), ('paired/HC', 'HC (5)')]
	fig, ax = plt.subplots(figsize=(7, 4.5))
	for k, lab in names:
		sv = np.array(res['step1'][k]['sv_normalised'])
		ax.loglog(np.arange(1, len(sv) + 1), sv, label=f"{lab}: erank {res['step1'][k]['effective_rank']:.1f}")
	ax.set_xlabel('index')
	ax.set_ylabel('singular value / largest')
	ax.set_title('Teacher singular-value spectra (fit scenes / utterances)')
	ax.grid(alpha=0.3, which='both')
	ax.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(f'{out}/sv_spectrum.png', dpi=150)
	plt.close(fig)
	# stacked shares: TDOA, reverb span, other (all6 minus reverb minus TDOA), residual; speech: SID span, residual
	rows = [('paired/A_after', "A′"), ('paired/HC', 'HC'), ('paired/B', 'oracle B'), ('cue/A_after', "A′ (cue set)")]
	fig, ax = plt.subplots(figsize=(7, 3.8))
	for r, (k, lab) in enumerate(rows):
		g = res['step3'][k]['pattern']
		tdoa, rev, all6 = g['label:TDOA']['mean_share'], g['span:reverb(T60,DRR,C50)']['mean_share'], g['span:all6']['mean_share']
		parts = [tdoa, rev, max(all6 - tdoa - rev, 0), 1 - all6]
		left = 0
		for v, c, n in zip(parts, ['#4C78A8', '#F58518', '#B0B0B0', '#E8E8E8'], ['TDOA dir', 'reverb span (T60, DRR, C50)', 'rest of 6-label span', 'residual']):
			ax.barh(r, v, left=left, color=c, label=n if r == 0 else None, edgecolor='white')
			left += v
	g = res['step3']['speech/S_after']['pattern']
	r = len(rows)
	for v, l0, c, n in ((g['label:SID_energy90']['mean_share'], 0, '#54A24B', 'SID span (90% energy)'),
						(1 - g['label:SID_energy90']['mean_share'], g['label:SID_energy90']['mean_share'], '#E8E8E8', None)):
		ax.barh(r, v, left=l0, color=c, label=n, edgecolor='white')
	ax.set_yticks(range(len(rows) + 1), [l for _, l in rows] + ['speech'])
	ax.invert_yaxis()
	ax.set_xlim(0, 1)
	ax.set_xlabel('mean share of squared held-out pair distance, encoding-pattern subspaces\n(TDOA and reverb directions are not orthogonal: approximate stack)')
	ax.legend(fontsize=7, loc='lower right')
	fig.tight_layout()
	fig.savefig(f'{out}/distance_shares.png', dpi=150)
	plt.close(fig)


if __name__ == '__main__':
	main()
