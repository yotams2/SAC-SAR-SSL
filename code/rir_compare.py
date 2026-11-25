""" A′-uniform ablation report (Yotam 11-25): RIR teachers side by side, all numbers computed the same way here.
Per model dir (rir_encoder.py output): held-out ridge probes TDOA / T60 / DRR / C50 (from its probes.json, 'trained', plus
'trained_meanpool' when present) and three Spearman correlations of the embedding distance (cosine distance of the
z-scored embedding, z-score over training rooms, as rir_encoder.py) on the same 10k random held-out room pairs
(RandomState(0), i != j, both sets): vs oracle-B Euclidean distance (overall), vs |dTDOA|, vs the reverb factor
distance (|d PC1| of oracle B's 5 reverberation dims, PC fitted on the paired training rooms; as sac_cue_check.py).
Reference rows: random init (from the uniform run) and a ridge probe on the true RIR length alone (log rir_len):
gpuRIR sets the RIR length from the simulated T60 (Tmax = Sabine time to decay 40 dB), so every encoder that can see
where the zero padding starts (A′ / v2 late envelope, A′-uniform's frame mask) can read T60 from it.
Pass test = A′ design 1's: TDOA <= 1.95 samples, T60 / DRR / C50 no worse than random init. Decision rule (Yotam):
A′-uniform within ~10% of A′ on all four probes -> candidate to replace A′; clearly worse on T60 / C50 only -> the late
energy view matters.
Usage: python rir_compare.py --uniform <dir> --a1 <dir> --v2 <dir> --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rir_encoder import D_DIR, SETS, FS, ridge_cv  # noqa: E402

CEIL = '/scratch200/yotams2/SAC-SAR-SSL/exp/rir_encoder/ceiling'
TASKS = ('TDOA', 'T60', 'DRR', 'C50')


def rank(v):
	return np.argsort(np.argsort(v)).astype(np.float64)


def spearman(a, b):
	return float(np.corrcoef(rank(a), rank(b))[0, 1])


def labels():
	lab, test, lens = {k: [] for k in TASKS}, [], []
	for name, tar, n_train in SETS:
		m = np.load(os.path.join(D_DIR, tar + '.meta.npz'), allow_pickle=True)
		z = np.load(os.path.join(D_DIR, tar + '.rirs.npz'))
		S = len(z['rir_len'])
		first = np.searchsorted(m['scene'], np.arange(S))
		g = lambda k: np.asarray(m[k], np.float64).reshape(len(m['scene']), -1)[first, 0]
		lab['TDOA'].append(g('TDOA') * FS); lab['T60'].append(g('T60_edc')); lab['DRR'].append(g('DRR')); lab['C50'].append(g('C50'))
		test.append(np.arange(S) >= n_train)
		lens.append(z['rir_len'])
	return {k: np.concatenate(v) for k, v in lab.items()}, np.concatenate(test), np.concatenate(lens)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--uniform', required=True)
	ap.add_argument('--a1', default='/scratch200/yotams2/SAC-SAR-SSL/exp/rir_encoder/a1_design1')
	ap.add_argument('--v2', default='/scratch200/yotams2/SAC-SAR-SSL/exp/rir_encoder/rirmae2_v2')
	ap.add_argument('--out', required=True)
	a = ap.parse_args()
	lab, test, lens = labels()
	B = np.concatenate([np.load(os.path.join(CEIL, f'oracle_B_{n}.npy')) for n, _, _ in SETS]).astype(np.float64)
	st = np.load(os.path.join(CEIL, 'oracle_B_stats.npz'))
	tdoa = B[:, 0] * st['std'][0] + st['mean'][0]
	R, ntr = B[:, 1:], SETS[0][2]
	_, _, Vt = np.linalg.svd(R[:ntr] - R[:ntr].mean(0), full_matrices=False)
	pc1 = (R - R[:ntr].mean(0)) @ Vt[0]
	ti = np.where(test)[0]
	rs = np.random.RandomState(0)
	i, j = ti[rs.randint(0, len(ti), 10000)], ti[rs.randint(0, len(ti), 10000)]
	i, j = i[i != j], j[i != j]
	ref = {'oracleB': np.linalg.norm(B[i] - B[j], axis=1), 'dTDOA': np.abs(tdoa[i] - tdoa[j]), 'reverb': np.abs(pc1[i] - pc1[j])}
	rows = {}
	for name, d in (('A′ design 1', a.a1), ('v2', a.v2), ('A′-uniform', a.uniform)):
		E = np.concatenate([np.load(os.path.join(d, f'emb_{n}.npy')) for n, _, _ in SETS]).astype(np.float64)
		assert len(E) == len(test), (name, E.shape)
		Z = (E - E[~test].mean(0)) / (E[~test].std(0) + 1e-6)
		Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
		de = 1 - (Zn[i] * Zn[j]).sum(1)
		p = json.load(open(os.path.join(d, 'probes.json')))
		rows[name] = {'probe_mae': {t: p['trained'][t]['mae'] for t in TASKS}, 'dim': int(E.shape[1]),
					  'spearman': {k: spearman(de, v) for k, v in ref.items()}}
		if 'trained_meanpool' in p:
			rows[name]['probe_mae_meanpool'] = {t: p['trained_meanpool'][t]['mae'] for t in TASKS}
	pu = json.load(open(os.path.join(a.uniform, 'probes.json')))
	rnd = {t: pu['random_init'][t]['mae'] for t in TASKS}
	X = np.log(lens.astype(np.float64))[:, None]
	lenrow = {t: float(np.abs(ridge_cv(X[~test], lab[t][~test], X[test])[0] - lab[t][test]).mean()) for t in TASKS}
	u, a1 = rows['A′-uniform']['probe_mae'], rows['A′ design 1']['probe_mae']
	rel = {t: u[t] / a1[t] for t in TASKS}
	passed = u['TDOA'] <= 1.95 and all(u[t] <= rnd[t] for t in ('T60', 'DRR', 'C50'))
	within = all(r <= 1.10 for r in rel.values())
	worse_late_only = all(rel[t] <= 1.10 for t in ('TDOA', 'DRR')) and any(rel[t] > 1.10 for t in ('T60', 'C50'))
	verdict = ('within 10% of A′ on all four probes: candidate to replace A′ (your call on rerunning V1b)' if within else
			   'worse than A′ by > 10% on T60 / C50 only: the late-energy view matters (report and stop)' if worse_late_only else
			   'mixed: worse than A′ by > 10% on ' + ', '.join(t for t in TASKS if rel[t] > 1.10))
	res = {'rows': rows, 'random_init': rnd, 'rir_len_only': lenrow, 'uniform_over_a1': rel, 'pass_test': passed,
		   'verdict': verdict, 'pairs': int(len(i))}
	json.dump(res, open(a.out, 'w'), indent=1)
	print('| model | TDOA | T60 | DRR | C50 | rho oracle-B | rho dTDOA | rho reverb |')
	print('|---|---|---|---|---|---|---|---|')
	for n, r in rows.items():
		m, s = r['probe_mae'], r['spearman']
		print(f"| {n} ({r['dim']}-d) | {m['TDOA']:.3f} | {m['T60']:.4f} | {m['DRR']:.3f} | {m['C50']:.3f} | "
			  f"{s['oracleB']:+.3f} | {s['dTDOA']:+.3f} | {s['reverb']:+.3f} |")
		if 'probe_mae_meanpool' in r:
			m = r['probe_mae_meanpool']
			print(f"| {n}, mean pool | {m['TDOA']:.3f} | {m['T60']:.4f} | {m['DRR']:.3f} | {m['C50']:.3f} | | | |")
	print(f"| random init (uniform arch) | {rnd['TDOA']:.3f} | {rnd['T60']:.4f} | {rnd['DRR']:.3f} | {rnd['C50']:.3f} | | | |")
	print(f"| log RIR length alone | {lenrow['TDOA']:.3f} | {lenrow['T60']:.4f} | {lenrow['DRR']:.3f} | {lenrow['C50']:.3f} | | | |")
	print(f"uniform / A′: " + ', '.join(f'{t} {rel[t]:.2f}x' for t in TASKS) + f"; pass test {'PASS' if passed else 'FAIL'}")
	print(f'verdict: {verdict}')


if __name__ == '__main__':
	main()
