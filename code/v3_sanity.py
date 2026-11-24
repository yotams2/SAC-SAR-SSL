""" V3 loader sanity check + dry run (CPU, 2026-11-24). Analysis only, no training.

1. Room consistency of the swap target: per half, the frame-averaged cos / sin IPD per STFT bin (512 / 256, bins
   1-256 = the spatial input's band; frames weighted by |L||R|) -> a 512-d room signature. Pearson r between
   h1 and h2 of the same clip (same room: should be high) vs h1 of clip i and h1 of a random other clip (low), and
   between the target and the V3 spectral view (other room by design: should be low, like the different-clip row).
2. WPE alignment: lag of the cross-correlation peak between WPE(h) and the recorded mid of h (should be ~0).
3. T60 of the RIR pool (paired v2 training scenes) vs the unpaired training clips (meta T60_edc quantiles).
4. Dry run: V3Dataset train / val through a DataLoader (4 workers): shapes, finiteness, s / item, val determinism.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_data import V3Dataset, HALF  # noqa: E402

M = '/scratch200/yotams2/SAC-SAR-SSL/data/MicSig'
TAR, WPE, POOL = f'{M}/simu_unpaired8_pretrain.tar', f'{M}/simu_unpaired8_wpe', f'{M}/simu_paired_pretrain_v2.tar.rirs.npz'
N_TR, N_ALL = 22500, 25000


def signature(x):
	""" (HALF, 2) -> 512-d frame-averaged (cos IPD, sin IPD) over bins 1..256, weighted by |L||R|. """
	w = np.hanning(512).astype(np.float32)
	fr = np.lib.stride_tricks.sliding_window_view(x, 512, axis=0)[::256]  # (nt, 2, 512)
	S = np.fft.rfft(fr * w, axis=-1)[:, :, 1:257]  # (nt, 2, 256)
	c = S[:, 0] * np.conj(S[:, 1])
	m = np.abs(c) + 1e-12
	wt = m / m.sum(0, keepdims=True)
	u = c / m
	return np.concatenate([(wt * u.real).sum(0), (wt * u.imag).sum(0)])


def pr(a, b):
	return float(np.corrcoef(a, b)[0, 1])


def main(out):
	t0 = time.time()
	res = {}
	ds = V3Dataset(TAR, WPE, POOL, 0, N_TR)
	print(f'[init] {time.time() - t0:.1f} s, pool {ds.rir.shape}', flush=True)
	rng = np.random.default_rng(0)
	idx = rng.choice(N_TR, 300, replace=False)
	same, other, specv, lags = [], [], [], []
	sig = {}
	for i in idx:
		x = ds.clips[int(i)]
		sig[int(i)] = (signature(x[:HALF]), signature(x[HALF:]))
		if len(sig) % 50 == 0:
			print(f'[sig] {len(sig)} / 300, {time.time() - t0:.0f} s', flush=True)
	keys = list(sig)
	for n, i in enumerate(keys):
		j = keys[(n + 1 + int(rng.integers(len(keys) - 1))) % len(keys)]
		same.append(pr(sig[i][0], sig[i][1]))
		other.append(pr(sig[i][0], sig[j][0]))
	for i in keys[:100]:
		trip, _ = ds.item(ds.targets.index(i))
		specv.append(pr(signature(trip[0]), signature(trip[1])))
		assert np.allclose(trip[0], ds.clips[i][:HALF]) and np.allclose(trip[2], ds.clips[i][HALF:])
		for h in (0, 1):
			mid = ds.clips[i][h * HALF:(h + 1) * HALF].mean(1)
			wp = ds.wpe(i, h)
			n_ = 1 << 18
			xc = np.fft.irfft(np.fft.rfft(mid, n_) * np.conj(np.fft.rfft(wp, n_)), n_)
			lag = int(np.argmax(np.abs(xc)))
			lags.append(lag if lag < n_ // 2 else lag - n_)
	q = lambda v: {'mean': float(np.mean(v)), 'median': float(np.median(v)), 'p10': float(np.percentile(v, 10)),
				   'p90': float(np.percentile(v, 90)), 'n': len(v)}
	res['ipd_signature_r'] = {'h1_vs_h2_same_clip': q(same), 'h1_vs_h1_other_clip': q(other),
							  'target_vs_v3_spectral_view': q(specv)}
	res['wpe_lag_samples'] = {'abs_max': int(np.max(np.abs(lags))), 'frac_zero': float(np.mean(np.array(lags) == 0)),
							  'median': float(np.median(lags))}
	print(json.dumps(res, indent=1), flush=True)

	mu = np.load(f'{TAR}.meta.npz', allow_pickle=True)
	mp = np.load(f'{M}/simu_paired_pretrain_v2.tar.meta.npz', allow_pickle=True)
	pool_t60 = mp['T60_edc'].reshape(-1, int(mp['k']))[:12000, 0]
	un_t60 = mu['T60_edc'][:N_TR]
	pc = [5, 25, 50, 75, 95]
	res['t60_edc_quantiles_5_25_50_75_95'] = {'rir_pool_paired_v2_train': np.percentile(pool_t60, pc).round(3).tolist(),
											  'unpaired8_train': np.percentile(un_t60, pc).round(3).tolist()}
	print(json.dumps(res['t60_edc_quantiles_5_25_50_75_95']), flush=True)

	val = V3Dataset(TAR, WPE, POOL, N_TR, N_ALL, fixed_seed=12345)
	a1, _ = val[7]; a2, _ = val[7]
	res['val_deterministic'] = bool(np.array_equal(a1, a2))
	tr = V3Dataset(TAR, WPE, POOL, 0, N_TR, swap_halves=True)
	t1 = time.time()
	dl = torch.utils.data.DataLoader(torch.utils.data.Subset(tr, list(range(0, N_TR, 90))), batch_size=32, num_workers=4)
	n, ok = 0, True
	for sig3, ids in dl:
		ok &= tuple(sig3.shape[1:]) == (3, HALF, 2) and bool(torch.isfinite(sig3).all()) and tuple(ids.shape[1:]) == (3, 2)
		n += sig3.shape[0]
	dt = time.time() - t1
	res['dry_run'] = {'items': n, 'shapes_finite_ok': bool(ok), 's_per_item_4_workers': dt / n,
					  'est_s_per_batch128_5_workers': dt / n * 128 * 4 / 5}
	print(json.dumps({k: res[k] for k in ('val_deterministic', 'dry_run')}), flush=True)
	json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
	main(sys.argv[1])
