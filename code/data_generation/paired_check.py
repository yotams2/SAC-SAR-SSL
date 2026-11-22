""" Checks for gen_simu_paired.py output. Usage: python paired_check.py <full tar> <rerun tar of scenes [0, 2)> """
import io, os, sys, tarfile
import numpy as np, soundfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ssamba_spatial_pretrain import TarMicSigDataset  # noqa: E402

tar, rerun = sys.argv[1:3]
m = dict(np.load(tar + '.meta.npz', allow_pickle=True))
k, S, A = int(m['k']), int(m['n_scenes']), m['assignment']
n = len(m['clip'])
assert n == S * k and (m['clip'] == np.arange(n)).all(), 'clip ids'
# 1. design
assert all(len(set(row)) == k for row in A), 'duplicate utterance within a scene'
assert (np.bincount(A.ravel(), minlength=S) == k).all(), 'utterance not in exactly k scenes'
assert (m['scene'] == np.arange(n) // k).all() and (m['utt'] == A.ravel()).all()
print(f'[check] design: {S} scenes x {k}, every utterance in {k} scenes, no in-scene duplicates')
# 2. same utterance -> identical dry source; different utterances differ
for u in range(S):
	c = np.where(m['utt'] == u)[0]
	for key in ('src_md5', 'spk', 'utt_path', 'crop'):
		assert len(set(m[key][c].tolist())) == 1, (u, key)
print(f'[check] same utterance -> identical dry source (md5, speaker, file, crop); '
	  f'{len(set(m["src_md5"]))} distinct sources for {S} utterances, {len(set(m["spk"]))} speakers')
# 3. same scene -> identical geometry + acoustics; SNR independent
for r in range(S):
	c = np.arange(r * k, r * k + k)
	for key in ('room_sz', 'mic_pos', 'src_traj_pts', 'TDOA', 'DRR', 'C50', 'T60_edc'):
		v = m[key][c]
		assert np.allclose(v, v[0]), (r, key)
	assert len(set(np.round(m['snr'][c], 6))) == k
print('[check] same scene -> identical room / mics / source position / TDOA / DRR / C50 / T60; SNR per clip')
# 4. tar readable by the pretraining loader
ds = TarMicSigDataset(tar, 'pretrain')
assert len(ds) == n
x = np.stack([ds[i] for i in range(n)])
assert x.shape[1:] == (65792, 2) and np.isfinite(x).all() and np.abs(x).max() <= 0.9 + 1e-3, x.shape
print(f'[check] TarMicSigDataset: {len(ds)} clips, shape {x.shape[1:]}, peak {np.abs(x).max():.3f}')
# 5. physics: GCC-PHAT TDOA from the signal vs the annotation
def gcc(sig, fs=16000, maxlag=64):
	X = np.fft.rfft(sig, n=2 * sig.shape[0], axis=0)
	R = X[:, 0] * np.conj(X[:, 1]); cc = np.fft.irfft(R / (np.abs(R) + 1e-12))
	cc = np.concatenate((cc[-maxlag:], cc[:maxlag + 1]))
	return (np.argmax(cc) - maxlag) / fs
est = np.array([gcc(x[i]) for i in range(n)]); ann = m['TDOA'].reshape(n, -1)[:, 0]
err = np.abs(est - ann) * 16000
for sgn in (1, -1):  # channel-order convention of the annotation
	e = np.abs(sgn * est - ann) * 16000
	print(f'[check] GCC-PHAT vs annotated TDOA (sign {sgn:+d}): median {np.median(e):.2f} samples, within 2: {np.mean(e <= 2):.0%}')
# 6. distribution vs the priors
for key in ('T60_edc', 'DRR', 'TDOA', 'snr', 'scene_tries'):
	v = m[key].astype(float).ravel(); print(f'[stats] {key}: min {v.min():.3g} median {np.median(v):.3g} max {v.max():.3g}')
# 7. determinism: a separate rerun of scenes [0, 2) gives the same bytes
with tarfile.open(tar) as a, tarfile.open(rerun) as b:
	names = b.getnames(); assert len(names) == 2 * k
	same = all(a.extractfile(nm).read() == b.extractfile(nm).read() for nm in names)
print(f'[check] rerun of scenes 0-1 byte-identical: {same}'); assert same
print('[check] all OK')
