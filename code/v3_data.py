""" V3 data (agreed 2026-11-23; loader built 2026-11-24, defaults from the 11-24 spec): unpaired swap on the 8.224-s
unpaired set, so no paired renderings are needed.

Each unpaired clip x (2 ch, 131,584 samples = 2 x 65,792; one room r, one utterance s) is split into halves h1, h2
(h1 = first half; --v3-swap-halves draws the order per access). One item = the same (3, 65792, 2) triplet as
swap_pretrain.PairedDataset, so the model, the loss and views() are unchanged:
  view 0, target        = h1 as recorded = x(s, r)                           -> the swap target (its ipd16)
  view 1, spectral input = EQ(WPE(h1) * RIR')                                 -> spectral encoder (mid channel)
            WPE(h1): the precomputed multichannel-WPE mid of h1 (v3_wpe.py, simu_unpaired8_wpe/shard_XX.tar)
            RIR': a two-channel RIR drawn uniformly from a DISJOINT pool (paired v2 training scenes [0, 12000), whose
                  rooms are never an unpaired clip's room), so its T60 follows the training-room distribution
            EQ: 3-5 peaking filters (gain U(-6, 6) dB, centre log-U(100, 7000) Hz, Q U(0.7, 2.0)) plus a tilt
                U(-3, 3) dB / octave around 1 kHz, applied as one zero-phase magnitude response to both channels
  view 2, spatial input  = h2 as recorded = x(s, r) at another time           -> spatial encoder
The target therefore shares only the content (s) with the spectral input and only the room (r) with the spatial input.
Training items draw RIR' / EQ / half order per access (torch RNG, seeded per DataLoader worker); validation items
use a per-item seed (fixed triplets). ids (3, 2) int64 = [[clip, clip]] * 3 (V1 / contrastive terms stay off in V3).
"""
import io
import os

import numpy as np
import soundfile
import torch
from scipy.signal import fftconvolve

HALF, FS = 65792, 16000
EQ_N, EQ_GAIN, EQ_FC, EQ_Q, TILT = (3, 5), 6.0, (100.0, 7000.0), (0.7, 2.0), 3.0


def stage_local(tar, wpe_dir, margin_gb=2.0, log=print):
	""" Optional (--v3-stage-local, off by default; 11-24): copy the unpaired tar (+ .index.json, .meta.npz) and, if
		there is still room, the WPE shards to node-local $STAGE_DIR (set per job by scripts/job_tmp.sh; default
		/tmp/v3stage_<job id>; never $TMPDIR, which is on /scratch200) at job start, to take random reads off
		/scratch200. Greedy: tar first (the larger random-read load), then the WPE shards; anything that does not fit
		(free space < size + margin) or fails to copy is read from /scratch200 as before. copy2 keeps mtimes, so
		tar_index's cache (<tar>.index.json, valid if not older than the tar) is reused. The copy is removed at exit.
		-> (tar, wpe_dir) to use. """
	import atexit
	import shutil
	import time
	dst = os.environ.get('STAGE_DIR') or f'/tmp/v3stage_{os.environ.get("SLURM_JOB_ID", os.getpid())}'
	root = os.path.dirname(dst)
	try:
		os.makedirs(dst, exist_ok=True)
	except OSError as e:
		log(f'[v3 stage] {root} unusable ({e}); reading from /scratch200')
		return tar, wpe_dir
	atexit.register(shutil.rmtree, dst, True)  # /tmp is not always job-scoped

	def copy(srcs, out_dir, what):
		need = sum(os.path.getsize(p) for p in srcs)
		free = shutil.disk_usage(dst).free
		if free < need + margin_gb * 2 ** 30:
			log(f'[v3 stage] {what}: {free / 2 ** 30:.1f} GB free < {need / 2 ** 30:.1f} GB + {margin_gb} GB; stays on /scratch200')
			return False
		t0 = time.time()
		try:
			os.makedirs(out_dir, exist_ok=True)
			for p in srcs:  # for the tar: the tar first, then its index (index mtime stays >= tar mtime)
				shutil.copy2(p, os.path.join(out_dir, os.path.basename(p)))
		except OSError as e:
			log(f'[v3 stage] {what}: copy failed ({e}); stays on /scratch200')
			shutil.rmtree(out_dir if out_dir != dst else os.path.join(dst, os.path.basename(tar)), ignore_errors=True)
			return False
		log(f'[v3 stage] {what}: {need / 2 ** 30:.1f} GB -> {out_dir} in {time.time() - t0:.0f} s')
		return True

	extras = [tar + s for s in ('.index.json', '.meta.npz') if os.path.exists(tar + s)]
	out_tar = os.path.join(dst, os.path.basename(tar)) if copy([tar] + extras, dst, 'unpaired tar') else tar
	shards = sorted(os.path.join(wpe_dir, f) for f in os.listdir(wpe_dir) if f.endswith('.tar') or f.endswith('.tar.index.json'))
	shards.sort(key=lambda p: p.endswith('.json'))  # each shard before its index (index mtime >= shard mtime)
	lwpe = os.path.join(dst, 'wpe')
	out_wpe = lwpe if copy(shards, lwpe, 'WPE shards') else wpe_dir
	return out_tar, out_wpe


def peaking_mag(f, fc, gain_db, q):
	""" |H(f)| of the RBJ peaking EQ biquad (bilinear, FS) at frequencies f (Hz). """
	A = 10 ** (gain_db / 40)
	w0 = 2 * np.pi * fc / FS
	alpha = np.sin(w0) / (2 * q)
	b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
	a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
	z = np.exp(-1j * 2 * np.pi * f / FS)
	return np.abs((b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2))


def random_eq(rng, n_fft):
	""" -> (magnitude response on rfft bins (n_fft // 2 + 1,), parameters dict). """
	f = np.fft.rfftfreq(n_fft, 1 / FS)
	n = int(rng.integers(EQ_N[0], EQ_N[1] + 1))
	fc = np.exp(rng.uniform(np.log(EQ_FC[0]), np.log(EQ_FC[1]), n))
	g = rng.uniform(-EQ_GAIN, EQ_GAIN, n)
	q = rng.uniform(*EQ_Q, n)
	tilt = rng.uniform(-TILT, TILT)
	H = 10 ** (tilt * np.log2(np.maximum(f, 50.0) / 1000.0) / 20)
	for i in range(n):
		H = H * peaking_mag(f, fc[i], g[i], q[i])
	return H, {'fc': fc.tolist(), 'gain_db': g.tolist(), 'q': q.tolist(), 'tilt_db_oct': float(tilt)}


_POOL = {}


def load_pool(path, hi):
	""" the RIR pool, loaded once per process (DataLoader workers inherit it by fork, copy-on-write). """
	if (path, hi) not in _POOL:
		z = np.load(path)
		_POOL[(path, hi)] = (z['rir'][:hi], z['rir_len'][:hi].astype(np.int64))
	return _POOL[(path, hi)]


class V3Dataset(torch.utils.data.Dataset):
	def __init__(self, tar, wpe_dir, rir_pool, clip_lo, clip_hi, max_items=None, fixed_seed=None, pool_hi=12000,
				 swap_halves=False, eq=True):
		from ssamba_spatial_pretrain import TarMicSigDataset
		self.clips = TarMicSigDataset(tar, 'pretrain')
		self.targets = list(range(clip_lo, clip_hi))[:max_items]
		assert self.targets[-1] < len(self.clips)
		self.wpe_dir, self.fixed_seed, self.swap_halves, self.eq = wpe_dir, fixed_seed, swap_halves, eq
		self.shard_size = 1000
		self._wpe_idx, self._wpe_fh = {}, {}
		self.rir, self.rir_len = load_pool(rir_pool, pool_hi)  # (n, 2, L) fp16, shared by train / val and the workers
		self.k = 1

	def __len__(self):
		return len(self.targets)

	def wpe(self, clip, h):
		s = clip // self.shard_size
		if s not in self._wpe_idx:
			p = os.path.join(self.wpe_dir, f'shard_{s:02d}.tar')
			from ssamba_spatial_pretrain import tar_index  # cached <shard>.index.json: no header scan per worker
			self._wpe_idx[s], self._wpe_fh[s] = tar_index(p), open(p, 'rb')
		off, size = self._wpe_idx[s][f'wpe/{clip}_{h}.wav']
		fh = self._wpe_fh[s]
		fh.seek(off)
		x, fs = soundfile.read(io.BytesIO(fh.read(size)), dtype='float32')
		assert fs == FS and x.shape == (HALF,), (fs, x.shape)
		return x

	def rng(self, clip):
		if self.fixed_seed is None:
			return np.random.default_rng(int(torch.randint(2 ** 62, ())))
		return np.random.default_rng((self.fixed_seed * 1_000_003 + clip) % 2 ** 63)

	def spectral_view(self, clip, h, rng, info=None):
		""" EQ(WPE(h) * RIR') as (HALF, 2) float32. """
		x = self.wpe(clip, h)
		j = int(rng.integers(len(self.rir)))
		r = self.rir[j, :, :self.rir_len[j]].astype(np.float32)  # (2, L)
		y = np.stack([fftconvolve(x, r[c])[:HALF] for c in range(2)], axis=1)  # causal, trimmed to the clip
		if self.eq:
			H, eqp = random_eq(rng, HALF)
			y = np.fft.irfft(np.fft.rfft(y, axis=0) * H[:, None], n=HALF, axis=0)
		else:
			eqp = None
		y = (y / (np.abs(y).max() + 1e-8) * 0.9).astype(np.float32)  # scale is irrelevant (per-clip STFT norm)
		if info is not None:
			info.update(rir_pool_idx=j, eq=eqp)
		return y

	def item(self, i, info=None):
		c = self.targets[i]
		rng = self.rng(c)
		x = self.clips[c]
		assert x.shape == (2 * HALF, 2), x.shape
		h1 = int(rng.integers(2)) if self.swap_halves else 0
		halves = (x[:HALF], x[HALF:])
		spec = self.spectral_view(c, h1, rng, info)
		return np.stack([halves[h1], spec, halves[1 - h1]]).astype(np.float32), np.full((3, 2), c, dtype=np.int64)

	def __getitem__(self, i):
		return self.item(i)
