"""
    Paired (scene x utterance) simulated pretraining set for Pilot C (swap reconstruction).

    Sparse bipartite design: n_scenes scenes, a pool of n_scenes utterances, k utterances per scene, and every utterance
    rendered in exactly k scenes. Clip id = scene * k + slot, so for any clip the set has
        - the same utterance in k-1 other scenes   (spectral partner: same source, different scene)
        - k-1 other utterances in the same scene   (spatial partner: same scene, different source)
    A scene is one room + mic array + static source position (same priors as gen_simu.py --stage pretrain, 2 omni mics).
    An utterance is one WSJ0 speaker, one utterance, one 4.112-s crop, fixed by its own seed, so it is the identical dry
    signal in every scene it appears in. Noise type and SNR prior as gen_simu.py, drawn independently per clip.

    Output: one uncompressed tar (scratch has few free inodes): <split>/<clip id>.wav (PCM16, as gen_simu.py), plus
    <tar>.meta.npz with per-clip scene / utterance ids, the scene config, utterance source, md5 of the dry source, and
    TDOA / DRR / C50 / T60 annotations. Readable by ssamba_spatial_pretrain.TarMicSigDataset(tar, split).

    Example:
        python gen_simu_paired.py --src_dir /scratch200/.../SrcSig/wsj0/tr --out_tar /scratch200/.../simu_paired.tar \
            --n_scenes 12500 --k 4 --workers 6
"""
import argparse
import hashlib
import io
import multiprocessing as mp
import os
import sys
import tarfile
import time

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
	os.environ[v] = '2'

import numpy as np
import soundfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_simu_rir_sig import SpatialAcoustics, RoomImpulseResponse, MicrophoneSignalOrRIR  # noqa: E402
from utils_src import explore_corpus  # noqa: E402
from utils_noise import NoiseSignal  # noqa: E402
from utils_array import mic_array_cfg_2ch  # noqa: E402

# gen_simu.py GenerateRandomMicSig defaults (the pretraining set's priors)
PRIORS = dict(room_sz_range=[(3, 15), (3, 10), (2.5, 6)], T60_range=(0.2, 1.3), abs_weights_range=[(0.5, 1)] * 6,
			  array_pos_ratio_range=[(0.2, 0.8), (0.2, 0.8), (0.1, 0.5)], num_source_range=(1, 1),
			  source_state='static', min_src_array_dist=0.3, min_src_boundary_dist=0.3, traj_pt_mode='time')
FS, C, ISM_DB, T = 16000, 343.0, 12, 4.112
NSAMPLE = int(T * FS)
MAX_SCENE_TRIES = 20


def assign(n_scenes, k, rng):
	""" (n_scenes, k) utterance ids: each row distinct, each utterance exactly k times (k random permutations,
		in-row duplicates repaired by swapping within the same permutation, which keeps the column a permutation). """
	A = np.stack([rng.permutation(n_scenes) for _ in range(k)], 1)
	for _ in range(100):
		bad = [(r, j) for r in range(n_scenes) for j in range(1, k) if A[r, j] in A[r, :j]]
		if not bad:
			return A
		for r, j in bad:
			s = rng.randint(n_scenes)
			A[r, j], A[s, j] = A[s, j], A[r, j]
	raise RuntimeError('could not build a duplicate-free assignment')


class Renderer:
	""" Per-process state (one gpuRIR context per process). """
	def __init__(self, src_dir, seed, snr_range):
		corpus, _ = explore_corpus(src_dir, 'wav')
		self.spks = sorted(corpus)
		self.utts = [sorted(corpus[s].values()) for s in self.spks]
		self.seed, self.snr_range = seed, snr_range
		self.sa = SpatialAcoustics()
		self.roomir = RoomImpulseResponse(fs=FS, c=C, ism_db=ISM_DB)
		self.msr = MicrophoneSignalOrRIR()
		self.noise = NoiseSignal(T=T, fs=FS, nmic=2, noise_type='diffuse_white', noise_path='', c=C)

	def utterance(self, u):
		""" Dry source of utterance u: (nsample,) float64 + description. Own RandomState, so identical in every scene.
			Same sampling as WSJ0Dataset: uniform speaker, uniform utterance, pad with the speaker's next utterances,
			uniform crop, mean removed. """
		rs = np.random.RandomState(self.seed + 1_000_000 + u)
		si = rs.randint(len(self.spks))
		paths = self.utts[si]
		ui = rs.randint(len(paths))
		sig, i = np.array([]), ui
		while sig.shape[0] < NSAMPLE:
			x, fs = soundfile.read(paths[i])
			assert fs == FS, (paths[i], fs)
			sig = np.concatenate((sig, x))
			i = (i + 1) % len(paths)
		st = rs.randint(0, sig.shape[0] - NSAMPLE + 1)
		s = sig[st:st + NSAMPLE]
		s = s - s.mean()
		return s, dict(spk=self.spks[si], utt_path=paths[ui], crop=int(st))

	def scene(self, r, n_scenes):
		""" Scene config + RIRs. RIRs are deterministic in the config, so an invalid one draws a new config. """
		for t in range(MAX_SCENE_TRIES):
			cfg = self.sa.generate_random_spatial_acoustics(c=C, ism_db=ISM_DB, mic_array_cfg=mic_array_cfg_2ch,
															 nb_points=1, room_cfg=None, seed=self.seed,
															 idx=r + t * n_scenes, **PRIORS)
			kw = dict(room_sz=cfg['room_sz'], beta=cfg['beta'], T60=cfg['T60_sabine'], mic_pos=cfg['mic_pos'],
					  mic_orV=cfg['mic_orV'], mic_pattern=cfg['mic_pattern'], src_traj_pts=cfg['src_traj_pts'])
			rir = self.roomir.generate_rir(**kw)
			rir_dp = self.roomir.generate_rir(**kw, dp_gen=True)
			if self.roomir.check_rir(rir) and self.roomir.check_rir(rir_dp):
				ok, t60_edc = self.roomir.check_rir_envelope(rir, cfg['T60_specify'], FS)
				if ok:
					cfg['T60_edc'], cfg['scene_tries'] = t60_edc, t
					return cfg, rir, rir_dp
		raise RuntimeError(f'scene {r}: no valid RIR in {MAX_SCENE_TRIES} draws')

	def render(self, job):
		""" One scene, all its k clips. -> [(clip id, wav bytes, info dict)] """
		r, utts, k, n_scenes = job
		cfg, rir, rir_dp = self.scene(r, n_scenes)
		annos = self.msr.generate_annotation(traj_pts=cfg['src_traj_pts'], array_pos=cfg['array_pos'],
											 mic_pos=cfg['mic_pos'], rir_srcs=rir, rir_srcs_dp=rir_dp, TDOA=True,
											 DRR=True, C50=True, src_single_static=True, fs=FS, c=C)
		out = []
		for j, u in enumerate(utts):
			cid = r * k + j
			src, desc = self.utterance(int(u))
			clean, _ = self.roomir.rir_conv_src(rir, src[:, None], gpu_conv=False)
			dp, _ = self.roomir.rir_conv_src(rir_dp, src[:, None], gpu_conv=False)
			np.random.seed(self.seed + 2_000_000 + cid)  # noise + SNR: independent per clip
			noi = self.noise.generate_random_noise(mic_pos=cfg['mic_pos'])
			snr = np.random.uniform(*self.snr_range)
			sig = self.noise.add_noise(clean, noi, snr, mic_sig_dp=dp)
			sig = sig / max(np.abs(sig).max(), np.abs(dp).max()) * 0.9  # as gen_simu.py
			buf = io.BytesIO()
			soundfile.write(buf, sig, FS, format='WAV')  # PCM16, as gen_simu.py's soundfile.write
			info = dict(clip=cid, scene=r, slot=j, utt=int(u), snr=snr, src_md5=hashlib.md5(src.tobytes()).hexdigest(),
						**desc, **{key: cfg[key] for key in ('room_sz', 'T60_specify', 'T60_edc', 'mic_pos', 'array_pos',
															 'array_scale', 'src_traj_pts', 'scene_tries')},
						**{key: np.asarray(v) for key, v in annos.items()})
			out.append((cid, buf.getvalue(), info))
		return out


_R = None


def _init(src_dir, seed, snr_range):  # all workers share the job's GPU (CUDA_VISIBLE_DEVICES from Slurm)
	global _R
	_R = Renderer(src_dir, seed, snr_range)


def _work(job):
	return _R.render(job)


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--src_dir', required=True, help='WSJ0 split dir (spk/*.wav), e.g. .../wsj0/tr')
	p.add_argument('--out_tar', required=True)
	p.add_argument('--split', default='pretrain', help='member prefix in the tar (TarMicSigDataset split)')
	p.add_argument('--n_scenes', type=int, default=12500)
	p.add_argument('--k', type=int, default=4)
	p.add_argument('--seed', type=int, default=7_000_000, help='distinct from gen_simu.py stage seeds (1, 2e6-6e6)')
	p.add_argument('--snr_range', type=float, nargs=2, default=(15, 30))
	p.add_argument('--workers', type=int, default=4, help='processes, all on the one GPU')
	p.add_argument('--first_scene', type=int, default=0, help='render scenes [first, last) only (tests)')
	p.add_argument('--last_scene', type=int, default=None)
	a = p.parse_args()
	last = a.n_scenes if a.last_scene is None else a.last_scene
	A = assign(a.n_scenes, a.k, np.random.RandomState(a.seed))
	jobs = [(r, A[r], a.k, a.n_scenes) for r in range(a.first_scene, last)]
	print(f'{len(jobs)} scenes x {a.k} = {len(jobs) * a.k} clips -> {a.out_tar} ({a.workers} workers)', flush=True)

	t0, infos = time.time(), []
	with tarfile.open(a.out_tar, 'w', format=tarfile.USTAR_FORMAT) as tar, \
			mp.get_context('spawn').Pool(a.workers, _init, (a.src_dir, a.seed, a.snr_range)) as pool:
		for n, clips in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
			for cid, wav, info in clips:
				ti = tarfile.TarInfo(f'{a.split}/{cid}.wav')
				ti.size, ti.mtime = len(wav), int(time.time())
				tar.addfile(ti, io.BytesIO(wav))
				infos.append(info)
			if n % 100 == 0 or n == len(jobs):
				el = time.time() - t0
				print(f'{n}/{len(jobs)} scenes, {el / 60:.1f} min, {el / n:.2f} s/scene, '
					  f'eta {(len(jobs) - n) * el / n / 3600:.2f} h', flush=True)
	infos.sort(key=lambda d: d['clip'])
	keys = infos[0].keys()
	np.savez(a.out_tar + '.meta.npz', assignment=A, k=a.k, n_scenes=a.n_scenes, seed=a.seed,
			 **{key: np.array([d[key] for d in infos]) for key in keys})
	open(a.out_tar + '.nfiles', 'w').write(f'{len(infos)}\n')
	print(f'done: {len(infos)} clips in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
	main()
