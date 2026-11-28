""" Teacher representation analysis (Yotam, 28 Nov), step 4: teacher -> student linear CKA (GPU, low priority).

Clips: the 2,000 held-out paired-v2 clips (scenes [12000, 12500) x k = 4 renderings; the scenes of steps 1-3), first
4.112 s, read from the tar ONCE and staged node-local, with the encoder checkpoints copied node-local as well (eval-style
staging; EVAL_LOCAL). Student embedding per encoder = exactly what the lineareval head / ridge_dump sees:
STFTLearner.data_preprocess -> JointSpatialDownstream(concat).patches -> mean over real patches (768 each).
Teacher per clip: its scene's A' / HC / oracle B cue, its utterance's speech cue (heldout_cues.npz from
teacher_geometry.py: the exact V1b cues).
CKA: linear CKA (Kornblith et al. 2019) and debiased linear CKA (unbiased HSIC, Song et al. 2012; Nguyen et al. 2021),
every student encoder x every teacher. Also on scene means (the 4 clips of a scene averaged) for the scene-level teachers.
Writes <out>/cka.json and <cache>/student_emb.npz.
Usage (GPU): python teacher_student_cka.py --cues <heldout_cues.npz> --out <dir> --cache <dir> name=<spec.pth>,<spat.pth> ...
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TAR = '/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar'
NS, FS = 65792, 16000


def cka(X, Y):
	X, Y = X - X.mean(0), Y - Y.mean(0)
	return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def hsic_u(K, L):
	n = len(K)
	K, L = K.copy(), L.copy()
	np.fill_diagonal(K, 0)
	np.fill_diagonal(L, 0)
	one = np.ones(n)
	return ((K * L).sum() + (one @ K @ one) * (one @ L @ one) / ((n - 1) * (n - 2)) - 2 / (n - 2) * (one @ K @ (L @ one))) / (n * (n - 3))


def cka_debiased(X, Y):
	X, Y = X - X.mean(0), Y - Y.mean(0)
	K, L = X @ X.T, Y @ Y.T
	return float(hsic_u(K, L) / np.sqrt(hsic_u(K, K) * hsic_u(L, L)))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--cues', required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--cache', required=True)
	ap.add_argument('runs', nargs='+')
	a = ap.parse_args()
	t0 = time.time()
	log = lambda s: print(f'[cka {time.time() - t0:5.0f}s] {s}', flush=True)
	local = os.environ.get('TMPDIR_LOCAL') or os.path.join(os.environ.get('TMPDIR', '/tmp'), f'tcka_{os.environ.get("SLURM_JOB_ID", "x")}')
	os.makedirs(local, exist_ok=True)
	m = np.load(TAR + '.meta.npz', allow_pickle=True)
	clips = np.arange(12000 * 4, 12500 * 4)
	assert (m['scene'][clips] == clips // 4).all()
	from ssamba_spatial_pretrain import TarMicSigDataset
	ds = TarMicSigDataset(TAR, 'pretrain')
	assert len(ds) == 50000, len(ds)
	sig_f = f'{local}/heldout_clips.npy'
	sig = np.stack([np.asarray(ds[int(c)][:NS], dtype=np.float32) for c in clips])
	np.save(sig_f, sig)
	log(f'staged {sig.shape} clips -> {sig_f}')
	import joint_spatial
	from learner import STFTLearner
	emb = {}
	for r in a.runs:
		name, paths = r.split('=', 1)
		lp = []
		for p in paths.split(','):
			d = f'{local}/{name}_{os.path.basename(p)}'
			shutil.copy(p, d)
			lp.append(d)
		torch.manual_seed(0)
		net = joint_spatial.JointSpatialDownstream(','.join(lp), 'concat').cuda().eval()
		learner = STFTLearner(net, win_len=512, win_shift_ratio=0.5, nfft=512, fre_used_ratio=1, fs=FS, task='T60', ch_mode='M')
		learner.cuda()
		acc = {'spec': [], 'spat': []}
		with torch.no_grad():
			for b in range(0, len(sig), 16):
				x, = learner.data_preprocess(torch.from_numpy(sig[b:b + 16]), None)
				p = net.patches(x)
				for k in acc:
					acc[k].append(p[k].mean(dim=(1, 2)).float().cpu().numpy())
		for k in acc:
			emb[f'{name}/{k}'] = np.concatenate(acc[k]).astype(np.float64)
		log(f'{name}: spec {emb[name + "/spec"].shape}, spat {emb[name + "/spat"].shape}')
		del net, learner
		torch.cuda.empty_cache()
	os.makedirs(a.cache, exist_ok=True)
	np.savez(f'{a.cache}/student_emb.npz', clips=clips, **{k.replace('/', '__'): v for k, v in emb.items()})
	c = np.load(a.cues)
	sc = clips // 4 - 12000
	utt = m['utt'][clips]
	teach = {'A′': c['A'][sc], 'HC': c['HC'][sc], 'oracle_B': c['B'][sc], 'speech': c['S_all'][utt]}
	res = {'n_clips': len(clips), 'clip_level': {}, 'scene_level': {}}
	for s, E in emb.items():
		res['clip_level'][s] = {t: {'cka': cka(E, T), 'cka_debiased': cka_debiased(E, T)} for t, T in teach.items()}
		Es = E.reshape(500, 4, -1).mean(1)
		res['scene_level'][s] = {t: {'cka': cka(Es, T[::4]), 'cka_debiased': cka_debiased(Es, T[::4])}
								 for t, T in teach.items() if t != 'speech'}
		log(s + ': ' + ', '.join(f'{t} {v["cka"]:.3f}/{v["cka_debiased"]:.3f}' for t, v in res['clip_level'][s].items()))
	# teacher x teacher reference (clip level)
	res['teacher_teacher'] = {f'{t1}~{t2}': {'cka': cka(T1, T2), 'cka_debiased': cka_debiased(T1, T2)}
							  for i, (t1, T1) in enumerate(teach.items()) for t2, T2 in list(teach.items())[i + 1:]}
	os.makedirs(a.out, exist_ok=True)
	json.dump(res, open(f'{a.out}/cka.json', 'w'), indent=1)
	shutil.rmtree(local, ignore_errors=True)
	log(f'-> {a.out}/cka.json')


if __name__ == '__main__':
	main()
