""" V1b-HC (Yotam 11-25): the matched control's spatial cues = the hand-crafted C + SAC spatial cues, one vector per paired
v2 scene, loaded by the V1b trainer through --tsac-cue-file-spat (used as they are; Euclidean distance in the kernel, as
C + SAC's non-factorised CWCL).

Per clip, exactly as C + SAC / sac_cue_check.py: normalised STFT of the two-channel 4.112-s reverberant clip (/ mean |L|),
bins FRE_RANGE, at_module.spatial_features_from_stft -> the 5 SAC_FEATURES (coherence low / mid / high, GCC-PHAT max,
TDOA), z-norm cue = clamp((v - mean) / (3 std), -1, 1) with the stats of sac_cue_check.json (sac_feature_stats on paired
v2 training targets, seed 1; the C + SAC procedure).
Per scene: the mean of its k = 4 renderings' cues (a scene-level cue, as V1b's RIR-teacher cue; the renderings differ
only in the utterance). All 12,500 scenes are written (training uses scenes [0, 12000)).
Outputs: <out>.npy (n_scenes, 5) float32 = the cue file; <out>_clips.npy (n_scenes * k, 5); <out>.json (stats used,
per-feature spread of clip vs scene cues, share of the clip variance that is between scenes).
Usage (CPU or GPU): python v1b_hc_cues.py --stats-json <sac_cue_check.json> --out <prefix>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.utils_module as at_module  # noqa: E402
from ssamba_spatial_pretrain import NFFT, WIN_LEN, WIN_SHIFT_RATIO, FRE_RANGE, FS, SAC_FEATURES, TarMicSigDataset  # noqa: E402

TAR = '/scratch200/yotams2/SAC-SAR-SSL/data/MicSig/simu_paired_pretrain_v2.tar'
NS = 65792  # 4.112 s


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--stats-json', required=True)
	ap.add_argument('--out', required=True)
	ap.add_argument('--tar', default=TAR)
	ap.add_argument('--bs', type=int, default=64)
	ap.add_argument('--workers', type=int, default=6)
	a = ap.parse_args()
	dev = 'cuda' if torch.cuda.is_available() else 'cpu'
	torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4')) - a.workers // 2))
	stats = json.load(open(a.stats_json))['stats']
	m = np.load(a.tar + '.meta.npz', allow_pickle=True)
	S, k = int(m['n_scenes']), int(m['assignment'].shape[1])
	ds = TarMicSigDataset(a.tar, 'pretrain')
	assert len(ds) == S * k, (len(ds), S, k)
	stft_mod = at_module.STFT(win_len=WIN_LEN, win_shift_ratio=WIN_SHIFT_RATIO, nfft=NFFT).to(dev)
	loader = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers,
										 collate_fn=lambda b: torch.stack([torch.as_tensor(x[:NS]) for x in b]))
	clips = np.zeros((S * k, len(SAC_FEATURES)), np.float32)
	n = 0
	with torch.no_grad():
		for sig in loader:
			stft = stft_mod(signal=sig.to(dev)).permute(0, 3, 1, 2)  # as jp.shared_stft: / mean |L|
			stft = stft / (stft[:, 0].abs().reshape(stft.shape[0], -1).mean(dim=1)[:, None, None, None] + 1e-6)
			f = at_module.spatial_features_from_stft(stft[:, :, FRE_RANGE], sample_rate=FS)
			v = torch.stack([((f[key] - stats[key]['mean']) / (3 * stats[key]['std'] + 1e-6)).clamp(-1, 1)
							 for key in SAC_FEATURES], 1)
			clips[n:n + len(v)] = v.float().cpu().numpy()
			n += len(v)
			if n % (100 * a.bs) == 0:
				print(f'[cues] {n} / {S * k} clips', flush=True)
	assert n == S * k and np.isfinite(clips).all(), (n, np.isfinite(clips).all())
	scene = clips.reshape(S, k, -1).mean(1)
	within = clips.reshape(S, k, -1).var(1).mean(0)  # mean within-scene variance per feature
	total = clips.var(0)
	info = {'n_scenes': S, 'k': k, 'features': list(SAC_FEATURES), 'stats': {f: stats[f] for f in SAC_FEATURES},
			'clip_std': clips.std(0).tolist(), 'scene_std': scene.std(0).tolist(),
			'between_scene_share': (1 - within / np.maximum(total, 1e-12)).tolist(),
			'clamped_frac': (np.abs(clips) >= 1).mean(0).tolist(),
			'scene_mean_pairwise_dist': float(np.linalg.norm(scene[:2000, None] - scene[None, :2000], axis=-1).mean())}
	assert (scene.std(0) > 1e-4).all(), f'a cue feature is constant: {info["scene_std"]}'
	os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
	np.save(a.out + '.npy', scene.astype(np.float32))
	np.save(a.out + '_clips.npy', clips)
	json.dump(info, open(a.out + '.json', 'w'), indent=1)
	for i, f in enumerate(SAC_FEATURES):
		print(f'[cues] {f:16s} clip std {info["clip_std"][i]:.3f}  scene std {info["scene_std"][i]:.3f}  '
			  f'between-scene share {info["between_scene_share"][i]:.2f}  clamped {info["clamped_frac"][i]:.3f}', flush=True)
	print(f'[done] {a.out}.npy {scene.shape}', flush=True)


if __name__ == '__main__':
	main()
