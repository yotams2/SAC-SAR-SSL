""" C2 crash check (2026-11-22). C2 (pilotC2_ctr) died at its epoch-5 save: save_atomic could not open
	spec_encoder_best.pth.tmp. save_atomic writes <f>.tmp and os.replace()s it, so a failed open replaces nothing; this
	verifies that on disk, then freezes what the resume would overwrite.

	1. Every checkpoint in the run dir loads, all tensors finite, key sets equal to the epoch-2 snapshot of the same encoder.
	2. spat_encoder_best == spat_encoder_epoch5 tensor for tensor (both written at step 1875 just before the crash).
	3. Snapshots (the resumed run rewrites these files): {spec,spat}_encoder_epoch5.pth -> *_ep5a.pth (byte copies,
	   re-loaded and compared) and the full model of latest.pth -> joint_step<step>.pth (encoders + decoder, for the
	   shuffle diagnostic; same format as joint_best.pth).
	Exit 1 on any failure (the resume job depends on this with afterok).

	Usage: python c2_crash_verify.py <exp dir>
"""
import os
import shutil
import sys

import torch


def tensors(obj, prefix=''):
	if torch.is_tensor(obj):
		yield prefix, obj
	elif isinstance(obj, dict):
		for k, v in obj.items():
			yield from tensors(v, f'{prefix}/{k}')


def check(path):
	ck = torch.load(path, map_location='cpu', weights_only=False)
	ts = dict(tensors(ck))
	bad = [k for k, t in ts.items() if t.is_floating_point() and not torch.isfinite(t).all()]
	assert not bad, f'{path}: non-finite tensors {bad[:5]}'
	print(f'[ok] {os.path.basename(path)}: {len(ts)} tensors, all finite', flush=True)
	return ck, ts


def same(ta, tb, what):
	assert ta.keys() == tb.keys(), f'{what}: key sets differ'
	diff = [k for k in ta if not torch.equal(ta[k], tb[k])]
	assert not diff, f'{what}: {len(diff)} tensors differ, e.g. {diff[:3]}'
	print(f'[ok] {what}: identical ({len(ta)} tensors)', flush=True)


def main():
	exp = sys.argv[1]
	p = lambda f: os.path.join(exp, f)
	assert not [f for f in os.listdir(exp) if f.endswith('.tmp')], 'leftover .tmp files'
	loaded = {}
	for enc in ('spec', 'spat'):
		for tag in ('epoch2', 'epoch5', 'best'):
			loaded[enc, tag] = check(p(f'{enc}_encoder_{tag}.pth'))[1]
		k2 = {k for k in loaded[enc, 'epoch2'] if k.startswith('/state_dict')}
		for tag in ('epoch5', 'best'):
			assert {k for k in loaded[enc, tag] if k.startswith('/state_dict')} == k2, f'{enc} {tag}: keys differ from epoch2'
		print(f'[ok] {enc}: epoch5 / best state_dict keys match epoch2', flush=True)
	sd = lambda t: {k: v for k, v in t.items() if k.startswith('/state_dict')}
	same(sd(loaded['spat', 'best']), sd(loaded['spat', 'epoch5']), 'spat_encoder_best vs spat_encoder_epoch5')
	check(p('joint_best.pth'))
	latest, _ = check(p('latest.pth'))
	print(f'[info] latest.pth step {latest["step"]}, best_val {latest["best_val"]:.4f}', flush=True)

	for enc in ('spec', 'spat'):
		src, dst = p(f'{enc}_encoder_epoch5.pth'), p(f'{enc}_encoder_ep5a.pth')
		shutil.copyfile(src, dst + '.tmp'); os.replace(dst + '.tmp', dst)
		same(sd(check(dst)[1]), sd(loaded[enc, 'epoch5']), f'{os.path.basename(dst)} vs epoch5')
	dst = p(f'joint_step{latest["step"]}.pth')
	torch.save({'model': latest['model'], 'step': latest['step'], 'note': 'model of latest.pth before the 11-22 resume'},
			   dst + '.tmp')
	os.replace(dst + '.tmp', dst)
	same(dict(tensors(check(dst)[0]['model'])), dict(tensors(latest['model'])), f'{os.path.basename(dst)} vs latest.pth model')
	print('[done] all checks passed', flush=True)


if __name__ == '__main__':
	main()
