""" Check 4 (agreed 2026-11-22, eval only): how much of the swap target each encoder supplies, on check 3's scale.

	The validation-time shuffle diagnostic in swap_pretrain.py (swap_<ch>_shuf<k>) has two problems:
	  - scale: it reports the raw loss, i.e. error relative to the GLOBAL mean (the per-bin prior counts as explained),
	    while check 3 (data_generation/room_share.py) splits the variance around each frequency bin's mean;
	  - donors: it rolls one encoder's tokens by one position in the batch, but the validation loader is unshuffled and
	    targets are ordered by clip id, so clips r*k .. r*k+k-1 share scene r. For k - 1 of every k targets the
	    "spatially shuffled" donor's spatial input is the SAME scene (another utterance). The spectral roll is valid
	    (the k utterances of a scene differ).
	Here, on the fixed validation triplets (val pairing seed 12345, as training):
	  conditions  normal; spec shuffled (donor's spectral input has another utterance than the target); spat shuffled
	              (donor's spatial input is from another scene); both (two different donors, each as above); and the
	              old roll-by-1 versions (spec_roll1, spat_roll1) for comparison with the logged numbers.
	  metric      per channel: sum W (pred - T)^2 / sum W (T - mu_f)^2, mu_f = W-weighted mean of the target per frequency
	              bin over all validation clips and frames; W = the loss weights (1 for log|L| / ILD, the magnitude weight
	              min(sqrt(|L||R|) / clip mean, 5) for cos / sin IPD), T standardised with the model's own spatial input
	              statistics: exactly the swap loss's terms. 1.0 = no better than the per-bin prior. Also reported: the
	              raw loss term (= swap_<ch>), and the same error over the variance around the global mean.
	  read        cost of shuffling encoder k = its ratio minus normal's, in units of bin-centred variance; compare
	              with check 3's scene / utterance shares.

	Usage: python swap_shufdiag.py <out json> <label>=<exp dir>:<joint ckpt file> [...]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.utils_module as at_module  # noqa: E402
import swap_pretrain as sp  # noqa: E402
from ssamba_spatial_pretrain import NFFT, WIN_LEN, WIN_SHIFT_RATIO  # noqa: E402

CONDS = ['normal', 'spec_shuf', 'spat_shuf', 'both_shuf', 'spec_roll1', 'spat_roll1']
BANDS = [(0, 64), (64, 128), (128, 192), (192, 256), (0, 256)]
BAND_NAMES = ['0-2 kHz', '2-4 kHz', '4-6 kHz', '6-8 kHz', 'all']


def donors(ok, start):
	""" ok (B, B) bool: ok[i, j] = clip j may serve as i's donor. -> (B,) donor index, the first ok one at offsets
		start, start + 1, ... back in the batch; and how many targets needed an offset beyond start. """
	B = ok.shape[0]
	out, moved = torch.empty(B, dtype=torch.long), 0
	for i in range(B):
		for o in range(start, start + B):
			j = (i - o) % B
			if j != i and ok[i, j]:
				out[i] = j; moved += int(o != start)
				break
		else:
			raise RuntimeError(f'no valid donor for target {i}')
	return out, moved


@torch.no_grad()
def run(model, loader, stft_mod, dev, k):
	unf = model.streams['spat'].enc.unfold
	assert tuple(np.atleast_1d(unf.kernel_size)) in ((16,), (16, 16)) and tuple(np.atleast_1d(unf.stride)) in ((16,), (16, 16)), \
		f'patches must tile the image: kernel {unf.kernel_size} stride {unf.stride}'
	z = lambda: torch.zeros(4, 256, dtype=torch.float64, device=dev)
	err = {c: z() for c in CONDS}
	sw, swx, swx2, n_el = z(), z(), z(), 0
	moved = {c: 0 for c in CONDS}
	logged = {c: z().sum(1) for c in ('normal',)}
	for sig3, ids in loader:
		sig3, ids = sig3.to(dev), ids.to(dev)
		spec, spat, tgt, w, _, _ = model.views(sig3, stft_mod)
		W = torch.cat([torch.ones_like(w), torch.ones_like(w), w, w], 1).double()  # (B, 4, 256, 256), loss weights
		T = tgt.double()
		toks = model.encode({'spec': spec, 'spat': spat}, None, fusion=False)
		B = T.shape[0]
		ar = torch.arange(B, device=dev)
		ok_spec = (ids[None, :, 1, 0] != ids[:, None, 0, 0]).cpu()  # donor's spectral-view utterance != target's utterance
		ok_spat = (ids[None, :, 2, 1] != ids[:, None, 0, 1]).cpu()  # donor's spatial-view scene != target's scene
		d_spec, m1 = donors(ok_spec, k)
		d_spat, m2 = donors(ok_spat, k)
		ok_both = ok_spat & (torch.arange(B)[None, :] != d_spec[:, None])  # second donor differs from the first
		d_spat2, m3 = donors(ok_both, 2 * k)
		moved['spec_shuf'] += m1; moved['spat_shuf'] += m2; moved['both_shuf'] += m3
		roll = (ar.cpu() - 1) % B
		plan = {'normal': (ar.cpu(), ar.cpu()), 'spec_shuf': (d_spec, ar.cpu()), 'spat_shuf': (ar.cpu(), d_spat),
				'both_shuf': (d_spec, d_spat2), 'spec_roll1': (roll, ar.cpu()), 'spat_roll1': (ar.cpu(), roll)}
		for c, (i_spec, i_spat) in plan.items():
			t2 = {'spec': toks['spec'][i_spec.to(dev)], 'spat': toks['spat'][i_spat.to(dev)]}
			pred = model.decoder(t2).float()  # (B, 256 patches, 4 * 256), channel-major per patch
			img = F.fold(pred.transpose(1, 2), output_size=(256, 256), kernel_size=16, stride=16).double()
			err[c] += (W * (img - T) ** 2).sum(dim=(0, 3))
			if c == 'normal':  # the loss as swap_losses computes it, for the cross-check
				ref = model.swap_losses(t2, tgt, w)
				logged['normal'] += torch.tensor([float(ref[f'swap_{n}']) * B for n in sp.CH_NAMES], device=dev, dtype=torch.float64)
		assert torch.allclose(F.fold(model.patchify(tgt).transpose(1, 2), (256, 256), kernel_size=16, stride=16), tgt)
		sw += W.sum(dim=(0, 3)); swx += (W * T).sum(dim=(0, 3)); swx2 += (W * T ** 2).sum(dim=(0, 3))
		n_el += B * 256  # elements per (channel, bin) = clips x frames
	n_clips = n_el // 256
	var_f = swx2 - swx ** 2 / sw.clamp_min(1e-12)  # (4, 256): sum W (T - mu_f)^2
	var_g = swx2.sum(1) - swx.sum(1) ** 2 / sw.sum(1)  # (4,): sum W (T - mu_global)^2
	res = {'n_clips': n_clips, 'donor_fallbacks': moved, 'bands': BAND_NAMES, 'channels': list(sp.CH_NAMES)}
	raw_norm = n_clips * 256 * 256
	check = (err['normal'].sum(1) / raw_norm).tolist(), (logged['normal'] / n_clips).tolist()
	res['crosscheck_normal_raw_vs_swap_losses'] = check
	assert np.allclose(check[0], check[1], rtol=1e-3), f'raw error does not reproduce swap_losses: {check}'
	for c in CONDS:
		res[c] = {'binc': [[float(err[c][ch, lo:hi].sum() / var_f[ch, lo:hi].sum()) for lo, hi in BANDS] for ch in range(4)],
				  'globalc': [float(err[c][ch].sum() / var_g[ch]) for ch in range(4)],
				  'raw': [float(err[c][ch].sum() / raw_norm) for ch in range(4)]}
	res['binc_var_over_global_var'] = [float(var_f[ch].sum() / var_g[ch]) for ch in range(4)]
	return res


def main():
	out, specs = sys.argv[1], sys.argv[2:]
	dev = 'cuda'
	stft_mod = at_module.STFT(win_len=WIN_LEN, win_shift_ratio=WIN_SHIFT_RATIO, nfft=NFFT).to(dev)
	allres = {}
	for s in specs:
		label, rest = s.split('=', 1)
		exp, ck = rest.split(':', 1)
		a = json.load(open(os.path.join(exp, 'args.json')))
		model = sp.SwapPretrain(a['mask_cols'], a['recon_lambda'], a['classif_lambda'], a['swap_lambda'], a['own_scale'],
								sac_lambda=a.get('sac_lambda', 0.0), ctr_head=a.get('ctr_head', False)).to(dev)  # SAC / ctr heads present in C + SAC / C2-head checkpoints
		model.load_state_dict(torch.load(os.path.join(exp, ck), map_location='cpu', weights_only=False)['model'])
		model.eval()
		n_scenes = int(np.load(a['tar'] + '.meta.npz')['n_scenes'])
		ds = sp.PairedDataset(a['tar'], n_scenes - a['val_scenes'], n_scenes, a.get('num_val'), fixed_seed=12345)
		loader = torch.utils.data.DataLoader(ds, batch_size=a['bs'], shuffle=False, num_workers=5, pin_memory=True)
		torch.manual_seed(1234)
		r = run(model, loader, stft_mod, dev, ds.k)
		allres[label] = r
		print(f'\n## {label} ({exp}/{ck}), {r["n_clips"]} validation targets, donor fallbacks {r["donor_fallbacks"]}')
		print(f'cross-check raw normal vs swap_losses: {np.round(r["crosscheck_normal_raw_vs_swap_losses"], 4).tolist()}')
		print('bin-centred variance / global variance per channel: ' +
			  ', '.join(f'{n} {v:.3f}' for n, v in zip(sp.CH_NAMES, r['binc_var_over_global_var'])))
		print('\nError / bin-centred variance (all bins); in brackets: / global variance, raw loss term')
		print('| condition | ' + ' | '.join(sp.CH_NAMES) + ' |'); print('|---|' + '---|' * 4)
		for c in CONDS:
			print(f'| {c} | ' + ' | '.join('%.3f (%.3f, %.3f)' % (r[c]['binc'][ch][-1], r[c]['globalc'][ch], r[c]['raw'][ch])
										  for ch in range(4)) + ' |')
		print('\nCost of shuffling (ratio - normal, bin-centred), per band ' + ' / '.join(BAND_NAMES))
		print('| channel | spec_shuf | spat_shuf | both_shuf | spat_roll1 (old) |'); print('|---|---|---|---|---|')
		for ch, n in enumerate(sp.CH_NAMES):
			cost = lambda c: ' / '.join('%.3f' % (r[c]['binc'][ch][b] - r['normal']['binc'][ch][b]) for b in range(len(BANDS)))
			print(f'| {n} | {cost("spec_shuf")} | {cost("spat_shuf")} | {cost("both_shuf")} | {cost("spat_roll1")} |')
		sys.stdout.flush()
	os.makedirs(os.path.dirname(out), exist_ok=True)
	json.dump(allres, open(out, 'w'), indent=1)
	print(f'\n[done] -> {out}')


if __name__ == '__main__':
	main()
