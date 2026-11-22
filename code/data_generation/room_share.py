"""
    Check 3 (agreed 2026-11-21): how much of the Pilot C swap target's variance only the scene can explain, per channel
    and frequency band. Data only, no training. Also writes the dry-source tar that C3 (transfer-magnitude target) needs.

    1. Dry sources: rebuilt for every utterance of the paired set exactly as gen_simu_paired.Renderer.utterance (own
       RandomState per utterance), each checked against the md5 stored in <tar>.meta.npz. Written as PCM16 to
       <dry tar> (member dry/<utt>.wav, peak-normalised to 0.9; scale is irrelevant: every clip is normalised by its own
       mean |STFT| downstream).
    2. Target channels per clip, exactly as the Pilot C loss sees them: the shared SAR-SSL STFT (/ mean |L|), ipd16
       (log|L|, ILD, cos IPD, sin IPD) standardised with Pilot C's spatial input statistics; IPD weighted per bin by
       min(sqrt(|L||R|) / clip mean, 5) as in swap_losses; plus the transfer magnitude T = log|X_L| - log|S_dry| (dry
       through the same STFT and normalisation; bins with |S_dry| < floor * max masked out).
    3. Variance around each frequency bin's mean (what a decoder cannot get from a constant), weighted as in the loss:
         total            over all clips, frames
         within-utterance across the k scenes of one utterance, same (bin, frame): only the scene (and the per-clip
                          noise) can explain it -> "scene share" = within-utterance / total
         within-scene     across the k utterances of one scene: only the content can explain it -> "content share"
       Utterances of the same source are time-aligned across scenes (same crop, delays << one 16-ms hop).
    4. (added 2026-11-22) Tbar: the time-averaged transfer magnitude per frequency bin, one 256-vector per clip
       (mean of T over the frames where the dry bin is above the floor; a bin with no such frame is left out), with the
       same decomposition: a candidate clip-level scene target. Reported as an extra row; the other rows are unchanged.
       The corrected decomposition (x k/(k-1); pure noise -> interaction + noise) is printed for every row.
    Output: a markdown table (channel x frequency band) on stdout and <out>.json.

    Usage: python room_share.py --tar .../simu_paired_pretrain.tar --src_dir .../wsj0/tr --stats .../pilotC_swap/joint_final.pth
                                --dry_tar .../simu_paired_dry.tar --out .../room_share [--n_utt 2000 --n_scene 2000]
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time

import numpy as np
import soundfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from utils_src import explore_corpus  # noqa: E402
import common.utils_module as at_module  # noqa: E402  (torch only: no mamba kernels needed, so this runs on a CPU node)

# as ssamba_spatial_pretrain / joint_pretrain / ssamba_spatial.stft_frontend (re-implemented to avoid importing the models)
FS, NSAMPLE = 16000, int(4.112 * 16000)
WIN_LEN, NFFT, WIN_SHIFT_RATIO, NT = 512, 512, 0.5, 256
FRE_RANGE = range(1, NFFT // 2 + 1)


def shared_stft(stft_mod, sig, eps=1e-6):
    stft = stft_mod(signal=sig).permute(0, 3, 1, 2)
    return stft / (stft[:, 0].abs().reshape(stft.shape[0], -1).mean(dim=1)[:, None, None, None] + eps)


def fit_frames(X, n=NT):
    return torch.nn.functional.pad(X, (0, n - X.shape[-1])) if X.shape[-1] < n else X[..., :n]


def ipd16_raw(S, eps=1e-3):
    L, R = S[:, 0, FRE_RANGE], S[:, 1, FRE_RANGE]
    logL, logR = torch.log(L.abs() + eps), torch.log(R.abs() + eps)
    ipd = torch.angle(L * R.conj())
    return fit_frames(torch.stack([logL, logR - logL, torch.cos(ipd), torch.sin(ipd)], dim=1))


class TarClips:
    """ pretrain/<id>.wav members of the tar through the offset index cached by ssamba_spatial_pretrain.tar_index. """
    def __init__(self, tar):
        idx = json.load(open(tar + '.index.json'))
        names = sorted((n for n in idx if n.startswith('pretrain/')), key=lambda n: int(os.path.basename(n)[:-4]))
        self.tar, self.items, self.fh = tar, [idx[n] for n in names], None

    def __getitem__(self, i):
        if self.fh is None:
            self.fh = open(self.tar, 'rb')
        off, size = self.items[i]
        self.fh.seek(off)
        return soundfile.read(io.BytesIO(self.fh.read(size)), dtype='float32')[0]
CH = ('logL', 'ILD', 'cosIPD', 'sinIPD', 'T')
BANDS = [(0, 64), (64, 128), (128, 192), (192, 256)]  # of the 256 bins 1..256: 0-2, 2-4, 4-6, 6-8 kHz
BAND_NAMES = ['0-2 kHz', '2-4 kHz', '4-6 kHz', '6-8 kHz', 'all']


class DrySources:
    """ gen_simu_paired.Renderer.utterance without the RIR machinery (same corpus order, same RNG calls). """
    def __init__(self, src_dir, seed):
        corpus, _ = explore_corpus(src_dir, 'wav')
        self.spks = sorted(corpus)
        self.utts = [sorted(corpus[s].values()) for s in self.spks]
        self.seed = seed

    def __call__(self, u):
        rs = np.random.RandomState(self.seed + 1_000_000 + u)
        si = rs.randint(len(self.spks))
        paths = self.utts[si]
        ui = rs.randint(len(paths))
        sig, i = np.array([]), ui
        while sig.shape[0] < NSAMPLE:
            x, fs = soundfile.read(paths[i])
            assert fs == FS
            sig = np.concatenate((sig, x))
            i = (i + 1) % len(paths)
        st = rs.randint(0, sig.shape[0] - NSAMPLE + 1)
        s = sig[st:st + NSAMPLE]
        return s - s.mean()


class Acc:
    """ weighted sums per (channel, bin) for total variance around the per-bin mean, and within-group sums. """
    def __init__(self, nch=len(CH)):
        z = lambda: np.zeros((nch, 256))
        self.w, self.wx, self.wx2, self.within, self.wgrp = z(), z(), z(), z(), z()

    def add_group(self, X, W):
        """ X, W: (k, C, 256, T) values and weights of one group (one utterance's k scenes, or one scene's k utterances). """
        sw = W.sum(0)  # (C, 256, T)
        m = np.where(sw > 0, (W * X).sum(0) / np.maximum(sw, 1e-12), 0.0)
        self.within += (W * (X - m) ** 2).sum(axis=(0, 3))
        self.wgrp += W.sum(axis=(0, 3))
        self.w += W.sum(axis=(0, 3)); self.wx += (W * X).sum(axis=(0, 3)); self.wx2 += (W * X ** 2).sum(axis=(0, 3))

    def per_band(self, within=True):
        tot_f = self.wx2 - self.wx ** 2 / np.maximum(self.w, 1e-12)  # sum of w (x - mean_f)^2
        out_t, out_w = [], []
        for lo, hi in BANDS + [(0, 256)]:
            out_t.append(tot_f[:, lo:hi].sum(1) / self.w[:, lo:hi].sum(1))
            out_w.append(self.within[:, lo:hi].sum(1) / self.wgrp[:, lo:hi].sum(1))
        return np.stack(out_t, 1), np.stack(out_w, 1)  # (C, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', required=True)
    ap.add_argument('--src_dir', required=True)
    ap.add_argument('--stats', required=True, help='Pilot C joint_final.pth (spatial input statistics)')
    ap.add_argument('--dry_tar', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_utt', type=int, default=2000)
    ap.add_argument('--n_scene', type=int, default=2000)
    ap.add_argument('--floor', type=float, default=1e-3)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    m = np.load(a.tar + '.meta.npz', allow_pickle=True)
    k, n_scenes, A = int(m['k']), int(m['n_scenes']), m['assignment']
    dry = DrySources(a.src_dir, int(m['seed']))
    utt_md5 = {int(u): md5 for u, md5 in zip(m['utt'], m['src_md5'])}

    # 1. dry tar (all utterances), md5-checked
    t0 = time.time()
    if not os.path.isfile(a.dry_tar + '.done'):
        with tarfile.open(a.dry_tar + '.tmp', 'w', format=tarfile.USTAR_FORMAT) as tf:
            for u in range(n_scenes):
                s = dry(u)
                assert hashlib.md5(s.tobytes()).hexdigest() == utt_md5[u], f'utterance {u}: md5 differs from the render'
                buf = io.BytesIO()
                soundfile.write(buf, s / max(np.abs(s).max(), 1e-9) * 0.9, FS, format='WAV')
                ti = tarfile.TarInfo(f'dry/{u}.wav'); ti.size = buf.tell(); ti.mtime = int(time.time())
                buf.seek(0); tf.addfile(ti, buf)
                if u % 2000 == 0:
                    print(f'[dry] {u}/{n_scenes} ({time.time() - t0:.0f} s)', flush=True)
        os.replace(a.dry_tar + '.tmp', a.dry_tar)
        open(a.dry_tar + '.done', 'w').write(f'{n_scenes} utterances, md5 checked against {a.tar}.meta.npz\n')
    print(f'[dry] {n_scenes} utterances, all md5 match the render -> {a.dry_tar} ({time.time() - t0:.0f} s)', flush=True)

    # 2. feature extraction as the C loss
    st = torch.load(a.stats, map_location='cpu', weights_only=False)['model']
    in_mean, in_std = st['streams.spat.in_mean'].float(), st['streams.spat.in_std'].float()
    print(f'[stats] spatial input mean {in_mean.flatten().tolist()} std {in_std.flatten().tolist()}', flush=True)
    stft_mod = at_module.STFT(win_len=WIN_LEN, win_shift_ratio=WIN_SHIFT_RATIO, nfft=NFFT)
    clips = TarClips(a.tar)

    @torch.no_grad()
    def feats(cids, utts):
        sig = torch.as_tensor(np.stack([clips[c] for c in cids])).float()
        S = shared_stft(stft_mod, sig)  # (n, 2, 257, nt)
        tgt = (ipd16_raw(S) - in_mean) / in_std  # (n, 4, 256, 256)
        mag = fit_frames((S[:, :, FRE_RANGE].abs().prod(dim=1)).sqrt()[:, None])
        w_ipd = (mag / mag.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)).clamp(max=5.0)
        d = torch.as_tensor(np.stack([dry(int(u)) for u in utts])).float()
        D = shared_stft(stft_mod, torch.stack([d, d], -1))[:, 0, FRE_RANGE].abs()
        XL = S[:, 0, FRE_RANGE].abs()
        T = fit_frames(torch.log(XL.clamp_min(1e-6)) - torch.log(D.clamp_min(1e-6)))
        wT = fit_frames((D >= a.floor * D.amax(dim=(1, 2), keepdim=True)).float())
        X = torch.cat([tgt, T[:, None]], 1).double().numpy()
        ones = torch.ones_like(w_ipd)
        W = torch.cat([ones, ones, w_ipd, w_ipd, wT[:, None]], 1).double().numpy()
        nT = wT.sum(-1)  # (n, 256) valid frames per bin
        Tbar = ((wT * T).sum(-1) / nT.clamp_min(1))[:, None, :, None].double().numpy()  # (n, 1, 256, 1)
        Wbar = (nT > 0).float()[:, None, :, None].double().numpy()
        return X, W, Tbar, Wbar

    rng = np.random.RandomState(0)
    acc_u, acc_s = Acc(), Acc()
    bar_u, bar_s = Acc(1), Acc(1)
    for i, u in enumerate(rng.choice(n_scenes, a.n_utt, replace=False)):  # utterance groups: same source, k scenes
        cids = [c for c in np.flatnonzero(m['utt'] == u)]
        X, W, Xb, Wb = feats(cids, [u] * len(cids))
        acc_u.add_group(X, W); bar_u.add_group(Xb, Wb)
        if i % 250 == 0:
            print(f'[utt groups] {i}/{a.n_utt} ({time.time() - t0:.0f} s)', flush=True)
    for i, r in enumerate(rng.choice(n_scenes, a.n_scene, replace=False)):  # scene groups: same scene, k sources
        cids = list(range(r * k, (r + 1) * k))
        X, W, Xb, Wb = feats(cids, A[r])
        acc_s.add_group(X, W); bar_s.add_group(Xb, Wb)
        if i % 250 == 0:
            print(f'[scene groups] {i}/{a.n_scene} ({time.time() - t0:.0f} s)', flush=True)

    tot_u, win_u = acc_u.per_band()
    tot_s, win_s = acc_s.per_band()
    bt_u, bw_u = bar_u.per_band()
    bt_s, bw_s = bar_s.per_band()
    tot_u, win_u, tot_s, win_s = (np.concatenate([x, y]) for x, y in ((tot_u, bt_u), (win_u, bw_u), (tot_s, bt_s), (win_s, bw_s)))
    CHS = CH + ('Tbar',)
    tot = (tot_u + tot_s) / 2  # both samples are random clips of the set
    res = {'channels': CHS, 'bands': BAND_NAMES, 'total_var': tot.tolist(), 'within_utt_var': win_u.tolist(),
           'within_scene_var': win_s.tolist(), 'scene_share': (win_u / tot_u).tolist(),
           'content_share': (win_s / tot_s).tolist(), 'n_utt': a.n_utt, 'n_scene': a.n_scene, 'k': k}
    f = k / (k - 1)
    scene_dep, content_dep = np.minimum(win_u / tot_u * f, 1), np.minimum(win_s / tot_s * f, 1)
    res['utterance_main'], res['scene_main'] = (1 - scene_dep).tolist(), (1 - content_dep).tolist()
    res['interaction_noise'] = (scene_dep + content_dep - 1).tolist()
    json.dump(res, open(a.out + '.json', 'w'), indent=1)
    print('\nScene share = within-utterance variance / total (only the scene + per-clip noise can explain it); '
          'content share = within-scene / total. Variance around each bin\'s mean, weighted as the C loss.\n')
    print('| channel | ' + ' | '.join(BAND_NAMES) + ' |')
    print('|---|' + '---|' * len(BAND_NAMES))
    for c, name in enumerate(CHS):
        print(f'| {name} | ' + ' | '.join('%.0f%% scene / %.0f%% content' % (100 * win_u[c, b] / tot_u[c, b],
                                                                           100 * win_s[c, b] / tot_s[c, b])
                                          for b in range(len(BAND_NAMES))) + ' |')
    print('\nCorrected decomposition (x k/(k-1)): utterance main / scene main / scene x content + noise\n')
    print('| channel | ' + ' | '.join(BAND_NAMES) + ' |')
    print('|---|' + '---|' * len(BAND_NAMES))
    for c, name in enumerate(CHS):
        print(f'| {name} | ' + ' | '.join('%.0f / %.0f / %.0f' % tuple(100 * np.array(
            [res['utterance_main'][c][b], res['scene_main'][c][b], res['interaction_noise'][c][b]]))
                                          for b in range(len(BAND_NAMES))) + ' |')
    print(f'\n[done] {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
