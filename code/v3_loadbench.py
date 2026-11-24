""" V3 load-time benchmark (CPU, 11-24): --v3-stage-local (node-local $TMPDIR) vs reading /scratch200.
Stages the unpaired tar + WPE shards with v3_data.stage_local (timed), then loads the same random training items
(5 workers, batch 32) from each copy; reports s / item and the implied s per batch of 128. Usage: v3_loadbench.py <out json>
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_data import V3Dataset, stage_local  # noqa: E402

M = '/scratch200/yotams2/SAC-SAR-SSL/data/MicSig'
TAR, WPE, POOL = f'{M}/simu_unpaired8_pretrain.tar', f'{M}/simu_unpaired8_wpe', f'{M}/simu_paired_pretrain_v2.tar.rirs.npz'


def bench(tar, wpe, idx, workers=5):
	ds = V3Dataset(tar, wpe, POOL, 0, 22500, swap_halves=True)
	dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=32, num_workers=workers)
	t0, n = time.time(), 0
	for sig3, _ in dl:
		n += sig3.shape[0]
	dt = time.time() - t0
	return {'items': n, 's_per_item': dt / n, 's_per_batch128': dt / n * 128}


def main(out):
	res = {'tmpdir': os.environ.get('TMPDIR'), 'stage_dir': os.environ.get('STAGE_DIR'), 'host': os.uname().nodename}
	from ssamba_spatial_pretrain import tar_index
	for sh in sorted(os.listdir(WPE)):  # build the shard index caches once (the loader reuses them)
		if sh.endswith('.tar'):
			tar_index(os.path.join(WPE, sh))
	t0 = time.time()
	ltar, lwpe = stage_local(TAR, WPE)  # copies the shard index caches built above
	res['stage_s'], res['staged_tar'], res['staged_wpe'] = time.time() - t0, ltar != TAR, lwpe != WPE
	rng = np.random.default_rng(1)
	perm = rng.permutation(22500).tolist()
	sets = [perm[i * 640:(i + 1) * 640] for i in range(5)]  # disjoint items: no page-cache reuse between passes
	bench(TAR, WPE, sets[0][:128])  # warm-up (imports, worker start, RIR pool)
	order = [('scratch', TAR, WPE), ('local', ltar, lwpe)] * 2 if ltar != TAR or lwpe != WPE else [('scratch', TAR, WPE)] * 2
	for k, (name, t, w) in enumerate(order):
		res[f'{name}_{k // 2 + 1}' if len(order) == 4 else f'{name}_{k + 1}'] = bench(t, w, sets[k + 1])
	print(json.dumps(res, indent=1), flush=True)
	json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
	main(sys.argv[1])
