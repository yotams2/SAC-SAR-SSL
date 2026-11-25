"""Check a resume state before resuming from it (11-25): it unpickles, every tensor is finite, and its model keys /
shapes match a reference checkpoint of the same run (e.g. joint_best.pth). Exit 1 on any problem.
Usage: python scripts/verify_ckpt.py <latest.pth> [<reference.pth>]"""
import sys
import torch

ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print('keys:', sorted(ck))
print('step:', ck.get('step'), 'best_val:', ck.get('best_val'))
sd = ck['model']
bad = [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
n_opt = sum(1 for s in ck['opt']['state'].values() for v in s.values() if torch.is_tensor(v))
bad_opt = sum(1 for s in ck['opt']['state'].values() for v in s.values()
			  if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all())
print(f'model tensors {len(sd)}, non-finite {len(bad)} {bad[:5]}; optimizer tensors {n_opt}, non-finite {bad_opt}')
ok = not bad and not bad_opt and n_opt > 0
if len(sys.argv) > 2:
	ref = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
	ref = ref.get('model', ref)
	miss, extra = sorted(set(ref) - set(sd)), sorted(set(sd) - set(ref))
	shape = [k for k in set(ref) & set(sd) if torch.is_tensor(ref[k]) and ref[k].shape != sd[k].shape]
	print(f'vs reference: missing {len(miss)} {miss[:5]}, extra {len(extra)} {extra[:5]}, shape mismatch {len(shape)} {shape[:5]}')
	ok = ok and not shape and not miss
print('VERIFY', 'OK' if ok else 'FAILED')
sys.exit(0 if ok else 1)
