# CPU check of learner.save_checkpoint_light (11-27): a model with a frozen encoder + BatchNorm head is trained for
# 8 epochs twice from the same seed, once with save_checkpoint(save_extra_hist=True), once with the light path; the
# ensemble of epochs [best-4, best], best_model.tar and latest_model.tar must be bit-identical. Also checks that the
# light per-epoch files hold only the head. Usage: python scripts/diag/light_ckpt_check.py <empty work dir>
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))
import torch, numpy as np
import learner as L

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.LayerNorm(256), torch.nn.Linear(256, 128))
        self.mlp_head = torch.nn.Sequential(torch.nn.BatchNorm1d(128), torch.nn.Linear(128, 3))
        for p in self.encoder.parameters(): p.requires_grad = False

class Lrn(L.Learner):
    def data_preprocess(self, *a, **k): pass
    def loss(self, *a, **k): pass
    def evaluate(self, *a, **k): pass
    def pretrain_evaluate(self, *a, **k): pass

def run(d, light):
    torch.manual_seed(0); np.random.seed(0)
    net = Net(); lr = Lrn(net); lr.device = 'cpu'
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1)
    x = torch.randn(512, 64); y = torch.randn(512, 3)
    scores = [0.1, 0.5, 0.4, 0.7, 0.6, 0.65, 0.9, 0.2]
    best = 0
    for ep in range(1, 9):
        net.train()
        for i in range(0, 512, 64):
            opt.zero_grad(); torch.nn.functional.mse_loss(net.mlp_head(net.encoder(x[i:i+64])), y[i:i+64]).backward(); opt.step()
        is_best = lr.is_best_epoch(scores[ep - 1]); best = ep if is_best else best
        if light: lr.save_checkpoint_light(ep, d, is_best_epoch=is_best)
        else: lr.save_checkpoint(ep, d, is_best_epoch=is_best, save_extra_hist=True)
    if light: lr.flush_checkpoints_light(d)
    lr.ensembling(d, list(range(max(1, best - 4), best + 1)))
    return best

w = sys.argv[1]; a, b = os.path.join(w, 'full'), os.path.join(w, 'light'); os.makedirs(a); os.makedirs(b)
ba, bb = run(a, False), run(b, True)
assert ba == bb == 7, (ba, bb)
ok = True
for f in ('ensemble_model.tar', 'best_model.tar', 'latest_model.tar'):
    sa, sb = torch.load(os.path.join(a, f))['model'], torch.load(os.path.join(b, f))['model']
    same = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    print(f'{f}: bit-identical {same}'); ok &= same
for f in ('epoch', 'max_score'):
    print('best_model', f, torch.load(os.path.join(a, 'best_model.tar'))[f], torch.load(os.path.join(b, 'best_model.tar'))[f])
d5 = torch.load(os.path.join(b, 'model5.tar'))['model_delta']
print('light model5.tar keys:', sorted(d5)); ok &= all(k.startswith('mlp_head') for k in d5)
sz = lambda d: sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
print(f'bytes on disk: full {sz(a)}, light {sz(b)}')
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
