#!/usr/bin/env python3
""" One registry of every SAR-SSL-side result, written into the results worktree so the `results` branch has it all.

    Fills the two gaps the per-run collector leaves:
      - pretraining runs (args.json + log.jsonl of exp/{ssamba,joint}_pretrain/<run>) -> results/pretrain/<run>/
      - speech probes on SC2/SID (sac-ssamba exp/_probes/*.jsonl, incl. Pilot A/B/C and leakage probes)
        -> results/probe_speech/<file>.jsonl
    and writes results/RUNS.md: per pretraining run (recipe, final val), per eval label (which pretraining run /
    checkpoint it read, status, one-line description, every task's finetune / frozen / scratch test MAE), per probe.

    Descriptions and status come from results/labels.tsv (hand-maintained: label <TAB> pretrain run or '-' <TAB>
    status ok|void|smoke|old <TAB> description). Labels missing from it are listed as UNDESCRIBED so they get one.
    Stdlib only, Python >= 3.6. Run by scripts/sync_results.sh before each commit; safe to run any time.

    Usage: build_registry.py <results dir> [--exp /scratch200/.../SAC-SAR-SSL/exp] [--probes .../sac-ssamba/exp/_probes]
"""
import argparse
import glob
import json
import os
import re
import shutil

TASKS = ('TDOA', 'T60', 'DRR', 'C50', 'ABS')
MODES = (('finetune', 'ft'), ('lineareval', 'frz'), ('scratchLOW', 'scr'))
RECIPE_KEYS = ('streams', 'fusion', 'input', 'nepoch', 'num_train', 'sac_lambda', 'sac_form', 'swap_lambda',
               'own_scale', 'seed', 'tar')
MAX_COPY = 1 << 20  # the results branch never stores files over 1 MB


def read_labels(path):
    out = {}
    if os.path.isfile(path):
        for line in open(path):
            if line.strip() and not line.startswith('#'):
                f = (line.rstrip('\n').split('\t') + ['', '', '', ''])[:4]
                out[f[0].strip()] = dict(pretrain=f[1].strip(), status=f[2].strip() or 'ok', desc=f[3].strip())
    return out


def ckpt_run(model):
    """ pretraining run(s) an eval read, from the summary's model text. """
    m = re.findall(r'exp/(?:ssamba|joint)_pretrain/([^/\s,]+)/', model or '')
    if m:
        return '+'.join(dict.fromkeys(m))
    if 'ssamba_ckpt random' in (model or '') or (model or '').strip() == 'random':
        return 'random init'
    m = re.search(r'/([^/\s]+)/models/(audio_model\.[^/\s]+)\.pth', model or '')
    if m:  # sac-ssamba run: the run-name tail + which checkpoint
        tail = re.split(r'librispeech-|global_median-', m.group(1))[-1]
        return 'sac-ssamba %s:%s' % (tail, m.group(2).replace('audio_model.', ''))
    g = re.search(r'gramt_model (\S+)', model or '')
    return g.group(1) if g else '?'


def fmt(x, task):
    if x is None:
        return ''
    return ('%.3f' if task in ('T60',) else '%.2f') % x if task != 'TDOA' or x >= 1 else '%.3f' % x


def copy_small(src, dst):
    if os.path.getsize(src) <= MAX_COPY:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.isfile(dst) or open(src, 'rb').read() != open(dst, 'rb').read():
            shutil.copyfile(src, dst)
        return True
    return False


def pretrain_runs(exp, R):
    rows = []
    for d in sorted(glob.glob(os.path.join(exp, '*_pretrain', '*'))):
        run = os.path.basename(d)
        if run.startswith('_') or not os.path.isfile(os.path.join(d, 'args.json')):
            continue
        a = json.load(open(os.path.join(d, 'args.json')))
        dst = os.path.join(R, 'pretrain', run)
        copy_small(os.path.join(d, 'args.json'), os.path.join(dst, 'args.json'))
        last_val, last_ep = None, None
        log = os.path.join(d, 'log.jsonl')
        if os.path.isfile(log):
            copy_small(log, os.path.join(dst, 'log.jsonl'))
            for line in open(log):
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                if 'val' in j:
                    last_val, last_ep = j['val'], j.get('epoch')
        done = os.path.isfile(os.path.join(d, 'encoder_final.pth')) or bool(glob.glob(os.path.join(d, '*_final.pth')))
        rows.append(dict(run=run, family=os.path.basename(os.path.dirname(d)), args=a, val=last_val, ep=last_ep,
                         done=done))
    return rows


def evals(R):
    by = {}  # label -> task -> newest summary
    for f in glob.glob(os.path.join(R, '*', '*__*', 'summary.json')):
        task = f.split(os.sep)[-3]
        if task not in TASKS or f.split(os.sep)[-2].endswith('__runai'):
            continue
        s = json.load(open(f))
        lab = s.get('experiment') or f.split(os.sep)[-2].rsplit('__', 1)[0]
        prev = by.setdefault(lab, {}).get(task)
        n = (prev or {}).get('_n', 0) + 1
        if prev is None or (s.get('date') or '') >= (prev.get('date') or ''):
            s['_n'] = n
            by[lab][task] = s
        else:
            prev['_n'] = n
    return by


def probes(pdir, R):
    rows = []
    for f in sorted(glob.glob(os.path.join(pdir, '*.jsonl'))):
        name = os.path.basename(f)[:-6]
        if 'smoke' in name:
            continue
        copy_small(f, os.path.join(R, 'probe_speech', name + '.jsonl'))
        for line in open(f):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            rows.append(dict(file=name, model=j.get('model', ''), j=j))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--exp', default='/scratch200/yotams2/SAC-SAR-SSL/exp')
    ap.add_argument('--probes', default='/scratch200/yotams2/sac-ssamba/exp/_probes')
    a = ap.parse_args()
    R = a.results
    labels = read_labels(os.path.join(R, 'labels.tsv'))
    L = ['# Run registry (generated by scripts/build_registry.py; edit results/labels.tsv, not this file)', '',
         'Test MAE, lower is better. ft = finetune, frz = frozen (lineareval), scr = scratchLOW. One seed unless noted.',
         '`n=2` = the label has 2 collected jobs for that task; the newest is shown. Status void/smoke/old rows are',
         'kept for the record, not for comparison.', '']

    L += ['## Pretraining runs', '', '| run | family | status | recipe | last val (epoch) | description |',
          '|---|---|---|---|---|---|']
    for r in pretrain_runs(a.exp, R) if os.path.isdir(a.exp) else []:
        rec = ', '.join('%s=%s' % (k, os.path.basename(str(r['args'][k])) if k == 'tar' else r['args'][k])
                        for k in RECIPE_KEYS if r['args'].get(k) not in (None, ''))
        val = ', '.join('%s %.3f' % (k, v) for k, v in (r['val'] or {}).items() if isinstance(v, (int, float)))
        lab = labels.get(r['run'], {})
        L.append('| %s | %s | %s | %s | %s (ep %s) | %s |' % (
            r['run'], r['family'].replace('_pretrain', ''), lab.get('status', 'ok') + ('' if r['done'] else ', running'),
            rec, val, r['ep'], lab.get('desc', 'UNDESCRIBED')))

    L += ['', '## Spatial evals (SAR-SSL tasks)', '',
          '| label | status | pretrain run (ckpt) | ' + ' | '.join(TASKS) + ' | description |',
          '|---|---|---|' + '---|' * len(TASKS) + '---|']
    by = evals(R)
    order = sorted(by, key=lambda k: max((s.get('date') or '') for s in by[k].values()), reverse=True)
    for lab in order:
        tasks = by[lab]
        anyS = next(iter(tasks.values()))
        meta = labels.get(lab, {})
        cells = []
        for t in TASKS:
            s = tasks.get(t)
            if not s:
                cells.append('')
                continue
            parts = ['%s %s' % (short, fmt(s['metrics'][m].get('lowest'), t)) for m, short in MODES if m in s['metrics']]
            cells.append(' / '.join(parts) + (' (n=%d)' % s['_n'] if s['_n'] > 1 else ''))
        pre = meta.get('pretrain') if meta.get('pretrain') not in (None, '', '-') else ckpt_run(anyS.get('model'))
        L.append('| %s | %s | %s | %s | %s |' % (lab, meta.get('status', 'ok'), pre, ' | '.join(cells),
                                                 meta.get('desc', 'UNDESCRIBED')))

    L += ['', '## Speech probes (SC2 / SID, frozen, linear accuracy %)', '',
          '| probe file | status | model (ckpt) | SC2 linear | SID linear | description |', '|---|---|---|---|---|---|']
    if os.path.isdir(a.probes):
        for p in probes(a.probes, R):
            j, meta = p['j'], labels.get(p['file'], {})
            st = meta.get('status') or ('void' if 'VOID' in p['file'] else 'ok')
            pct = lambda k: '%.1f' % (100 * j[k]) if isinstance(j.get(k), (int, float)) else ''
            m = p['model']
            L.append('| %s | %s | %s | %s | %s | %s |' % (
                p['file'], st, ckpt_run(m) if m not in ('', 'gramt') else (m or '?'),
                pct('sc2/linear_acc'), pct('sid/linear_acc'), meta.get('desc', 'UNDESCRIBED')))

    open(os.path.join(R, 'RUNS.md'), 'w').write('\n'.join(L) + '\n')
    print('[build_registry] wrote %s' % os.path.join(R, 'RUNS.md'))


if __name__ == '__main__':
    main()
