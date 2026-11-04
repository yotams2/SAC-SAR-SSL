#!/usr/bin/env python3
"""Write SAR-SSL results to results/<task>/<time>__<job_id>/ in the `results` worktree (~/SAC-SAR-SSL-results).

Stdlib only and Python 3.9 compatible: it has to run where torch can't, and needs no venv. A run is one model timestamp
under one task: <exp>/<TDOA|T60|DRR|ABS|C50>/<time>/ for downstream, <exp>/pretrain/<time>/ for pretraining.

    python3 scripts/collect_result.py --exp_root <exp>                 # every run under exp, plus summary_tables/
    python3 scripts/collect_result.py --exp_root <exp> --task T60 --time 09151152 --job_id 12345

Per test mode it keeps what code/gen_summary_table.py computes: lowest, mean, seed_std (population std across trials,
which is seed to seed) and the raw trials. metric_test is an error (MAE), so lower is better. Never fails a job:
a run that can't be parsed gets a summary.json with a parse_error field, and the exit code is always 0.
"""
import argparse
import datetime
import fcntl
import glob
import json
import os
import re
import shutil
import statistics
import sys
import tempfile

TASKS = ('TDOA', 'T60', 'DRR', 'ABS', 'C50')
RESULT_RE = re.compile(r'downstream_test_res(?:_(.+))?\.json$')
MAX_FILE_BYTES = 1000 * 1000
NOTE_FILES = ('description.txt', 'pretext_test_res.json')


def read_json(path):
    with open(path, errors='replace') as f:
        return json.load(f)


def summarise_trials(entries):
    """Same numbers as generate_json_data() in gen_summary_table.py."""
    values = [float(e['metric_test']) for e in entries]
    stds = [e.get('metric_test_std') for e in entries]
    valid_stds = [float(s) for s in stds if s is not None]
    return {
        'lowest': min(values),
        'mean': statistics.fmean(values),
        'seed_std': statistics.pstdev(values),
        'n_trials': len(values),
        'mean_test_std': statistics.fmean(valid_stds) if valid_stds else None,
        'trials': [{'trial_idx': e.get('trial_idx'), 'lr_idx': e.get('lr_idx'), 'bs_idx': e.get('bs_idx'),
                    'metric_test': float(e['metric_test']), 'loss_test': e.get('loss_test')} for e in entries],
    }


def collect_run(run_dir, task, job_id, stage):
    """Fills `stage` and writes its summary.json."""
    time_stamp = os.path.basename(run_dir.rstrip('/'))
    summary = {
        'task': task, 'experiment': time_stamp, 'model': None, 'job_id': job_id, 'status': 'unknown',
        'date': None, 'collected_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'seed': None, 'lr': None, 'frozen': None, 'git_commit': None, 'ckpt_dir': run_dir, 'metrics': {},
    }
    errors = []
    desc = os.path.join(run_dir, 'description.txt')
    if os.path.isfile(desc):
        with open(desc, errors='replace') as f:
            summary['model'] = f.read().strip()[:500] or None

    copy = [os.path.join(run_dir, name) for name in NOTE_FILES]
    if task == 'pretrain':
        config = os.path.join(run_dir, 'config.json')
        copy.append(config)
        try:
            if os.path.isfile(config):
                data = read_json(config)
                if isinstance(data, list):   # run_pretrain.py saves [args, ...]
                    data = data[0]
                summary['seed'] = data.get('seed')
                summary['lr'] = data.get('lr')
            pretext = os.path.join(run_dir, 'pretext_test_res.json')
            if os.path.isfile(pretext):
                summary['metrics']['pretext_test'] = read_json(pretext)
                summary['status'] = 'done'
        except Exception as e:
            errors.append(f'{type(e).__name__}: {e}')
    else:
        for path in sorted(glob.glob(os.path.join(run_dir, 'downstream_test_res*.json'))):
            m = RESULT_RE.search(os.path.basename(path))
            if not m:          # "... copy.json" and the like
                continue
            mode = m.group(1) or 'default'
            try:
                summary['metrics'][mode] = summarise_trials(read_json(path))
                copy.append(path)
            except Exception as e:
                errors.append(f'{os.path.basename(path)}: {type(e).__name__}: {e}')
        if summary['metrics']:
            summary['status'] = 'done'
        elif not errors:
            summary['status'] = 'incomplete'
            errors.append(f'no downstream_test_res*.json in {run_dir}')
        modes = set(summary['metrics'])
        if modes and modes <= {'lineareval', 'lineareval_real'}:
            summary['frozen'] = True
        elif modes and not modes & {'lineareval', 'lineareval_real'}:
            summary['frozen'] = False

    times = []
    for path in copy:
        if os.path.isfile(path):
            times.append(os.path.getmtime(path))
            if os.path.getsize(path) <= MAX_FILE_BYTES:
                shutil.copyfile(path, os.path.join(stage, os.path.basename(path)))
    if times:
        summary['date'] = datetime.datetime.fromtimestamp(max(times)).astimezone().isoformat(timespec='seconds')
    if errors:
        summary['parse_error'] = '; '.join(errors)
    with open(os.path.join(stage, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def publish(stage, target, root):
    # Same lock as sync_results.sh: its rebase must not see a half-written folder
    with open(os.path.join(root, '.sync.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copytree(stage, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp_root', required=True, help='folder holding pretrain/, TDOA/, ...')
    parser.add_argument('--task', choices=TASKS + ('pretrain',), help='with --time: collect one run')
    parser.add_argument('--time', help='model timestamp')
    parser.add_argument('--job_id', default=os.environ.get('SLURM_JOB_ID', 'runai'),
                        help='default: the Slurm job, else runai (runs copied from the RunAI server)')
    parser.add_argument('--results_root', default=os.environ.get('SAC_RESULTS_ROOT', os.path.expanduser('~/SAC-SAR-SSL-results')))
    args = parser.parse_args()

    if not os.path.isdir(args.results_root):
        print(f'[collect_result] no results worktree at {args.results_root}, nothing written')
        return
    if args.task and args.time:
        runs = [(args.task, os.path.join(args.exp_root, args.task, args.time))]
    else:
        runs = [(task, d) for task in TASKS + ('pretrain',)
                for d in sorted(glob.glob(os.path.join(args.exp_root, task, '*'))) if os.path.isdir(d)]

    counts = {}
    for task, run_dir in runs:
        time_stamp = os.path.basename(run_dir.rstrip('/'))
        with tempfile.TemporaryDirectory() as stage:
            summary = collect_run(run_dir, task, args.job_id, stage)
            publish(stage, os.path.join(args.results_root, 'results', task, f'{time_stamp}__{args.job_id}'), args.results_root)
        counts[summary['status']] = counts.get(summary['status'], 0) + 1
        if 'parse_error' in summary:
            print(f"[collect_result] {task}/{time_stamp}: {summary['parse_error']}")

    tables = glob.glob(os.path.join(args.exp_root, 'summary_tables', '*.md'))
    if tables and not args.task:
        target = os.path.join(args.results_root, 'results', 'summary_tables')
        os.makedirs(target, exist_ok=True)
        for path in tables:
            if os.path.getsize(path) <= MAX_FILE_BYTES:
                shutil.copyfile(path, os.path.join(target, os.path.basename(path)))
    print(f'[collect_result] {len(runs)} runs: {counts}, {len(tables)} summary tables')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # a job must never fail because of its results bookkeeping
        print(f'[collect_result] failed: {type(e).__name__}: {e}', file=sys.stderr)
    sys.exit(0)
