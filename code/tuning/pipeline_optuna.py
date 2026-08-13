import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import argparse
import runpy
import builtins
import csv
import gc

# We import optuna for hyperparameter optimization
import optuna

sys.path.insert(0, '/storage/yotam/SAR-SSL/SAR-SSL/code')
from opt import opt_pretrain

class ProbeComplete(Exception):
    pass

builtins.ProbeComplete = ProbeComplete

def build_sys_argv(sigma, tau, out_dir, bs):
    return [
        "run_pretrain.py",
        "--pretrain",
        "--simu-exp",
        "--gpu-id", "0,",
        "--no-wandb",
        "--feature-sim-loss",
        "--feature-sim-temperature", str(tau),
        "--feature-sim-sigma", str(sigma),
        "--bs", str(bs), str(bs), str(bs),
        "--exp-dir", out_dir
    ]

def objective(trial, args, out_dir):
    # Suggest hyper-parameters
    # We sample sigma logarithmically since it can span multiple scales
    sigma = trial.suggest_float('sigma', args.sigma_min, args.sigma_max, log=True)
    # Tau is typically in [0.01, 1.0]
    tau = trial.suggest_float('tau', args.tau_min, args.tau_max)
    
    print(f"\n--- Optuna Trial {trial.number}: Testing Sigma = {sigma:.4f}, Tau = {tau:.4f} ---")
    
    os.environ["TAU_SWEEP"] = "1"
    
    probe_results = {'uniformity': [], 'alignment': [], 'effective_rank': []}
    step_counter = [0]
    
    original_torch_save = torch.save
    def mock_torch_save(*args_mock, **kwargs):
        pass
        
    def probe_callback(uni, ali, er):
        probe_results['uniformity'].append(float(uni))
        probe_results['alignment'].append(float(ali))
        probe_results['effective_rank'].append(float(er))
        
        step_counter[0] += 1
        
        # We can report intermediate values to Optuna, but since we are running very few steps (e.g. 20),
        # we'll just wait for the probe to finish.
        if step_counter[0] >= args.probe_steps:
            raise builtins.ProbeComplete("Sufficient steps reached.")
            
    builtins.PROBE_CALLBACK = probe_callback
    torch.save = mock_torch_save
    
    run_out_dir = os.path.join(out_dir, f"trial_{trial.number}_sig{sigma:.4f}_tau{tau:.4f}")
    # opt.py prepends work_dir + '/SAR-SSL/' to --exp-dir, so we must pass a relative path
    rel_run_out_dir = os.path.relpath(run_out_dir, '/storage/yotam/SAR-SSL/SAR-SSL')
    old_argv = sys.argv.copy()
    sys.argv = build_sys_argv(sigma, tau, rel_run_out_dir, args.batch_size)
    
    try:
        os.chdir('/storage/yotam/SAR-SSL/SAR-SSL/code')
        runpy.run_path('run_pretrain.py', run_name='__main__')
    except builtins.ProbeComplete:
        pass
    except Exception as e:
        print(f"Genuine error during probe: {e}")
        # Restore state and prune if the run crashes
        sys.argv = old_argv
        torch.save = original_torch_save
        if hasattr(builtins, 'PROBE_CALLBACK'):
            del builtins.PROBE_CALLBACK
        raise optuna.exceptions.TrialPruned()
    finally:
        sys.argv = old_argv
        torch.save = original_torch_save
        if hasattr(builtins, 'PROBE_CALLBACK'):
            del builtins.PROBE_CALLBACK
        # Clean up memory to prevent CUDA OOM across trials
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    traj_len = len(probe_results['uniformity'])
    if traj_len < 3:
        raise optuna.exceptions.TrialPruned()
        
    res = {
        'sigma': float(sigma),
        'tau': float(tau),
        'uniformity_avg': float(np.mean(probe_results['uniformity'][-3:])),
        'alignment_avg': float(np.mean(probe_results['alignment'][-3:])),
        'effective_rank_avg': float(np.mean(probe_results['effective_rank'][-3:]))
    }
    
    print(f"Result for Trial {trial.number}: Uni={res['uniformity_avg']:.4f}, Ali={res['alignment_avg']:.4f}, Rank={res['effective_rank_avg']:.4f}")
    
    # Store metrics to be used by the constraint function
    trial.set_user_attr('uniformity', res['uniformity_avg'])
    trial.set_user_attr('effective_rank', res['effective_rank_avg'])
    
    # Objective: Maximize Alignment
    return res['alignment_avg']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sigma-min', type=float, default=0.01)
    parser.add_argument('--sigma-max', type=float, default=10.0)
    parser.add_argument('--tau-min', type=float, default=0.05)
    parser.add_argument('--tau-max', type=float, default=1.0)
    parser.add_argument('--probe-steps', type=int, default=20)
    parser.add_argument('--min-rank', type=float, default=3.2, help="Floor for effective rank to avoid collapsed representations")
    parser.add_argument('--max-uniformity', type=float, default=-1.5, help="Ceiling for uniformity")
    parser.add_argument('--n-trials', type=int, default=30, help="Number of optuna trials")
    
    args, unknown = parser.parse_known_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/storage/yotam/SAR-SSL/SAR-SSL/metrics/sigma_tau_pipeline/optuna/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    def constraints(trial):
        # Optuna constraints: <= 0 is feasible, > 0 is infeasible.
        # We want: uniformity <= max_uniformity  ==> uniformity - max_uniformity <= 0
        # We want: effective_rank >= min_rank    ==> min_rank - effective_rank <= 0
        uni = trial.user_attrs.get('uniformity', float('inf'))
        rank = trial.user_attrs.get('effective_rank', 0.0)
        return (uni - args.max_uniformity, args.min_rank - rank)
        
    # We use a persistent SQLite database so results are saved as they happen and can be resumed
    storage_path = "/storage/yotam/SAR-SSL/SAR-SSL/metrics/sigma_tau_pipeline/optuna/optuna_study.db"
    storage_url = f"sqlite:///{storage_path}"
    
    study_name = "sigma_tau_tuning"
    
    # TPESampler handles constraints natively
    sampler = optuna.samplers.TPESampler(constraints_func=constraints, seed=42)
    study = optuna.create_study(
        direction="maximize", 
        sampler=sampler, 
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True
    )
    
    print(f"Starting Optuna optimization for {args.n_trials} trials...")
    print(f"Objective: Maximize Alignment")
    print(f"Constraints: Uniformity <= {args.max_uniformity}, Effective Rank >= {args.min_rank}")
    print(f"Storage: {storage_path} (Safe to interrupt and resume!)")
    
    # Save the current state to CSV before starting in case we are resuming
    if len(study.trials) > 0:
        print(f"Resuming study '{study_name}' with {len(study.trials)} existing trials.")
    
    # Optional: Callback to save CSV after every trial
    def save_csv_callback(study, trial):
        df = study.trials_dataframe()
        df.to_csv(os.path.join(out_dir, 'optuna_trials.csv'), index=False)
        
    study.optimize(lambda trial: objective(trial, args, out_dir), n_trials=args.n_trials, callbacks=[save_csv_callback])
    
    print("\n" + "="*50)
    print("OPTUNA TUNING COMPLETED")
    print("="*50)
    
    try:
        best_trial = study.best_trial
        print("Best trial found satisfying constraints!")
        print(f"  Value (Alignment): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value:.4f}")
        
        print("  Metrics:")
        for key, value in best_trial.user_attrs.items():
            print(f"    {key}: {value:.4f}")
            
        best_res = {
            'sigma': best_trial.params['sigma'],
            'tau': best_trial.params['tau'],
            'alignment_avg': best_trial.value,
            'uniformity_avg': best_trial.user_attrs.get('uniformity'),
            'effective_rank_avg': best_trial.user_attrs.get('effective_rank')
        }
        with open(os.path.join(out_dir, 'best_sigma_tau.json'), 'w') as f:
            json.dump(best_res, f, indent=4)
            
    except ValueError:
        print("No trials satisfied the constraints (min-rank and max-uniformity).")
        
    # Save all trials to CSV
    df = study.trials_dataframe()
    df.to_csv(os.path.join(out_dir, 'optuna_trials.csv'), index=False)
    print(f"All trials saved to {os.path.join(out_dir, 'optuna_trials.csv')}")

if __name__ == "__main__":
    main()
