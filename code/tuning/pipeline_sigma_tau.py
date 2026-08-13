import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import datetime
import math
import argparse
import runpy
import builtins
import csv

sys.path.insert(0, '/storage/yotam/SAR-SSL/SAR-SSL/code')
from opt import opt_pretrain
import dataset as at_dataset
from common.utils_module import STFT, spatial_features_from_stft

class ProbeComplete(Exception):
    pass

builtins.ProbeComplete = ProbeComplete

def load_heldout_features(num_samples, batch_size, device):
    print(f"Loading up to {num_samples} samples from simulated preval set...")
    old_argv = sys.argv.copy()
    sys.argv = ['pipeline.py', '--pretrain', '--simu-exp']
    opts = opt_pretrain()
    args = opts.parse()
    dirs = opts.dir()
    sys.argv = old_argv
    
    fs = args.acoustic_setting['fs']
    dataset = at_dataset.FixMicSigDataset(
        data_dir=dirs['micsig_simu_preval'],
        fs=fs,
        load_anno=False,
        dataset_sz=num_samples,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers)
    
    do_stft = STFT(win_len=512, win_shift_ratio=0.5, nfft=512).to(device)
    feature_keys = ['coherence_low', 'coherence_mid', 'coherence_high', 'gcc_phat_max', 'tdoa']
    all_features = []
    
    loaded_count = 0
    with torch.no_grad():
        for batch in dataloader:
            waveform = batch[0] if isinstance(batch, list) else batch
            waveform = waveform.to(device)
            
            if waveform.dim() == 3:
                stft_out = do_stft(waveform)
                stft_out = stft_out.permute(0, 3, 1, 2)
                feats_dict = spatial_features_from_stft(stft_out, sample_rate=fs)
                
                batch_feats = []
                for k in feature_keys:
                    batch_feats.append(feats_dict[k].unsqueeze(1))
                batch_feats = torch.cat(batch_feats, dim=1)
                
                all_features.append(batch_feats)
                loaded_count += batch_feats.size(0)
                
                if loaded_count >= num_samples:
                    break
                    
    all_features = torch.cat(all_features, dim=0)[:num_samples]
    
    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True)
    all_features = (all_features - mean) / (3.0 * std + 1e-6)
    all_features = all_features.clamp(-1.0, 1.0)
    
    return all_features

def run_stage1(args, out_dir):
    print("\n" + "="*50)
    print(f"STAGE 1: {args.stage1_mode} distribution analysis")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = load_heldout_features(args.num_samples, args.batch_size, device)
    N = features.shape[0]
    B = args.batch_size
    
    sigmas = np.logspace(math.log10(args.sigma_min), math.log10(args.sigma_max), args.sigma_steps)
    
    if args.stage1_mode == 'tail_mass':
        m = 1.0 / (B - 1)
        std_max = math.sqrt(((B - 2) * m**2 + (1 - m)**2) / (B - 1))
        std_low = args.eps_low_ratio * std_max
        std_high = args.eps_high_ratio * std_max
        
        results = {
            'std_max': float(std_max),
            'std_low': float(std_low),
            'std_high': float(std_high),
            'sigmas': sigmas.tolist(), 
            'low_mass': [], 
            'high_mass': [], 
            'total_mass': []
        }
        
        for sigma in sigmas:
            all_stds = []
            for i in range(0, N - B + 1, B):
                batch_feats = features[i:i+B]
                D = torch.cdist(batch_feats, batch_feats, p=2)
                off_diag = ~torch.eye(B, dtype=torch.bool, device=device)
                
                D_sq = D**2
                D_sq_off_diag = D_sq.masked_fill(~off_diag, float('inf'))
                row_min = D_sq_off_diag.min(dim=1, keepdim=True).values
                W = torch.exp(- (D_sq_off_diag - row_min) / sigma**2)
                
                W_masked = W * off_diag.float()
                W_sum = W_masked.sum(dim=1, keepdim=True) + 1e-8
                W_norm = W_masked / W_sum
                
                W_norm_neighbors = W_norm[off_diag].view(B, B - 1)
                std_i = W_norm_neighbors.std(dim=1, unbiased=False)
                all_stds.append(std_i.cpu().numpy())
                
            all_stds = np.concatenate(all_stds)
            
            low_mass = np.mean(all_stds < std_low)
            high_mass = np.mean(all_stds > std_high)
            total_mass = low_mass + high_mass
            
            results['low_mass'].append(float(low_mass))
            results['high_mass'].append(float(high_mass))
            results['total_mass'].append(float(total_mass))
            
        best_idx = np.argmin(results['total_mass'])
        best_sigma = sigmas[best_idx]
        
        with open(os.path.join(out_dir, 'stage1_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        plt.figure(figsize=(10, 6))
        plt.plot(sigmas, results['low_mass'], label='Low Mass (Flat / Uninformative)')
        plt.plot(sigmas, results['high_mass'], label='High Mass (Spiky / Collapsed)')
        plt.plot(sigmas, results['total_mass'], label='Total Mass', linewidth=2, color='black')
        plt.axvline(best_sigma, color='red', linestyle='--', label=f'Best Sigma: {best_sigma:.4f}')
        plt.xscale('log')
        plt.xlabel('Sigma')
        plt.ylabel('Mass Fraction')
        plt.title('Stage 1: Tail Mass (Std) vs Sigma')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, 'stage1_tail_mass.png'))
        plt.close()
        
        top_k = min(3, len(sigmas))
        top_indices = np.argsort(results['total_mass'])[:top_k]
        candidate_sigmas = sigmas[top_indices]
        
        print(f"Stage 1 completed. Top {top_k} candidate sigmas: {candidate_sigmas}")
        return candidate_sigmas
        
    elif args.stage1_mode == 'max_entropy':
        results = {
            'sigmas': sigmas.tolist(), 
            'hist_entropy': []
        }
        
        for sigma in sigmas:
            all_w_norm = []
            for i in range(0, N - B + 1, B):
                batch_feats = features[i:i+B]
                D = torch.cdist(batch_feats, batch_feats, p=2)
                off_diag = ~torch.eye(B, dtype=torch.bool, device=device)
                
                D_sq = D**2
                D_sq_off_diag = D_sq.masked_fill(~off_diag, float('inf'))
                row_min = D_sq_off_diag.min(dim=1, keepdim=True).values
                W = torch.exp(- (D_sq_off_diag - row_min) / sigma**2)
                
                W_masked = W * off_diag.float()
                W_sum = W_masked.sum(dim=1, keepdim=True) + 1e-8
                W_norm = W_masked / W_sum
                
                W_norm_neighbors = W_norm[off_diag]
                all_w_norm.append(W_norm_neighbors.cpu())
                
            all_w_norm = torch.cat(all_w_norm)
            
            hist = torch.histc(all_w_norm, bins=100, min=0.0, max=1.0)
            hist_prob = hist / hist.sum()
            hist_prob = hist_prob[hist_prob > 0]
            hist_entropy = - (hist_prob * torch.log(hist_prob)).sum().item()
            
            results['hist_entropy'].append(float(hist_entropy))
            
        best_idx = np.argmax(results['hist_entropy'])
        best_sigma = sigmas[best_idx]
        
        with open(os.path.join(out_dir, 'stage1_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        plt.figure(figsize=(10, 6))
        plt.plot(sigmas, results['hist_entropy'], label='Histogram Entropy', color='green', linewidth=2)
        plt.axvline(best_sigma, color='red', linestyle='--', label=f'Best Sigma: {best_sigma:.4f}')
        plt.xscale('log')
        plt.xlabel('Sigma')
        plt.ylabel('Histogram Entropy')
        plt.title('Stage 1: Histogram Entropy vs Sigma')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, 'stage1_hist_entropy.png'))
        plt.close()
        
        top_k = min(3, len(sigmas))
        top_indices = np.argsort(results['hist_entropy'])[::-1][:top_k]
        candidate_sigmas = sigmas[top_indices]
        
        print(f"Stage 1 completed. Top {top_k} candidate sigmas: {candidate_sigmas}")
        return candidate_sigmas
        
    elif args.stage1_mode == 'effective_sample_size':
        results = {
            'sigmas': sigmas.tolist(), 
            'avg_ess': [],
            'std_ess': []
        }
        
        for sigma in sigmas:
            all_ess = []
            for i in range(0, N - B + 1, B):
                batch_feats = features[i:i+B]
                D = torch.cdist(batch_feats, batch_feats, p=2)
                off_diag = ~torch.eye(B, dtype=torch.bool, device=device)
                
                D_sq = D**2
                D_sq_off_diag = D_sq.masked_fill(~off_diag, float('inf'))
                row_min = D_sq_off_diag.min(dim=1, keepdim=True).values
                W = torch.exp(- (D_sq_off_diag - row_min) / sigma**2)
                
                W_masked = W * off_diag.float()
                W_sum = W_masked.sum(dim=1, keepdim=True) + 1e-8
                W_norm = W_masked / W_sum
                
                W_norm_neighbors = W_norm[off_diag].view(B, B - 1)
                
                # Effective Sample Size: 1 / sum(w^2)
                sum_sq = (W_norm_neighbors ** 2).sum(dim=1)
                sum_sq = torch.clamp(sum_sq, min=1e-7)  # Prevent division by zero
                ess = 1.0 / sum_sq
                
                # Sanity check: ESS should never exceed B - 1
                if ess.max() > B - 1 + 1e-4:
                    print(f"Warning: ESS exceeded max bound. Max ESS: {ess.max().item()}, B-1: {B-1}")
                    ess = torch.clamp(ess, max=B-1)
                    
                all_ess.append(ess.cpu())
                
            all_ess = torch.cat(all_ess)
            assert all_ess.max() <= B - 1 + 1e-4, f"ESS computed exceeded B-1 bound! Max: {all_ess.max().item()}"
            
            avg_ess = all_ess.mean().item()
            std_ess = all_ess.std().item()
            
            results['avg_ess'].append(avg_ess)
            results['std_ess'].append(std_ess)
            
        target_ess = (B-1) / 2
        avg_ess_array = np.array(results['avg_ess'])
        distance = np.abs(avg_ess_array - target_ess)
        best_idx = np.argmin(distance)
        best_sigma = sigmas[best_idx]
        
        results['target_ess'] = target_ess
        results['distance_to_target'] = distance.tolist()
        
        with open(os.path.join(out_dir, 'stage1_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        plt.figure(figsize=(10, 6))
        plt.plot(sigmas, results['avg_ess'], color='tab:blue', linewidth=2, label='Avg ESS')
        plt.axhline(target_ess, color='tab:gray', linestyle=':', label=f'Target ESS: {target_ess:.2f}')
        plt.axvline(best_sigma, color='red', linestyle='--', label=f'Best Sigma: {best_sigma:.4f}')
        plt.xscale('log')
        plt.xlabel('Sigma (log scale)')
        plt.ylabel('Average Effective Sample Size')
        plt.title('Stage 1: ESS Target Matching vs Sigma')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, 'stage1_ess_target.png'))
        plt.close()
        
        top_k = min(3, len(sigmas))
        top_indices = np.argsort(distance)[:top_k]
        candidate_sigmas = sigmas[top_indices]
        
        print(f"Stage 1 completed. Top {top_k} candidate sigmas: {candidate_sigmas}")
        return candidate_sigmas

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

def plot_trajectories(sigma, tau, probe_results, out_dir):
    steps = range(1, len(probe_results['uniformity']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Trajectories for Sigma={sigma:.4f}, Tau={tau}")
    
    axes[0].plot(steps, probe_results['uniformity'], marker='o', color='b')
    axes[0].set_title('Uniformity')
    axes[0].grid(True)
    
    axes[1].plot(steps, probe_results['alignment'], marker='o', color='g')
    axes[1].set_title('Alignment')
    axes[1].grid(True)
    
    axes[2].plot(steps, probe_results['effective_rank'], marker='o', color='r')
    axes[2].set_title('Effective Rank')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'traj_sig{sigma:.4f}_tau{tau}.png'))
    plt.close()

def update_global_csv(csv_path, res):
    rows = []
    headers = ['sigma', 'tau', 'uniformity', 'alignment', 'effective_rank']
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                
    found = False
    for row in rows:
        if abs(float(row['sigma']) - res['sigma']) < 1e-5 and abs(float(row['tau']) - res['tau']) < 1e-5:
            row['uniformity'] = f"{res['uniformity_avg']:.6f}"
            row['alignment'] = f"{res['alignment_avg']:.6f}"
            row['effective_rank'] = f"{res['effective_rank_avg']:.6f}"
            found = True
            break
            
    if not found:
        rows.append({
            'sigma': f"{res['sigma']:.6f}",
            'tau': f"{res['tau']:.6f}",
            'uniformity': f"{res['uniformity_avg']:.6f}",
            'alignment': f"{res['alignment_avg']:.6f}",
            'effective_rank': f"{res['effective_rank_avg']:.6f}"
        })
        
    rows.sort(key=lambda x: (float(x['sigma']), float(x['tau'])))
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def run_stage2(args, candidate_sigmas, out_dir):
    print("\n" + "="*50)
    print("STAGE 2: Joint (sigma, tau) probe sweep")
    print("="*50)
    
    os.environ["TAU_SWEEP"] = "1"
    
    results = []
    
    original_torch_save = torch.save
    
    for sigma in candidate_sigmas:
        for tau in args.tau_values:
            print(f"\n--- Testing Sigma = {sigma:.4f}, Tau = {tau} ---")
            
            probe_results = {'uniformity': [], 'alignment': [], 'effective_rank': []}
            step_counter = [0]
            
            def mock_torch_save(*args_mock, **kwargs):
                pass
                
            def probe_callback(uni, ali, er):
                probe_results['uniformity'].append(float(uni))
                probe_results['alignment'].append(float(ali))
                probe_results['effective_rank'].append(float(er))
                
                step_counter[0] += 1
                if step_counter[0] >= args.probe_steps:
                    print(f"\n[Probe] Sufficient steps ({args.probe_steps}) reached. Early stopping.")
                    raise builtins.ProbeComplete("Sufficient steps reached.")
                        
            builtins.PROBE_CALLBACK = probe_callback
            torch.save = mock_torch_save
            
            run_out_dir = os.path.join(out_dir, f"probe_sig{sigma:.4f}_tau{tau}")
            old_argv = sys.argv.copy()
            sys.argv = build_sys_argv(sigma, tau, run_out_dir, args.batch_size)
            
            try:
                os.chdir('/storage/yotam/SAR-SSL/SAR-SSL/code')
                runpy.run_path('run_pretrain.py', run_name='__main__')
            except builtins.ProbeComplete:
                pass
            except Exception as e:
                print(f"Genuine error during probe: {e}")
            finally:
                sys.argv = old_argv
                torch.save = original_torch_save
                if hasattr(builtins, 'PROBE_CALLBACK'):
                    del builtins.PROBE_CALLBACK
                
            # Log trajectories and check stability
            traj_len = len(probe_results['uniformity'])
            if traj_len > 0:
                plot_trajectories(sigma, tau, probe_results, out_dir)
                
                res = {
                    'sigma': float(sigma),
                    'tau': float(tau),
                    'trajectory': probe_results,
                    'uniformity_avg': float(np.mean(probe_results['uniformity'][-3:])),
                    'alignment_avg': float(np.mean(probe_results['alignment'][-3:])),
                    'effective_rank_avg': float(np.mean(probe_results['effective_rank'][-3:]))
                }
                
                # Plateau check
                if traj_len >= 6:
                    uni_last3 = res['uniformity_avg']
                    uni_prev3 = np.mean(probe_results['uniformity'][-6:-3])
                    if abs(uni_prev3) > 1e-6:
                        rel_change = abs(uni_last3 - uni_prev3) / abs(uni_prev3)
                        if rel_change > 0.1:
                            print(f"WARNING: Uniformity has not stabilized! Rel change = {rel_change*100:.1f}%")
                            res['stabilized'] = False
                        else:
                            res['stabilized'] = True
                
                results.append(res)
                print(f"Result: Uni={res['uniformity_avg']:.4f}, Ali={res['alignment_avg']:.4f}, Rank={res['effective_rank_avg']:.4f}")
                
                # Update global CSV live
                global_csv_path = '/storage/yotam/SAR-SSL/SAR-SSL/metrics/sigma_tau_pipeline/global_results.csv'
                try:
                    update_global_csv(global_csv_path, res)
                except Exception as e:
                    print(f"Warning: Failed to update global CSV: {e}")
    
    with open(os.path.join(out_dir, 'stage2_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    # Final Selection
    print("\n" + "="*50)
    print("FINAL SELECTION")
    print("="*50)
    
    valid_results = []
    for r in results:
        if r['effective_rank_avg'] >= args.min_rank and r['uniformity_avg'] <= args.max_uniformity:
            valid_results.append(r)
            
    if not valid_results:
        print(f"Warning: No valid results passed the floor (rank >= {args.min_rank}) and ceiling (uni <= {args.max_uniformity}) thresholds.")
        print("Selecting from all results instead.")
        valid_results = results
        
    best_res = min(valid_results, key=lambda x: x['uniformity_avg'])
    print(f"\n*** Selected Sigma={best_res['sigma']:.4f}, Tau={best_res['tau']} ***")
    print(f"Metrics -> Uniformity: {best_res['uniformity_avg']:.4f}, Alignment: {best_res['alignment_avg']:.4f}, Rank: {best_res['effective_rank_avg']:.4f}")
    
    with open(os.path.join(out_dir, 'selected_sigma_tau.json'), 'w') as f:
        json.dump(best_res, f, indent=4)
        
    print(f"\nPipeline completed successfully! Results saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eps-low-ratio', type=float, default=0.1)
    parser.add_argument('--eps-high-ratio', type=float, default=0.9)
    parser.add_argument('--sigma-min', type=float, default=0.01)
    parser.add_argument('--sigma-max', type=float, default=10.0)
    parser.add_argument('--sigma-steps', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=5000)
    parser.add_argument('--tau-values', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument('--probe-steps', type=int, default=20)
    parser.add_argument('--min-rank', type=float, default=1.5, help="Floor for effective rank to avoid collapsed representations")
    parser.add_argument('--max-uniformity', type=float, default=-0.5, help="Ceiling for uniformity")
    parser.add_argument('--stage1-only', action='store_true', help="Run only Stage 1")
    parser.add_argument('--stage1-mode', type=str, default='effective_sample_size', choices=['tail_mass', 'max_entropy', 'effective_sample_size'], help="Mode for Stage 1 sigma selection")
    parser.add_argument('--force-sigmas', type=float, nargs='+', help="Manually specify sigmas for Stage 2 (skips Stage 1 if provided)")
    
    args, unknown = parser.parse_known_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/storage/yotam/SAR-SSL/SAR-SSL/metrics/sigma_tau_pipeline/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    if args.force_sigmas:
        candidate_sigmas = np.array(args.force_sigmas)
        print(f"Skipping Stage 1. Using forced sigmas: {candidate_sigmas}")
    else:
        candidate_sigmas = run_stage1(args, out_dir)
        
    if not args.stage1_only:
        run_stage2(args, candidate_sigmas, out_dir)
