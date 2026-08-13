import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import builtins
import datetime
import json
import runpy
import re
from unittest.mock import patch

def build_sys_argv(tau, out_dir):
    return [
        "run_pretrain.py",
        "--pretrain",
        "--simu-exp",
        "--gpu-id", "0,",
        "--no-wandb",
        "--feature-sim-loss",
        "--feature-sim-temperature", str(tau),
        "--feature-sim-sigma", "0.2947",
        "--bs", "128", "128", "128",
        "--exp-dir", f"../metrics/tau_search/exp_tau_{tau}"
    ]

def run_sweep():
    os.environ["TAU_SWEEP"] = "1"
    tau_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/storage/yotam/SAR-SSL/SAR-SSL/metrics/tau_search/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    metadata = {
        "timestamp": timestamp,
        "tau_values_swept": tau_values,
    }
    
    with open(os.path.join(out_dir, "sweep_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    results = {}
    
    original_print = builtins.print
    original_torch_save = torch.save
    
    for tau in tau_values:
        print(f"\n{'='*50}\nStarting Sweep for tau = {tau}\n{'='*50}")
        
        tau_results = {
            'steps': [],
            'uniformity': [],
            'alignment': [],
        }
        
        step_counter = [0]
        
        def mock_torch_save(*args, **kwargs):
            pass
            
        def mock_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            original_print(*args, **kwargs)
            
            if "  [Diagnostics] Uniformity:" in text:
                try:
                    parts = text.split(',')
                    uni_str = parts[0].split('Uniformity:')[1].strip()
                    ali_str = parts[1].split('Alignment_Top10:')[1].strip()
                    
                    tau_results['uniformity'].append(float(uni_str))
                    tau_results['alignment'].append(float(ali_str))
                    
                    step_counter[0] += 1
                    tau_results['steps'].append(step_counter[0])
                    
                    # Stop after 20 diagnostic steps (which is 20 * 20 batches = 400 batches, enough for stable metrics)
                    if step_counter[0] >= 20:
                        original_print(f"\n[Tau Sweep] Sufficient steps reached. Early stopping for tau={tau}.\n")
                        raise StopIteration("Early stopping")
                except StopIteration:
                    raise
                except Exception as e:
                    original_print(f"Error parsing diagnostics: {e}")
                    
        builtins.print = mock_print
        torch.save = mock_torch_save
        
        sys.argv = build_sys_argv(tau, out_dir)
        
        try:
            os.chdir('/storage/yotam/SAR-SSL/SAR-SSL/code')
            runpy.run_path('run_pretrain.py', run_name='__main__')
        except StopIteration:
            pass
        except Exception as e:
            original_print(f"Error during run for tau={tau}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            builtins.print = original_print
            
        results[tau] = tau_results
        
        # --- Progressive Saving ---
        results_path = os.path.join(out_dir, "tau_sweep_results.npy")
        np.save(results_path, results)
        original_print(f"\n[Progress] Saved tracking matrix up to tau={tau} to {results_path}")
        
        plot_path = os.path.join(out_dir, "tau_trajectory_plot.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, completed_tau in enumerate(results.keys()):
            steps = results[completed_tau]['steps']
            uniformity = results[completed_tau]['uniformity']
            alignment = results[completed_tau]['alignment']
            
            color = colors[i % len(colors)]
            ax1.plot(steps, uniformity, label=f'$\\tau$ = {completed_tau}', color=color, linewidth=2, marker='o')
            ax2.plot(steps, alignment, label=f'$\\tau$ = {completed_tau}', color=color, linewidth=2, marker='o')

        ax1.set_title('Latent Uniformity Trajectory', fontsize=14)
        ax1.set_xlabel('Training Steps (Diagnostic intervals)', fontsize=12)
        ax1.set_ylabel('Uniformity (Lower is Better)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=11)
        
        ax2.set_title('Latent Alignment Trajectory', fontsize=14)
        ax2.set_xlabel('Training Steps (Diagnostic intervals)', fontsize=12)
        ax2.set_ylabel('Alignment (Cosine Sim, Higher is Better)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        original_print(f"[Progress] Updated trajectory plot to {plot_path}\n")
        
    torch.save = original_torch_save

if __name__ == "__main__":
    run_sweep()
