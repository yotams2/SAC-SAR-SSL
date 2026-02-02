
import os
import sys
import time
import subprocess
import json
import glob
import random
import numpy as np

def get_time_stamp():
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    # Added seconds to reduce collision probability for parallel runs
    return time.strftime('%m%d%H%M%S', local_time)

import traceback

def run_command(command, description, log_dir=None):
    print(f"\n{'='*20}")
    print(f"Running Step: {description}")
    print(f"Command: {command}")
    print(f"{'='*20}\n")
    
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running step '{description}': {e}")
        
        error_file_path = "error.log"
        if log_dir:
            error_file_path = os.path.join(log_dir, "error.log")
            
        with open(error_file_path, "a") as f:
            f.write(f"\n{'='*20}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error running step '{description}':\n")
            f.write(f"Command: {command}\n")
            f.write(f"Error: {e}\n")
            f.write("-" * 20 + "\n")
            f.write(traceback.format_exc())
        raise e

def load_and_aggregate_results(exp_dir_full, tasks):
    """
    Scans the experiment directory for all results, recovering sigma/temp/timestamp
    from pretrain configs and downstream result files.
    """
    all_results = {}
    
    # 1. Find all pretrain configs to identify runs and their parameters
    # Path: exp_dir_full/pretrain/<timestamp>/config.json
    config_pattern = os.path.join(exp_dir_full, "pretrain", "*", "config.json")
    config_files = glob.glob(config_pattern)
    
    print(f"Found {len(config_files)} existing runs (config files).")

    for config_path in config_files:
        # Timestamp is the parent directory name
        timestamp = os.path.basename(os.path.dirname(config_path))
        
        try:
            with open(config_path, 'r') as f:
                # config saves [args_dict, dirs, ...]
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    args_dict = data[0]
                    sigma = args_dict.get('feature_sim_sigma')
                    temp = args_dict.get('feature_sim_temperature')
                    
                    if sigma is None or temp is None:
                        # Fallback or skip if parameters not found
                        continue
                    
                    key = (sigma, temp, timestamp)
                    if key not in all_results:
                        all_results[key] = {}
                    
                    # 2. Look for downstream results for this timestamp
                    for task in tasks:
                        # Path: exp_dir_full/<task>/<timestamp>/downstream_test_res.json
                        res_json_path = os.path.join(exp_dir_full, task, timestamp, "downstream_test_res.json")
                        
                        if os.path.exists(res_json_path):
                            try:
                                with open(res_json_path, 'r') as rf:
                                    res_data = json.load(rf)
                                    # res_data is list of dicts
                                    metrics = [entry['metric_test'] for entry in res_data]
                                    losses = [entry['loss_test'] for entry in res_data]
                                    
                                    all_results[key][task] = {
                                        'metrics': metrics,
                                        'losses': losses
                                    }
                            except Exception as e:
                                print(f"Error reading results for {task} at {timestamp}: {e}")
        except Exception as e:
            print(f"Error reading config at {timestamp}: {e}")
            
    return all_results

def generate_summary_table(all_results, summary_file, tasks, script_start_time):
    # all_results structure:
    # {
    #   (sigma, temp, timestamp): {
    #       task: {
    #           'metrics': [m1, m2, m3, m4],
    #           'losses': [l1, l2, l3, l4]
    #       }
    #   }
    # }
    
    lines = []
    lines.append(f"# Grid Search Summary (Script Started: {script_start_time})")
    lines.append(f"Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Table 1: Best Result (Lowest Metric)
    lines.append("## Table 1: Best Result (Lowest Metric across seeds)")
    headers = ["Model (Sigma, Temp)", "Time"] + tasks
    
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    lines.append(header_line)
    lines.append(sep_line)
    
    sorted_keys = sorted(all_results.keys(), key=lambda x: (x[0], x[1])) # Sort by Sigma, Temp
    
    for (sigma, temp, timestamp) in sorted_keys:
        model_name = f"Sigma={sigma}, Temp={temp}"
        row = [model_name, timestamp]
        
        for task in tasks:
            if task in all_results[(sigma, temp, timestamp)]:
                metrics = all_results[(sigma, temp, timestamp)][task]['metrics']
                if metrics:
                    # Best result for this run = min(metrics)
                    best_val = np.min(metrics)
                    row.append(f"{best_val:.4f}")
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("")

    # Table 2: Average Performance (Mean)
    lines.append("## Table 2: Average Performance (Mean across seeds)")
    lines.append(header_line)
    lines.append(sep_line)
    
    for (sigma, temp, timestamp) in sorted_keys:
        model_name = f"Sigma={sigma}, Temp={temp}"
        row = [model_name, timestamp]
        
        for task in tasks:
            if task in all_results[(sigma, temp, timestamp)]:
                metrics = all_results[(sigma, temp, timestamp)][task]['metrics']
                if metrics:
                    # Average result for this run = mean(metrics)
                    mean_val = np.mean(metrics)
                    row.append(f"{mean_val:.4f}")
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        
        lines.append("| " + " | ".join(row) + " |")

    with open(summary_file, 'w') as f:
        f.write("\n".join(lines))
    print(f"Updated summary table at {summary_file}")


def get_existing_experiments(exp_dir_full):
    """
    Scans the experiment directory for existing pretrain configs to identify
    generated sigma/temp combinations.
    """
    existing_params = set()
    config_pattern = os.path.join(exp_dir_full, "pretrain", "*", "config.json")
    config_files = glob.glob(config_pattern)
    
    for config_path in config_files:
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    args_dict = data[0]
                    sigma = args_dict.get('feature_sim_sigma')
                    temp = args_dict.get('feature_sim_temperature')
                    if sigma is not None and temp is not None:
                        existing_params.add((sigma, temp))
        except Exception:
            pass
    return existing_params

def main():
    sigmas = [0.1, 0.3, 0.5, 1.0]
    temps = [0.07, 0.1, 0.3, 0.5]
    tasks = ["TDOA", "T60"]
    
    script_start_time = get_time_stamp()
    exp_dir = "exp_grid_search"
    work_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    exp_dir_full = os.path.join(work_dir, 'SAR-SSL', exp_dir)
    
    os.makedirs(exp_dir_full, exist_ok=True)
    
    summary_file = os.path.join(exp_dir_full, f"summary_table_{script_start_time}.md")
    print(f"Summary table will be saved to: {summary_file}")
    
    # Grid Search Loop
    # We loop through all combinations, but check freshness at each step
    # to support parallel execution where another instance might have picked it up.
    for sigma in sigmas:
        for temp in temps:
            # Check if this combination already exists on disk
            existing_experiments = get_existing_experiments(exp_dir_full)
            if (sigma, temp) in existing_experiments:
                print(f"Skipping Sigma={sigma}, Temp={temp} (Already exists)")
                continue

            run_timestamp = get_time_stamp()
            print(f"\nStarting Grid Point: Sigma={sigma}, Temp={temp}, Time={run_timestamp}")

            description = f"sigma = {sigma}, temp = {temp}"
            
            # 1. Pretrain
            cmd_pretrain = (
                f"python run_pretrain.py --pretrain --simu-exp --gpu-id 0, "
                f'--d "{description}" '
                f"--num-train-samples 50000 --nepoch 10 --feature-sim-loss --no-wandb "
                f"--exp-dir {exp_dir} "
                f"--feature-sim-sigma {sigma} --feature-sim-temperature {temp} "
                f"--time {run_timestamp}"
            )
            run_command(cmd_pretrain, f"Pretrain (Sig={sigma}, Temp={temp})", log_dir=exp_dir_full)
        
            for task in tasks:
                # 2. Downstream Train
                cmd_ds_train = (
                    f"python run_downstream.py --ds-train --ds-trainmode finetune --simu-exp "
                    f"--ds-nsimroom 8 --ds-task {task} "
                    f"--time {run_timestamp} --gpu-id 0, "
                    f"--exp-dir {exp_dir}"
                )
                run_command(cmd_ds_train, f"Downstream Train {task}", log_dir=exp_dir_full)

                # 3. Downstream Test
                cmd_ds_test = (
                    f"python run_downstream.py --ds-test --test-mode cal_metric --ds-trainmode finetune --simu-exp "
                    f"--ds-nsimroom 8 --ds-task {task} "
                    f"--time {run_timestamp} --gpu-id 0, "
                    f"--exp-dir {exp_dir}"
                )
                run_command(cmd_ds_test, f"Downstream Test {task}", log_dir=exp_dir_full)
                
            # 4. Result Aggregation (Dynamic)
            # Scan disk for all results, including those from other parallel instances
            print("aggregating all results...")
            all_results = load_and_aggregate_results(exp_dir_full, tasks)

            # Update Summary Table after each task
            generate_summary_table(all_results, summary_file, tasks, script_start_time)

if __name__ == "__main__":
    main()
