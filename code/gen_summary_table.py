
import argparse
import json
import os
import numpy as np
import sys
import shlex
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Generate summary tables for downstream tasks.")
    parser.add_argument('--m', '--models', nargs='+', required=True, help="List of model timestamps (e.g., 12281023 12281234)")
    parser.add_argument('--t', '--tasks', nargs='+', default=['TDOA', 'T60', 'ABS', 'DRR'], help="List of tasks (e.g., TDOA DRR)")
    
    args = parser.parse_args()
    
    # Predefined names for specific timestamps
    model_name_map = {
        "12281023": "CCSR Run 1",
        "12281234": "CCSR + Feat Sim",
        "01061410": "Autoenc + Pseudo-Hierarchical Feat Sim",
        "01081544": "Autoenc + Feat Sim wo. ILD and HF_LF, temp 0.1",
        "01111059": "Autoenc + Pseudo-Hierarchical Feat Sim wo. ILD and HF_LF, doa_temp 0.3, room_temp 0.07",
        "01111117": "Autoenc + Pseudo-Hierarchical Feat Sim, mono spectral,  wo. ILD and HF_LF, doa_temp 0.3, room_temp 0.07",
        "01122345": "Autoenc + Feat Sim w. Newest Features (decay slope etc), temp 0.1",
        "01131440": "Autoenc + Feat Sim w. Newest Features (decay slope etc), Hierarchical, sigma 1/1, temp 0.3/0.07",
        "01141100": "Autoenc + Feat Sim w. Newest Features (decay slope etc), Hierarchical, sigma 0.4/1.6, temp 0.3/0.07",
        "01151136": "Autoenc + Feat Sim w. Some Of The Newest Features (decay slope etc), Hierarchical, sigma 0.4/0.5*sqrt(dim), temp 0.1/0.07",
        "01161203": "Autoenc + Feat Sim w. Some Of The Newest Features (decay slope etc), Hierarchical, sigma 0.4/0.5*sqrt(dim), temp 0.4/0.2",
        "01190956": "Autoenc + Feat Sim w. Some Of The Newest Features (decay slope etc), OBT, Hierarchical, sigma 0.4/0.5*sqrt(dim), temp 0.4/0.2",
        "01231932": "Autoenc ONLY",
        "01261622": "CCSR + Feat Sim w. 5 features (coherences, gcc max, tdoa), BN, sigma 1, temp 0.5",
        "01271707": "CCSR + New Version of Feat Sim w. 5 features (coherences, gcc max, tdoa), BN, sigma 1, temp 0.07",
        "01272122": "CCSR + New Version of Feat Sim w. minus sim_max, 5 features (coherences, gcc max, tdoa), BN, sigma 1, temp 0.07",
        "02010901": "CCSR + FeatSim",
        "02021006": "AE + FeatSim",
        "02041034": "CCSR Run 2",
        "02041438": "No Pretrain",
        "02121359": "Only FeatSim"
    }
    
    models = args.m
    tasks = args.t
    
    base_path = "/storage/yotam/SAR-SSL/SAR-SSL/exp"
    output_dir = os.path.join(base_path, "summary_tables")
    os.makedirs(output_dir, exist_ok=True)
    
    # Data collection
    # results[model][task] = [list of (metric, std) tuples]
    results_finetune = {m: {t: [] for t in tasks} for m in models}
    results_linear = {m: {t: [] for t in tasks} for m in models}
    
    for task in tasks:
        for model_id in models:
            # 1. Finetune or scrathcLOW
            for file_name in ["downstream_test_res_finetune.json", "downstream_test_res.json", "downstream_test_res_scratchLOW.json"]:
                file_path = os.path.join(base_path, task, model_id, file_name)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            metrics = []
                            for entry in data:
                                val = entry['metric_test']
                                std = entry.get('metric_test_std', None)
                                metrics.append((val, std))
                            results_finetune[model_id][task] = metrics
                            break
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
            
            # 2. Linear Eval
            file_name_linear = "downstream_test_res_lineareval.json"
            file_path_linear = os.path.join(base_path, task, model_id, file_name_linear)
            if os.path.exists(file_path_linear):
                 try:
                    with open(file_path_linear, 'r') as f:
                        data = json.load(f)
                        metrics = []
                        for entry in data:
                            val = entry['metric_test']
                            std = entry.get('metric_test_std', None)
                            metrics.append((val, std))
                        results_linear[model_id][task] = metrics
                 except Exception as e:
                    print(f"Error reading {file_path_linear}: {e}")

    # Helper to calculate stats
    def calculate_best_stats(results_dict):
        best_min = {}
        best_avg = {}
        for task in tasks:
            task_min_vals = []
            task_mean_vals = []
            for model_id in models:
                metrics_list = results_dict[model_id][task]
                if metrics_list:
                    vals = [m[0] for m in metrics_list]
                    task_min_vals.append(np.min(vals))
                    task_mean_vals.append(np.mean(vals))
            
            if task_min_vals:
                best_min[task] = min(task_min_vals)
            if task_mean_vals:
                best_avg[task] = min(task_mean_vals)
        return best_min, best_avg

    best_ft_min, best_ft_avg = calculate_best_stats(results_finetune)
    best_lin_min, best_lin_avg = calculate_best_stats(results_linear)

    # Generate Output Content
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"summary_table_{timestamp}.md"
    output_path = os.path.join(output_dir, output_filename)
    
    lines = []
    lines.append(f"# Summary Results - {timestamp}")
    lines.append("")

    # Record the command used
    command_used = f"python {sys.argv[0]} " + " ".join(shlex.quote(arg) for arg in sys.argv[1:])
    lines.append("## Command Used")
    lines.append(f"```bash\n{command_used}\n```")
    lines.append("")
    
    def format_val(val_tuple, best_val):
        # Format string, bold if matches best
        # val_tuple is (val, std)
        if isinstance(val_tuple, tuple):
            val, std = val_tuple
        else:
            val = val_tuple
            std = None
            
        s = f"{val:.4f}"
        if std is not None:
            s += f" ± {std:.4f}"
            
        if best_val is not None and np.isclose(val, best_val, atol=1e-6):
            return f"**{s}**"
        return s

    def generate_aligned_table(title, results_dict, best_results):
        table_lines = []
        table_lines.append(f"## {title}")
        
        # 1. Collect all rows including header to determine widths
        headers = ["Model"] + tasks
        rows = [headers]
        
        for model_id in models:
            display_name = model_name_map.get(model_id, f"Model {model_id}")
            row_data = [display_name]
            for task in tasks:
                metrics_list = results_dict[model_id][task] # list of (val, std)
                if metrics_list:
                    vals = [m[0] for m in metrics_list]
                    stds = [m[1] for m in metrics_list]
                    # stds might contain None if some entries didn't have it, but we handle None in format_val
                    # actually we need mean of stds for mean table
                    
                    if "Lowest" in title:
                        # Find index of min value
                        idx = np.argmin(vals)
                        val = vals[idx]
                        std = stds[idx]
                        display_tuple = (val, std)
                        best = best_results.get(task)
                        row_data.append(format_val(display_tuple, best))
                    else:
                        # Mean
                        val = np.mean(vals)
                        # valid_stds
                        valid_stds = [s for s in stds if s is not None]
                        if valid_stds:
                            mean_std = np.mean(valid_stds)
                        else:
                            mean_std = None
                        
                        display_tuple = (val, mean_std)
                        best = best_results.get(task)
                        row_data.append(format_val(display_tuple, best))
                         
                else:
                    row_data.append("N/A")
            rows.append(row_data)
            
        # 2. Calculate max width for each column
        # Transpose rows to iterate by column
        num_cols = len(headers)
        col_widths = []
        for i in range(num_cols):
            # +2 for padding (space before and after)
            max_len = max(len(row[i]) for row in rows) 
            col_widths.append(max_len + 2) 

        # 3. Format lines
        def format_row(row_items):
            # | item | item |
            line = "|"
            for i, item in enumerate(row_items):
                # Padding: space + item + space, then fill rest
                content = f" {item} "
                line += content.ljust(col_widths[i]) + "|"
            return line

        # Header
        table_lines.append(format_row(rows[0]))
        
        # Separator
        # | :--- | :--- | ...
        sep_line = "|"
        for width in col_widths:
            # width includes the 2 spaces padding from calculation
            dashes = "-" * (width - 3)
            sep = f" :{dashes} |"
            sep_line += sep
        table_lines.append(sep_line)
        
        # Data rows
        for row_data in rows[1:]:
             table_lines.append(format_row(row_data))
             
        table_lines.append("")
        table_lines.append("")
        return table_lines

    lines.extend(generate_aligned_table("Table 1: Best Result (Finetune - Lowest Metric)", results_finetune, best_ft_min))
    lines.extend(generate_aligned_table("Table 2: Average Performance (Finetune - Mean)", results_finetune, best_ft_avg))
    lines.extend(generate_aligned_table("Table 3: Best Result (Linear Eval - Lowest Metric)", results_linear, best_lin_min))
    lines.extend(generate_aligned_table("Table 4: Average Performance (Linear Eval - Mean)", results_linear, best_lin_avg))
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"Summary tables saved to: {output_path}")

if __name__ == "__main__":
    main()
