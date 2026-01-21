
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
        "12281023": "Original",
        "12281234": "Original + Feat Sim",
        "12281809": "Autoenc + Feat Sim",
        "12291627": "Autoenc + Feat Sim w. New Normalization",
        "12291841": "Autoenc + Feat Sim w. New Normalization * 4",
        "01041159": "Autoenc + Feat Sim w. Percentile Normalization",
        "01041536": "Autoenc + Feat Sim w. Mean+STD+Clip Normalization",
        "01051314": "Autoenc + Feat Sim w. New Features, temp 0.07",
        "01051625": "Autoenc + Feat Sim w. New Features, temp 0.1",
        "01061055": "Autoenc + Feat Sim w. New Features, temp 0.3",
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
        "01191734": "Autoenc + Feat Sim w. Some Of The Newest Features (decay slope etc), sigma 0.5*sqrt(dim), temp 0.1",
        "01200950": "Autoenc + Feat Sim w. 5 features (coherences, gcc max, tdoa), sigma 1, temp 0.1",
        "01201345": "Autoenc + Feat Sim w. 5 features (coherences, gcc max, tdoa), sigma 1, temp 0.5"
    }
    
    models = args.m
    tasks = args.t
    
    base_path = "/storage/yotam/SAR-SSL/SAR-SSL/exp"
    output_dir = os.path.join(base_path, "summary_tables")
    os.makedirs(output_dir, exist_ok=True)
    
    # Data collection
    # results[model][task] = [list of metrics]
    results = {m: {t: [] for t in tasks} for m in models}
    
    for task in tasks:
        for model_id in models:
            file_path = os.path.join(base_path, task, model_id, "downstream_test_res.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        metrics = [entry['metric_test'] for entry in data]
                        results[model_id][task] = metrics
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                pass # Sent warning previously, keeping it clean for now

    # Helper to find best values per column
    # For min table: best = min(min(metrics))
    # For avg table: best = min(mean(metrics))
    
    best_results_min = {} # {task: min_value}
    best_results_avg = {} # {task: min_mean_value}

    for task in tasks:
        # Check all models for this task
        task_min_vals = []
        task_mean_vals = []
        for model_id in models:
            metrics = results[model_id][task]
            if metrics:
                task_min_vals.append(np.min(metrics))
                task_mean_vals.append(np.mean(metrics))
        
        if task_min_vals:
            best_results_min[task] = min(task_min_vals)
        if task_mean_vals:
            best_results_avg[task] = min(task_mean_vals)

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
    
    def format_val(val, best_val):
        # Format string, bold if matches best
        s = f"{val:.4f}"
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
                metrics = results_dict[model_id][task]
                if metrics:
                    val = np.min(metrics) if "Lowest" in title else np.mean(metrics)
                    best = best_results.get(task)
                    row_data.append(format_val(val, best))
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
                # Actually, standard is " item ".ljust(width)
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
            # standard md col: " :----- " (space, colon, dashes, space) = width chars
            # so 1 + 1 + dashes + 1 = width => dashes = width - 3
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

    lines.extend(generate_aligned_table("Table 1: Best Result (Lowest Metric)", results, best_results_min))
    lines.extend(generate_aligned_table("Table 2: Average Performance (Mean)", results, best_results_avg))
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"Summary tables saved to: {output_path}")

if __name__ == "__main__":
    main()
