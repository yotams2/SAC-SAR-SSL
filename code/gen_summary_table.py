
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
        "12281023": "CCSR",
        "12281234": "CCSR + Feat Sim",
        "01231932": "Autoenc ONLY",
        "02010901": "CCSR + FeatSim",
        "02021006": "AE + FeatSim",
        "02041034": "CCSR Run 2",
        "02041438": "No Pretrain",
        "02121359": "Only FeatSim",
        "02241046": "Retry Only FeatSim",
        "02251839": "Spatial FeatSim + Spectral FeatSim, sigma 1/1, temp 0.3/0.3",
        "03041251": "Hierarchical Feat Sim, 2 shared layers",
        "03051222": "Hierarchical Feat Sim, 2 shared layers, OBT",
        "03081455": "Hierarchical Feat Sim, 1 shared layer, spat_dembed 128",
        "03101553": "Hierarchical Feat Sim",
        "03151553": "Only FeatSim + Differetintiable Feature Weighting",
        "03161349": "Only FeatSim + Differetintiable Feature Weighting with L1 applied on sigmoid",
        "03181907": "Only FeatSim + Differetintiable Feature Weighting with softmax and tau=3",
        "03191118": "Only FeatSim + Differetintiable Feature Weighting with softmax and tau=6",
        "03231109": "Only FeatSim + Differetintiable Feature Weighting with anchored TDOA",
        "03251037": "Only FeatSim + Fixed feature weights from run with fixed TDOA",
        "03251936": "Hierarchical Feat Sim, 0 shared layers, adv. separation",
        "03301114": "Hierarchical Feat Sim, 0 shared layers, adv. separation",
        "03301429": "Hierarchical Feat Sim split spatial encoder",
        "04061208": "Factorized FeatSim",
        "04061450": "Factorized FeatSim + PCGrad",
        "04151436": "FeatSim with GT features",
        "04271542": "Hierarchical Feat Sim, stop grad",
        "04291449": "Hierarchical Feat Sim, channel attention",
        "04291721": "Only FeatSim, 40,000 samples only",
        "05061023": "Hierarchical Feat Sim, 0 shared layers, GT features",
        "05111531": "Hierarchical Feat Sim, stop grad on DOA branch",
        "06241207": "Only FeatSim + Cross Attention Layer with global family medians sigma",
        "06241636": "Only FeatSim + Cross Attention Layer with global family medians sigma and family-specific-bn",
        "07062141": "CCSR + FeatSim with tau=0.2 sigma 0.295",
        "07082312": "CCSR + FeatSim with tau=0.5 sigma 0.613",
        "07172334": "CCSR + FeatSim with tau=0.4 sigma 0.62",
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
    results_real = {m: {t: [] for t in tasks} for m in models}
    
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

            # 3. Real Data
            for file_name in ["downstream_test_res_finetune_real.json", "downstream_test_res_real.json", "downstream_test_res_scratchLOW_real.json"]:
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
                            results_real[model_id][task] = metrics
                            break
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

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
    best_real_min, best_real_avg = calculate_best_stats(results_real)

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
    lines.extend(generate_aligned_table("Table 5: Best Result (Real Data - Lowest Metric)", results_real, best_real_min))
    lines.extend(generate_aligned_table("Table 6: Average Performance (Real Data - Mean)", results_real, best_real_avg))
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
        
    # --- Generate LLM friendly JSON ---
    def generate_json_data(results_dict):
        table_data = {}
        for model_id in models:
            display_name = model_name_map.get(model_id, f"Model {model_id}")
            model_res = {}
            for task in tasks:
                metrics_list = results_dict[model_id][task]
                if metrics_list:
                    vals = [m[0] for m in metrics_list]
                    stds = [m[1] for m in metrics_list]
                    valid_stds = [s for s in stds if s is not None]
                    
                    idx = int(np.argmin(vals))
                    min_val = float(vals[idx])
                    min_std = float(stds[idx]) if stds[idx] is not None else None
                    
                    mean_val = float(np.mean(vals))
                    mean_std = float(np.mean(valid_stds)) if valid_stds else None
                    
                    model_res[task] = {
                        "lowest_metric": min_val,
                        "lowest_std": min_std,
                        "mean_metric": mean_val,
                        "mean_std": mean_std,
                        "raw_trials": [{"val": float(v), "std": float(s) if s is not None else None} for v, s in zip(vals, stds)]
                    }
                else:
                    model_res[task] = None
            table_data[display_name] = model_res
        return table_data

    def to_float_dict(d):
        return {k: float(v) for k, v in d.items()}

    terms_definitions = {
        "CCSR": "Cross Channel Signal Reconstruction. A baseline method from Yang and Li's paper 'Self-Supervised Learning of Spatial Acoustic Representation with Cross-Channel Signal Reconstruction and Multi-Channel Conformer'.",
        "AE (Autoencoder)": "A baseline model where the input is not masked, and the loss is Mean Squared Error (MSE) computed on the fully reconstructed STFT tensor.",
        "FeatSim (SAC)": "Feature Similarity Loss, also referred to as Soft Acoustic Contrastive (SAC) loss. A novel self-supervised method to learn spatial acoustic representations.",
        "No Pretrain": "The encoder is trained from scratch purely on downstream labeled data, without any self-supervised pretraining.",
        "Hierarchical FeatSim": "A hierarchical architecture where the spatial encoder (a shared CNN followed by Conformer layers) branches into a DOA branch (for TDOA) and a Room branch (for T60/C50/ABS/DRR). '0/1/2 shared layers' denotes how many Conformer layers are shared before the split. 'temp/sigma' denote parameters for the FeatSim loss.",
        "OBT": "Orthogonal Barlow Twins loss. A penalty applied to the correlation between DOA and Room embeddings to disentangle their latent spaces, encouraging branch-specific features.",
        "spat_dembed": "The dimensionality of the spatial encoder's output embedding.",
        "Differential Feature Weighting": "Also known as Learnable SAC Targets. Introduces learnable scalar weights for specific acoustic features (TDOA, GCC-max, Coherence bands) prior to computing pairwise distances for the SAC loss.",
        "adv. separation": "Adversarial separation training applied between the DOA and Room branches to further disentangle the representations.",
        "split spatial encoder": "An architecture utilizing two completely distinct spatial encoders for DOA and Room information, with no shared parameters.",
        "Factorized FeatSim": "Computes separate soft contrastive losses for each specific acoustic feature, rather than fusing them into a single aggregate metric.",
        "PCGrad": "Projecting Conflicting Gradients. A technique used to physically alter gradient vectors to prevent interference when the DOA and Room branches provide conflicting updates to the shared CNN weights.",
        "GT features": "Ground Truth features. Uses exact labels from simulated RIRs as targets instead of estimated features. This provides an upper bound on performance but is no longer purely self-supervised.",
        "stop grad": "Inserts a stop-gradient (.detach()) operation at the input to the Room branch to prevent gradients from flowing backward into the shared encoder layers.",
        "channel attention": "Inserts Channel Attention Gates at the branch split point. The shared CNN output feeds into a doa_gate and a room_gate, allowing each branch to dynamically mask irrelevant channels before gradients conflict in the shared layers.",
        "Pretraining Samples": "All models used 50,000 samples during the pretraining stage, unless stated otherwise."
    }

    llm_data = {
        "timestamp": timestamp,
        "terms_definitions": terms_definitions,
        "finetune_results": generate_json_data(results_finetune),
        "linear_eval_results": generate_json_data(results_linear),
        "real_data_results": generate_json_data(results_real),
        "best_baseline_lowest": {
            "finetune": to_float_dict(best_ft_min),
            "linear_eval": to_float_dict(best_lin_min),
            "real_data": to_float_dict(best_real_min)
        },
        "best_baseline_mean": {
            "finetune": to_float_dict(best_ft_avg),
            "linear_eval": to_float_dict(best_lin_avg),
            "real_data": to_float_dict(best_real_avg)
        }
    }

    json_output_dir = os.path.join(base_path, "llm_summary")
    os.makedirs(json_output_dir, exist_ok=True)
    json_output_filename = f"summary_{timestamp}.json"
    json_output_path = os.path.join(json_output_dir, json_output_filename)

    with open(json_output_path, 'w') as f:
        json.dump(llm_data, f, indent=4)

    print(f"Summary tables saved to: {output_path}")
    print(f"LLM JSON summary saved to: {json_output_path}")

if __name__ == "__main__":
    main()
