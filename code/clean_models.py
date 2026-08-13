import os
import sys
import json
import shutil
import argparse
import re

import find_unused_models

def get_lr_set(is_sim):
    """
    Returns the learning rate set based on experiment type (simulated or real).
    Values are taken from code/opt.py.
    """
    if is_sim:
        return [0.001, 0.0005, 0.0001, 0.00005]
    else:
        return [0.001, 0.0001]

def parse_result_json(json_path):
    """
    Parses the result JSON file to find the winning lr_idx.
    Returns the best_lr_idx, or None if not found/error.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list) and len(data) > 0:
            # check the first entry
            return data[0].get('lr_idx')
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None
    return None

def process_experiment_dir(exp_dir, dry_run, force):
    """
    Processes a single experiment directory to clean up non-winning models.
    """
    print(f"\nProcessing experiment: {exp_dir}")
    
    # 2. Identify Result Files
    modes = ['finetune', 'lineareval']
    types = ['sim', 'real']
    winning_lrs = {} # (mode, dtype) -> winning_lr_value

    for mode in modes:
        for dtype in types:
            is_sim = (dtype == 'sim')
            lr_set = get_lr_set(is_sim)
            
            suffix = "" if is_sim else "_real"
            json_name = f"downstream_test_res_{mode}{suffix}.json"
            json_path = os.path.join(exp_dir, json_name)
            
            # Fallback for finetune if specific file doesn't exist
            if mode == 'finetune' and not os.path.exists(json_path):
                 fallback_name = f"downstream_test_res{suffix}.json"
                 fallback_path = os.path.join(exp_dir, fallback_name)
                 if os.path.exists(fallback_path):
                     json_path = fallback_path
            
            if os.path.exists(json_path):
                lr_idx = parse_result_json(json_path)
                if lr_idx is not None:
                    if lr_idx < len(lr_set):
                        winning_lr = lr_set[lr_idx]
                        winning_lrs[(mode, dtype)] = winning_lr
                        print(f"[{mode} - {dtype}] Found winning LR index {lr_idx} -> LR {winning_lr} (from {os.path.basename(json_path)})")
                    else:
                        print(f"[{mode} - {dtype}] Warning: lr_idx {lr_idx} out of bounds for LR set {lr_set}.")
                else:
                    print(f"[{mode} - {dtype}] Could not extract lr_idx from {os.path.basename(json_path)}.")
            else:
                pass

    if not winning_lrs:
        print("No winning LRs identified. Skipping this directory.")
        return

    # 3. Iterate and Select Files for Deletion
    dirs_to_delete = []
    
    for item in os.listdir(exp_dir):
        item_path = os.path.join(exp_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        # Check if it matches a relevant mode
        matched_mode = None
        for mode in modes:
            if item.startswith(f"{mode}-"):
                matched_mode = mode
                break
        
        if not matched_mode:
            continue
            
        # Determine if it's sim or real
        if '-sim_' in item:
            dtype = 'sim'
        elif '-real_' in item:
            dtype = 'real'
        else:
            print(f"Warning: Could not determine if {item} is sim or real. Skipping.")
            continue
            
        if (matched_mode, dtype) not in winning_lrs:
            continue
            
        winning_lr = winning_lrs[(matched_mode, dtype)]
        
        # Parse LR from directory name
        # Format: finetune-all-mlp-800-0.0001-8-0-spat-sim_R8
        parts = item.split('-')
        dir_lr = None
        
        # Try index 4 first
        try:
            dir_lr = float(parts[4])
        except (IndexError, ValueError):
            # Try combining 4 and 5 for scientific notation (e.g. 5e-05)
            if len(parts) > 5:
                try:
                    combined = f"{parts[4]}-{parts[5]}"
                    dir_lr = float(combined)
                except ValueError:
                    pass

        if dir_lr is None:
            print(f"Warning: Could not parse LR from directory name {item}. Skipping.")
            continue
            
        # Check if this dir matches the winning LR
        if abs(dir_lr - winning_lr) > 1e-9:
            dirs_to_delete.append(item_path)
            
    # 4. Execute Deletion
    print(f"Found {len(dirs_to_delete)} directories to delete.")
    
    for p in sorted(dirs_to_delete):
        if dry_run or not force:
            print(f"[Would Delete] {p}")
        else:
            try:
                shutil.rmtree(p)
                print(f"[Deleted] {p}")
            except Exception as e:
                print(f"[Error Deleting] {p}: {e}")

def clean_unused_models_func(dry_run, force):
    """
    Cleans up unused models by deleting their directories.
    """
    base_dir = "/storage/yotam/SAR-SSL/SAR-SSL"
    exp_path = os.path.join(base_dir, "exp")
    gen_summary_path = os.path.join(base_dir, "code/gen_summary_table.py")

    used_models = find_unused_models.get_used_models(gen_summary_path)
    
    dirs_to_delete = []
    if os.path.exists(exp_path):
        for root, dirs, files in os.walk(exp_path):
            for d in dirs:
                if re.fullmatch(r'\d{8}', d):
                    if d not in used_models:
                        dirs_to_delete.append(os.path.join(root, d))
    
    print(f"Found {len(dirs_to_delete)} unused model directories to delete.")
    
    for p in sorted(dirs_to_delete):
        if dry_run or not force:
            print(f"[Would Delete] {p}")
        else:
            try:
                shutil.rmtree(p)
                print(f"[Deleted] {p}")
            except Exception as e:
                print(f"[Error Deleting] {p}: {e}")

def clean_tar_files_func(target_dir, dry_run, force):
    """
    Cleans up intermediate modelX.tar files that are no longer needed after ensembling.
    """
    if target_dir is None:
        base_dir = "/storage/yotam/SAR-SSL/SAR-SSL"
        target_dir = os.path.join(base_dir, "exp")
        
    files_to_delete = []
    if os.path.exists(target_dir):
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                if re.fullmatch(r'model\d+\.tar', f):
                    files_to_delete.append(os.path.join(root, f))
                    
    print(f"Found {len(files_to_delete)} intermediate tar files to delete in {target_dir}.")
    
    for p in sorted(files_to_delete):
        if dry_run or not force:
            print(f"[Would Delete] {p}")
        else:
            try:
                os.remove(p)
                print(f"[Deleted] {p}")
            except Exception as e:
                print(f"[Error Deleting] {p}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Clean up model weights in experiment directory (or parent directory), keeping only winning LR trials.")
    parser.add_argument("target_dir", type=str, nargs='?', default=None, help="Path to the experiment directory (e.g., .../exp/C50/12281023) or a parent directory. Optional if --clean-unused-models or --clean-tar-files is used.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without actually deleting.")
    parser.add_argument("--force", action="store_true", help="Actually execute the deletion. If not set, defaults to dry-run behavior.")
    parser.add_argument("--clean-unused-models", action="store_true", help="Clean all unused models found by find_unused_models.py")
    parser.add_argument("--clean-tar-files", action="store_true", help="Clean all intermediate modelX.tar files, keeping only latest, best, and ensemble models.")
    
    args = parser.parse_args()
    
    if args.dry_run or not args.force:
        print("--- DRY RUN MODE (No files will be deleted) ---")
        print("Use --force to actually delete files.")
    else:
        print("--- DELETION MODE (Files WILL be deleted) ---")

    if args.clean_unused_models:
        print("--- CLEANING UNUSED MODELS ---")
        clean_unused_models_func(args.dry_run, args.force)
        
        if args.dry_run or not args.force:
            print("\nDone (Dry run). No changes made.")
        else:
            print("\nDone. Cleanup complete.")
        return

    if args.clean_tar_files:
        print("--- CLEANING INTERMEDIATE TAR FILES ---")
        clean_tar_files_func(args.target_dir, args.dry_run, args.force)
        
        if args.dry_run or not args.force:
            print("\nDone (Dry run). No changes made.")
        else:
            print("\nDone. Cleanup complete.")
        return
        
    if args.target_dir is None:
        print("Error: target_dir is required unless --clean-unused-models or --clean-tar-files is specified.")
        parser.print_help()
        sys.exit(1)
        
    target_dir = os.path.abspath(args.target_dir)
    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} does not exist.")
        sys.exit(1)

    # Check if target_dir is a single experiment or a parent
    # Criteria: Checks for 'downstream_test_res*.json' inside
    is_experiment = False
    for fname in os.listdir(target_dir):
        if fname.startswith("downstream_test_res") and fname.endswith(".json"):
            is_experiment = True
            break
            
    if is_experiment:
        process_experiment_dir(target_dir, args.dry_run, args.force)
    else:
        # Assume parent directory
        print(f"Processing parent directory: {target_dir}")
        subdirs = sorted([os.path.join(target_dir, d) for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
        
        for subdir in subdirs:
            # Check if subdir looks like an experiment
            has_json = False
            for fname in os.listdir(subdir):
                if fname.startswith("downstream_test_res") and fname.endswith(".json"):
                    has_json = True
                    break
            
            if has_json:
                process_experiment_dir(subdir, args.dry_run, args.force)
            else:
                print(f"\nSkipping {subdir}: No result JSONs found (not an experiment dir?).")

    if args.dry_run or not args.force:
        print("\nDone (Dry run). No changes made.")
    else:
        print("\nDone. Cleanup complete.")

if __name__ == "__main__":
    main()
