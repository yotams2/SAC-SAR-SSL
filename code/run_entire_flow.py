
import os
import sys
import time
import subprocess
import argparse

def get_time_stamp():
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    return time.strftime('%m%d%H%M', local_time)

def run_command(command, description):
    print(f"\n{'='*20}")
    print(f"Running Step: {description}")
    print(f"Command: {command}")
    print(f"{'='*20}\n")
    
    try:
        # Using shell=True for easier handling of complex strings, though list format is generally safer.
        # User requested supporting ' and " for description, so simple string concatenation might be tricky if not careful.
        # However, passing sys.argv inputs directly to subprocess call is usually robust.
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running step '{description}': {e}")
        sys.exit(1)

def main():
    # Parse arguments for the flow script itself?
    # Actually, we want to treat unknown arguments as flags for run_pretrain.
    
    parser = argparse.ArgumentParser(description='Run entire SAR-SSL flow', add_help=False)
    parser.add_argument('--time', type=str, default=None, help='Force specific timestamp')
    args, unknown_args = parser.parse_known_args()
    
    if args.time:
        timestamp = args.time
    else:
        timestamp = get_time_stamp()
        print(f"Generated Timestamp: {timestamp}")

    # Reconstruct unknown args string
    # We need to be careful with quoting if arguments contain spaces
    extra_flags = []
    for arg in unknown_args:
        if ' ' in arg:
            extra_flags.append(f'"{arg}"')
        else:
            extra_flags.append(arg)
    extra_flags_str = ' '.join(extra_flags)

    # 1. Pretrain
    # python run_pretrain.py --pretrain --simu-exp --gpu-id 0, <flags> --time <time>
    cmd_1 = f"python run_pretrain.py --pretrain --simu-exp --gpu-id 0, {extra_flags_str} --time {timestamp}"
    run_command(cmd_1, "1. Pretraining")

    # 2. Visualize
    # python visualize_embeddings.py -t <time>
    # Note: Using visualize_embeddings.py (plural) as per file system
    cmd_2 = f"python visualize_embeddings.py -t {timestamp}"
    run_command(cmd_2, "2. Visualize Embeddings")

    # 3. Pretrain Test
    # python run_pretrain.py --test --simu-exp --time <time> --test-mode all --gpu-id 0,
    cmd_3 = f"python run_pretrain.py --test --simu-exp --time {timestamp} --test-mode all --gpu-id 0, {extra_flags_str}"
    run_command(cmd_3, "3. Pretrain Test")

    # 4. Downstream Train
    # python run_downstream.py --ds-train --ds-trainmode finetune --simu-exp --ds-nsimroom 8 --ds-task TDOA --time <time> --gpu-id 0,
    cmd_4 = f"python run_downstream.py --ds-train --ds-trainmode finetune --simu-exp --ds-nsimroom 8 --ds-task TDOA --time {timestamp} --gpu-id 0,"
    run_command(cmd_4, "4. Downstream Train")

    # 5. Downstream Test
    # python run_downstream.py --ds-test --test-mode cal_metric --ds-trainmode finetune --simu-exp --ds-nsimroom 8 --ds-task TDOA --time <time> --gpu-id 0,
    cmd_5 = f"python run_downstream.py --ds-test --test-mode cal_metric --ds-trainmode finetune --simu-exp --ds-nsimroom 8 --ds-task TDOA --time {timestamp} --gpu-id 0,"
    run_command(cmd_5, "5. Downstream Test")

    print(f"\n{'='*20}")
    print(f"Entire flow completed successfully for Experiment Time: {timestamp}")
    print(f"{'='*20}\n")

if __name__ == "__main__":
    main()
