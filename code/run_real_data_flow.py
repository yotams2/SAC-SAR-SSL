import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run downstream real data flow for multiple tasks.")
    parser.add_argument("timestamp", type=str, help="Model timestamp (e.g., 12281023)")
    parser.add_argument("--gpu-id", type=str, default="0,", help="GPU ID to use (default: 0,)")
    parser.add_argument('--ds-trainmode', type=str, default='finetune', help='Downstream train mode (default: finetune)')
    args = parser.parse_args()

    tasks = ['T60', 'C50', 'DRR', 'ABS']

    for task in tasks:
        # Note: Since --ds-test cannot be run on real data (as you noted),
        # this script automates the --ds-train real data command you were running earlier.
        # It will automatically save the test metrics at the end of each run thanks to your recent update!
        cmd = [
            "python", "run_downstream.py", 
            "--ds-train", 
            "--ds-trainmode", args.ds_trainmode, 
            "--ds-real-sim-ratio", "1", "0", 
            "--ds-task", task, 
            "--time", args.timestamp, 
            "--gpu-id", args.gpu_id
        ]
        
        cmd_str = " ".join(cmd)
        print(f"\n{'='*60}")
        print(f"Starting {task} estimation...")
        print(f"Command: {cmd_str}")
        print(f"{'='*60}\n")
        
        try:
            # Run the command and stream the output to the console
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Task {task} failed with exit code {e.returncode}.")
            print("Continuing to the next task...\n")

if __name__ == "__main__":
    main()
