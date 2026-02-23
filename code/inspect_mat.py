import scipy.io
import sys
import numpy as np

def inspect_mat(mat_path):
    try:
        data = scipy.io.loadmat(mat_path)
        print(f"Inspecting {mat_path}")
        
        if 'best_lr_idx' in data:
            idx = data['best_lr_idx'][0][0]
            print(f"best_lr_idx: {idx}")
        else:
            print("best_lr_idx not found in mat file")
            
        if 'lr_set' in data:
            lr_set = data['lr_set']
            # lr_set might be a list or array, print it clearly
            print(f"lr_set: {lr_set}")
            if 'best_lr_idx' in data:
                # Try to index it
                try:
                    # Handle different shapes of lr_set
                    if lr_set.shape[0] == 1:
                        winning_lr = lr_set[0][idx]
                    else:
                        winning_lr = lr_set[idx][0]
                    print(f"winning_lr value from set: {winning_lr}")
                except Exception as e:
                    print(f"Could not extract winning_lr value: {e}")
    except Exception as e:
        print(f"Error loading mat file: {e}")

if __name__ == "__main__":
    inspect_mat(sys.argv[1])
