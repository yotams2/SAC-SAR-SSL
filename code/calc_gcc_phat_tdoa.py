
import os
import torch
import numpy as np
import dataset as at_dataset
import common.utils_module as at_utils
from tqdm import tqdm

def main():
    # Configuration
    data_dir = '/storage/yotam/SAR-SSL/SAR-SSL/data/MicSig/simu_ds/test/'
    fs = 16000
    T = 4.112 # 1.04 # Duration implied for TDOA task
    
    print(f"Loading data from: {data_dir}")
    
    # Selecting transform to match run_downstream.py usage
    # "selecting = at_dataset.Selecting(select_range=[0, int(T*fs)])"
    selecting = at_dataset.Selecting(select_range=[0, int(T*fs)])
    
    # Initialize dataset
    try:
        dataset = at_dataset.FixMicSigDataset(
            data_dir=data_dir,
            load_anno=True,
            load_dp=False,
            fs=fs,
            dataset_sz=None, # Process all
            transforms=[selecting]
        )
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return

    print(f"Dataset size: {len(dataset)}")
    
    errors = []
    
    # Iterate over dataset
    for i in tqdm(range(len(dataset)), desc="Processing"):
        try:
            data = dataset[i]
            mic_sig = data[0] # (nsample, nch) numpy array
            annos = data[1]
            
            # Ground truth TDOA
            gt_tdoa = annos['TDOA']
            
            # Prepare input for GCC-PHAT
            # compute_spatial_features expects [B, 2, T]
            # mic_sig is [T, 2] -> convert, transpose, unsqueeze
            
            mic_sig_tensor = torch.from_numpy(mic_sig).float()
            mic_sig_tensor = mic_sig_tensor.transpose(0, 1) # [2, T]
            mic_sig_tensor = mic_sig_tensor.unsqueeze(0)    # [1, 2, T]
            
            # Compute features using utils_module
            # Default n_fft=512, hop_length=256 match the project defaults
            features = at_utils.compute_spatial_features(
                x=mic_sig_tensor,
                sample_rate=fs,
                n_fft=512,
                hop_length=256
            )
            
            est_tdoa = -features['tdoa'].item()
            
            # Calculate absolute error
            # TDOA in seconds
            error_seconds = np.abs(est_tdoa - gt_tdoa)
            # Convert to samples for comparison with model training (fs=16000)
            error_samples = error_seconds * fs
            
            errors.append(error_samples)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            
    if len(errors) == 0:
        print("No samples processed successfully.")
        return

    # Calculate statistics
    errors = np.array(errors)
    mae = np.mean(errors)
    std = np.std(errors)
    
    print("-" * 30)
    print(f"Results over {len(errors)} samples (fs={fs}):")
    print(f"MAE: {mae:.4f} [samples]")
    print(f"STD: {std:.4f} [samples]")
    print("-" * 30)

if __name__ == "__main__":
    main()
