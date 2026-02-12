import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import dataset as at_dataset
import learner as at_learner
from common.utils_module import spatial_features_from_stft
from common.utils import set_seed

def count_wav_files(directory_or_dict):
    """
    Recursively counts .wav files in a directory or dict of directories.
    Excludes *_dp.wav files.
    """
    count = 0
    if isinstance(directory_or_dict, dict):
        for key, val in directory_or_dict.items():
            count += count_wav_files(val)
    elif isinstance(directory_or_dict, list):
        for d in directory_or_dict:
             count += count_wav_files(d)
    elif isinstance(directory_or_dict, str):
        path = Path(directory_or_dict)
        if path.exists():
            # Logic from dataset.py:FixMicSigDataset
            files = list(path.rglob('*.wav'))
            dp_files = list(path.rglob('*_dp.wav'))
            valid_files = [f for f in files if f not in dp_files]
            count = len(valid_files)
    return count

def calculate_features_benchmark(dataloader, device, learner, fs):
    print("Benchmarking global feature statistics calculation on entire dataset...")
    
    start_time = time.time()
    total_batches = len(dataloader)
    
    # Helper for stats (placeholder to ensure computation happens)
    def get_stats(tensor_list):
        if not tensor_list: return 0, 0
        all_vals = torch.cat(tensor_list)
        mean = torch.mean(all_vals).item()
        std = torch.std(all_vals).item()
        return mean, std

    # Storage for verifying computation (optional, effectively doing the work)
    ild_values = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Remove the early break condition
            # if i >= num_batches_for_std:
            #     break
            
            mic_sig_batch = data[0].to(device)
            # Calculate STFT using learner's stft module
            # learner.stft returns [B, F, T, C]
            stft = learner.stft(mic_sig_batch) 
            stft = stft.permute(0, 3, 1, 2) # [B, C, F, T]
            
            # This is the heavy lifting function we want to benchmark
            feats = spatial_features_from_stft(stft, fs, max_ipd_freq=1500.0)
            
            # Store one feature to prevent compiler optimizations eliding work (unlikely in eager pytorch but good practice)
            ild_values.append(feats['ild_mean'].cpu())

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_batch = total_time / total_batches if total_batches > 0 else 0
    
    print("\n" + "="*40)
    print(f"Benchmark Results:")
    print(f"Total Batches Processed: {total_batches}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Batch: {avg_time_per_batch:.4f} seconds")
    print("="*40 + "\n")
    
    return total_time

def main():
    # Settings
    data_dir = '/storage/yotam/SAR-SSL/SAR-SSL/data/MicSig/simu/pretrain'
    fs = 16000
    batch_size = 128 # Default from run_pretrain.py args.bs
    workers = 8
    seed = 1
    
    # Setup Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    set_seed(seed)
    
    # Data Setup
    dataset_sz = 50000
    # print(f"Counting files in {data_dir}...")
    # dataset_sz = count_wav_files(data_dir)
    # print(f"Found {dataset_sz} files.")
    
    dataset_pretrain = at_dataset.FixMicSigDataset( 
        data_dir = data_dir, 
        load_anno=False, 
        load_dp=False,
        fs = fs,
        dataset_sz=dataset_sz,
        transforms = None,
        sort=False # Default to False if not using CL
    )
    
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset_pretrain, 
        batch_size=batch_size, 
        shuffle=False, # No need to shuffle for benchmark
        **kwargs
    )
    
    # Learner Setup (Need dummy model for STFTLearner)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)
        def forward(self, x): return x
        
    dummy_model = DummyModel()
    
    # STFT parameters from run_pretrain.py
    win_len = 512
    nfft = 512
    win_shift_ratio = 0.5
    fre_used_ratio = 1
    
    learner = at_learner.STFTLearner(
        dummy_model, 
        win_len=win_len, 
        win_shift_ratio=win_shift_ratio, 
        nfft=nfft, 
        fre_used_ratio=fre_used_ratio, 
        fs=fs, 
        task=None, 
        ch_mode='M'
    )
    
    if use_cuda:
        learner.cuda()
    else:
        learner.cpu()
        
    # Run Benchmark
    calculate_features_benchmark(dataloader, device, learner, fs)

if __name__ == "__main__":
    main()
