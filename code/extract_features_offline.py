import os
import glob
import torch
import soundfile
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import common.utils_module as at_module
import scipy.signal

# Parameters matching run_pretrain.py
FS = 16000
NFFT = 512
WIN_LEN = 512
WIN_SHIFT_RATIO = 0.5

def init_worker():
    global stft_module
    # Initialize the STFT module once per worker process to save overhead
    # We must use CPU since multiprocessing with CUDA is tricky
    device = torch.device('cpu')
    stft_module = at_module.STFT(
        win_len=WIN_LEN, 
        win_shift_ratio=WIN_SHIFT_RATIO, 
        nfft=NFFT,
        win='hann'
    ).to(device)

def process_file(wav_path):
    try:
        # Load audio
        mic_sig, fs = soundfile.read(wav_path)
        
        # Resample if necessary
        if fs != FS:
            mic_sig = scipy.signal.resample_poly(mic_sig, FS, fs)
            
        mic_sig = mic_sig.astype(np.float32)
        
        # Ensure it has the shape (1, nsample, nch) expected by STFT
        if mic_sig.ndim == 1:
            mic_sig = mic_sig[:, np.newaxis]
        
        # Create tensor and shape to (batch=1, nsample, nch)
        sig_tensor = torch.tensor(mic_sig).unsqueeze(0)
        
        # Compute STFT: output shape (nbatch, nf, nt, nch)
        stft_out = stft_module(sig_tensor)
        
        # Permute to (nbatch, nch, nf, nt) as expected by spectral_features_from_stft
        stft_out = stft_out.permute(0, 3, 1, 2)
        
        # Extract features
        # Note: at_module.spectral_features_from_stft expects complex tensor (B, C, F, T) or (B, F, T)
        feats_dict = at_module.spectral_features_from_stft(
            stft_out, 
            sample_rate=FS,
            hop_length=int(WIN_LEN * WIN_SHIFT_RATIO),
            n_fft=NFFT
        )
        
        # Strip batch dimension (it was B=1)
        out_dict = {
            'feature_matrix': feats_dict['feature_matrix'][0].clone(),
            'voiced_prob': feats_dict['voiced_prob'][0].clone()
        }
        
        # Save as .pt next to .wav
        out_path = str(wav_path).replace('.wav', '_spec_feats.pt')
        torch.save(out_dict, out_path)
        return True
    
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline feature extraction for SAR-SSL")
    parser.add_argument("--data_dirs", nargs='+', required=True, help="List of dataset directories to process")
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count()-2), help="Number of CPU workers")
    args = parser.parse_args()

    wav_files = []
    for d in args.data_dirs:
        # Recursively find all wav files
        found = list(Path(d).rglob('*.wav'))
        # Exclude _dp.wav (direct path files used for GT)
        found = [f for f in found if not str(f).endswith('_dp.wav')]
        wav_files.extend(found)
        
    print(f"Found {len(wav_files)} total .wav files (excluding _dp.wav).")
    
    # Filter files that are already processed if we want resuming capability
    unprocessed = []
    for w in wav_files:
        pt_path = str(w).replace('.wav', '_spec_feats.pt')
        if not os.path.exists(pt_path):
            unprocessed.append(w)
            
    print(f"Processing {len(unprocessed)} unfinished files with {args.num_workers} workers...")
    
    if len(unprocessed) == 0:
        return
        
    with Pool(processes=args.num_workers, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, unprocessed), total=len(unprocessed)))
        
    success = sum(results)
    print(f"Finished! Successfully extracted features for {success}/{len(unprocessed)} files.")

if __name__ == '__main__':
    main()
