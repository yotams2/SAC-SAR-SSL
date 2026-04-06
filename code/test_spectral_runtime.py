import torch
import time
import sys
import numpy as np
import librosa
from common.utils_module import spectral_features_from_stft, spatial_features_from_stft

fs = 16000
B_list = [1, 2, 4, 8, 16]
T_frames = 300

print("Testing spectral_features_from_stft runtime scaling with Batch Size (B):")
for B in B_list:
    stft = torch.randn(B, 1, 257, T_frames, dtype=torch.cfloat).cuda()
    
    start = time.time()
    spectral_features_from_stft(stft, fs)
    torch.cuda.synchronize()
    t_spec = time.time() - start
    
    stft_spatial = torch.randn(B, 2, 257, T_frames, dtype=torch.cfloat).cuda()
    start = time.time()
    spatial_features_from_stft(stft_spatial, fs)
    torch.cuda.synchronize()
    t_spat = time.time() - start
    
    print(f"B={B}: spectral={t_spec:.4f}s (normalized: {t_spec/B:.4f}s/sample), spatial={t_spat:.4f}s (normalized: {t_spat/B:.4f}s/sample)")

print("\nTesting librosa.pyin exclusively with length:")
T_list = [50, 100, 200, 400]
for T in T_list:
    # Approximate waveform length for T frames at hop=256
    y = np.random.randn((T - 1) * 256)
    
    start = time.time()
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        sr=fs, 
        hop_length=256,
        fill_na=0.0
    )
    t_pyin = time.time() - start
    print(f"T={T} frames ({len(y)/fs:.2f}s): librosa.pyin={t_pyin:.4f}s")
