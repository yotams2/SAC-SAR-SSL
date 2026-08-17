"""
Median Heuristic Bandwidth Computation for Soft Acoustic Contrastive (SAC) Loss.

This module provides:
1. `compute_median_heuristic_sigma`: a clean, self-contained utility function to calculate
   the median-heuristic bandwidth for Gaussian kernels.
2. A standalone script to extract actual normalized spatial feature vectors from the pretraining dataset,
   compute median-heuristic sigma for D=13 (hybrid feature set) vs D=5, and compare against old tuned values.
"""

import os
import sys
import argparse
import numpy as np
import scipy.spatial.distance as dist
import torch

# Ensure code directory is in sys.path
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from opt import opt_pretrain
import dataset as at_dataset
from torch.utils.data import DataLoader
from common.utils_module import STFT, spatial_features_from_stft

# D=13 Hybrid Feature Set (11 PFA-selected features + TDOA & GCC-PHAT anchors)
D13_HYBRID_KEYS = [
    'coherence_mean',
    'ipd_cos_mean',
    'ipd_sin_mean',
    'early_late_ratio_median',
    'decay_slope_median',
    'hf_lf_ratio',
    'ild_mean',
    'mod_low_high_ratio',
    'coherence_low',
    'coherence_var',
    'decay_slope_p90',
    'tdoa',
    'gcc_phat_max',
]

# D=5 Previous Production Feature Set
D5_ORIGINAL_KEYS = [
    'coherence_low',
    'coherence_mid',
    'coherence_high',
    'gcc_phat_max',
    'tdoa',
]

PREVIOUS_TUNED_SIGMA_D5 = 0.613  # Alignment/uniformity tuned sigma for D=5


def compute_median_heuristic_sigma(feature_matrix, subsample_size=2000, seed=42):
    """
    Compute the median-heuristic bandwidth for a Gaussian kernel over feature_matrix.

    feature_matrix: array of shape (N, D), the actual candidate feature vectors
                     (the current D=13 hybrid feature set, already normalized
                     to [-1, 1] as in the existing production pipeline).
    subsample_size: if N is large, subsample this many rows (without replacement)
                     before computing pairwise distances, to keep this tractable
                     (pairwise distance computation is O(N^2)).
    seed: random seed for the subsample, for reproducibility.

    Returns: a single float, sigma, set to the median of all pairwise Euclidean
             distances (off-diagonal only, i.e. exclude each point's distance to itself)
             among the (sub)sampled feature vectors.
    """
    if hasattr(feature_matrix, 'detach'):
        feature_matrix = feature_matrix.detach().cpu().numpy()
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)

    N = feature_matrix.shape[0]
    if N > subsample_size:
        rng = np.random.RandomState(seed)
        indices = rng.choice(N, size=subsample_size, replace=False)
        subsampled_matrix = feature_matrix[indices]
    else:
        subsampled_matrix = feature_matrix

    # scipy.spatial.distance.pdist excludes self-distances and diagonal by construction
    distances = dist.pdist(subsampled_matrix, metric='euclidean')
    sigma = float(np.median(distances))
    return sigma


def load_normalized_feature_matrices(num_samples=5000, batch_size=32, device='cpu', dataset_type='pretrain'):
    """Extracts raw features from pretraining/validation dataset and normalizes them to [-1, 1]."""
    old_argv = sys.argv.copy()
    sys.argv = ['compute_median_sigma.py', '--pretrain', '--simu-exp']
    opts = opt_pretrain()
    args = opts.parse()
    dirs = opts.dir()
    sys.argv = old_argv

    fs = args.acoustic_setting['fs']
    data_dir = dirs['micsig_simu_pretrain'] if dataset_type == 'pretrain' else dirs['micsig_simu_preval']

    print(f"[INFO] Loading dataset from: {data_dir}")
    torch.manual_seed(42)
    dataset = at_dataset.FixMicSigDataset(
        data_dir=data_dir,
        fs=fs,
        load_anno=False,
        dataset_sz=num_samples,
        sort=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    do_stft = STFT(win_len=512, win_shift_ratio=0.5, nfft=512).to(device)

    all_keys = list(set(D13_HYBRID_KEYS + D5_ORIGINAL_KEYS))
    raw_collections = {k: [] for k in all_keys}

    loaded_count = 0
    with torch.no_grad():
        for batch in dataloader:
            waveform = batch[0] if isinstance(batch, list) else batch
            waveform = waveform.to(device)

            if waveform.dim() == 3:  # [B, T, C]
                stft_out = do_stft(waveform)  # [B, F, T, C]
                stft_out = stft_out.permute(0, 3, 1, 2)  # [B, C, F, T]
                feats_dict = spatial_features_from_stft(stft_out, sample_rate=fs)

                for k in all_keys:
                    if k in feats_dict:
                        raw_collections[k].append(feats_dict[k].cpu())

                loaded_count += waveform.size(0)
                if loaded_count >= num_samples:
                    break

    # Concatenate all batches
    normalized_dict = {}
    for k in all_keys:
        vals = torch.cat(raw_collections[k])[:num_samples]
        mean = torch.mean(vals)
        std = torch.std(vals)
        val_norm = (vals - mean) / (3.0 * std + 1e-6)
        val_norm = torch.clamp(val_norm, -1.0, 1.0)
        normalized_dict[k] = val_norm.numpy()

    # Construct D=13 and D=5 feature matrices
    X_D13 = np.column_stack([normalized_dict[k] for k in D13_HYBRID_KEYS])
    X_D5 = np.column_stack([normalized_dict[k] for k in D5_ORIGINAL_KEYS])

    return X_D13, X_D5


def main():
    parser = argparse.ArgumentParser(description="Compute Median Heuristic Sigma for SAC Loss")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of dataset samples to extract (default: 5000)")
    parser.add_argument("--subsample-size", type=int, default=2000, help="Subsample size for pdist computation (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dataset-type", type=str, default="pretrain", choices=["pretrain", "preval"], help="Dataset split (default: pretrain)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    print("=" * 70)
    print(" SAC LOSS - MEDIAN HEURISTIC BANDWIDTH (SIGMA) COMPUTATION ")
    print("=" * 70)
    print(f"Device:            {args.device}")
    print(f"Dataset Split:     {args.dataset_type}")
    print(f"Extracted Samples: {args.num_samples}")
    print(f"Subsample Size:    {args.subsample_size}")
    print(f"Random Seed:       {args.seed}")
    print("-" * 70)

    X_D13, X_D5 = load_normalized_feature_matrices(
        num_samples=args.num_samples,
        batch_size=64,
        device=args.device,
        dataset_type=args.dataset_type
    )

    print(f"[DATA] D=13 Feature Matrix Shape: {X_D13.shape}")
    print(f"[DATA] D=5  Feature Matrix Shape: {X_D5.shape}")

    # Compute Median Heuristic Sigmas
    sigma_median_D13 = compute_median_heuristic_sigma(X_D13, subsample_size=args.subsample_size, seed=args.seed)
    sigma_median_D5 = compute_median_heuristic_sigma(X_D5, subsample_size=args.subsample_size, seed=args.seed)

    print("\n" + "=" * 70)
    print(" COMPUTED BANDWIDTH (SIGMA) RESULTS & COMPARISON")
    print("=" * 70)
    print(f"-> NEW D=13 Median-Heuristic Sigma:  {sigma_median_D13:.6f}")
    print(f"-> Previous Tuned D=5 Sigma:         {PREVIOUS_TUNED_SIGMA_D5:.6f}")
    print(f"-> Previous Default D=5 Sigma:       1.000000")
    print(f"-> D=5 Median-Heuristic Sigma:       {sigma_median_D5:.6f}")
    print("-" * 70)
    print(f"Ratio (Median D=13 / Median D=5):    {sigma_median_D13 / sigma_median_D5:.4f} (Theoretical sqrt(13/5) = {np.sqrt(13/5):.4f})")
    print(f"Ratio (Median D=13 / Tuned D=5):     {sigma_median_D13 / PREVIOUS_TUNED_SIGMA_D5:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
