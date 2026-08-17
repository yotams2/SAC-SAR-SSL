"""
Multi-Seed Stability Check for Median-Heuristic Bandwidth (Sigma) Computation.

This script runs `compute_median_heuristic_sigma` across 5 different random seeds
(42, 123, 7, 2026, 99) using the production-realistic sampling path:
  - `shuffle=True`, `sort=False`
  - Shuffled 5,000-file sample from the full 100,000-file pretraining set per seed
  - Evaluates D=13 (hybrid feature set) and D=5 median-heuristic sigmas
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
from compute_median_sigma import (
    compute_median_heuristic_sigma,
    D13_HYBRID_KEYS,
    D5_ORIGINAL_KEYS,
    PREVIOUS_TUNED_SIGMA_D5,
)

SEEDS = [42, 123, 7, 2026, 99]


def extract_sample_and_compute(seed, num_samples=5000, subsample_size=2000, batch_size=64, device='cuda'):
    """Extracts a fresh 5,000-file shuffled sample for a given seed and computes D=13 and D=5 median sigmas."""
    # Set seeds for PyTorch, NumPy, and random generator
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    old_argv = sys.argv.copy()
    sys.argv = ['check_multi_seed_stability.py', '--pretrain', '--simu-exp']
    opts = opt_pretrain()
    args = opts.parse()
    dirs = opts.dir()
    sys.argv = old_argv

    fs = args.acoustic_setting['fs']
    data_dir = dirs['micsig_simu_pretrain']

    dataset = at_dataset.FixMicSigDataset(
        data_dir=data_dir,
        fs=fs,
        load_anno=False,
        dataset_sz=None,  # Full dataset pool (100,000 files)
        sort=False        # Production-realistic path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0
    )

    do_stft = STFT(win_len=512, win_shift_ratio=0.5, nfft=512).to(device)

    all_keys = sorted(list(set(D13_HYBRID_KEYS + D5_ORIGINAL_KEYS)))
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

    # Concatenate and normalize features to [-1, 1]
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

    sigma_D13 = compute_median_heuristic_sigma(X_D13, subsample_size=subsample_size, seed=seed)
    sigma_D5 = compute_median_heuristic_sigma(X_D5, subsample_size=subsample_size, seed=seed)

    return sigma_D13, sigma_D5


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Stability Check for Median-Heuristic Sigma")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of shuffled dataset samples (default: 5000)")
    parser.add_argument("--subsample-size", type=int, default=2000, help="Subsample size for pdist computation (default: 2000)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    print("=" * 75)
    print(" MULTI-SEED STABILITY CHECK FOR MEDIAN HEURISTIC BANDWIDTH (SIGMA) ")
    print("=" * 75)
    print(f"Device:             {args.device}")
    print(f"Sample Size:        {args.num_samples} files per seed (shuffled from 100k pool)")
    print(f"Subsample Size:     {args.subsample_size} for pdist")
    print(f"Seeds Evaluated:    {SEEDS}")
    print("-" * 75)

    results_D13 = []
    results_D5 = []

    for seed in SEEDS:
        print(f"\n[INFO] Running evaluation for Seed {seed}...")
        sig_13, sig_5 = extract_sample_and_compute(
            seed=seed,
            num_samples=args.num_samples,
            subsample_size=args.subsample_size,
            batch_size=64,
            device=args.device
        )
        results_D13.append(sig_13)
        results_D5.append(sig_5)
        print(f"  --> Seed {seed:4d}: D=13 Sigma = {sig_13:.6f} | D=5 Sigma = {sig_5:.6f} | Ratio = {sig_13/sig_5:.4f}")

    arr_13 = np.array(results_D13)
    arr_5 = np.array(results_D5)

    mean_13, std_13 = np.mean(arr_13), np.std(arr_13)
    min_13, max_13 = np.min(arr_13), np.max(arr_13)
    cv_13 = (std_13 / mean_13) * 100.0

    mean_5, std_5 = np.mean(arr_5), np.std(arr_5)
    min_5, max_5 = np.min(arr_5), np.max(arr_5)
    cv_5 = (std_5 / mean_5) * 100.0

    print("\n" + "=" * 75)
    print(" MULTI-SEED STABILITY RESULTS & STATISTICAL SUMMARY ")
    print("=" * 75)
    print(" Seed      | D=13 Sigma   | D=5 Sigma    | Ratio (D13/D5)")
    print("-" * 75)
    for i, s in enumerate(SEEDS):
        print(f" {s:<9d} | {results_D13[i]:<12.6f} | {results_D5[i]:<12.6f} | {results_D13[i]/results_D5[i]:<12.4f}")
    print("-" * 75)
    print(f" MEAN      | {mean_13:<12.6f} | {mean_5:<12.6f} | {mean_13/mean_5:<12.4f}")
    print(f" STD       | {std_13:<12.6f} | {std_5:<12.6f} |")
    print(f" MIN       | {min_13:<12.6f} | {min_5:<12.6f} |")
    print(f" MAX       | {max_13:<12.6f} | {max_5:<12.6f} |")
    print(f" CV (%)    | {cv_13:<11.2f}% | {cv_5:<11.2f}% |")
    print("=" * 75)

    if cv_13 <= 5.0:
        print("\n[STABILITY CONFIRMED] The D=13 median-heuristic sigma values across seeds are tightly")
        print(f"clustered (Coefficient of Variation = {cv_13:.2f}% <= 5.0%).")
        print(f"Selecting the mean sigma = {mean_13:.4f} is a stable, representative constant for pretraining.")
    else:
        print("\n[WARNING: HIGH VARIANCE DETECTED] The median-heuristic sigma values across seeds vary")
        print(f"meaningfully (Coefficient of Variation = {cv_13:.2f}% > 5.0%).")
        print("This indicates dataset heterogeneity makes a single precomputed constant sigma unreliable.")
        print("Sigma should instead be computed dynamically per batch/run.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
