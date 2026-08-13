import os
import sys
import argparse
import numpy as np
import torch
import math

# Add root code directory to path to allow importing from tuning/ or common/
sys.path.insert(0, '/storage/yotam/SAR-SSL/SAR-SSL/code')
try:
    from pipeline_sigma_tau import load_heldout_features
except ImportError:
    from tuning.pipeline_sigma_tau import load_heldout_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma-min', type=float, default=0.1)
    parser.add_argument('--sigma-max', type=float, default=1.5)
    parser.add_argument('--sigma-steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--bins', type=int, default=60)
    parser.add_argument('--num-batches', type=int, default=4)
    parser.add_argument('--out', type=str, default='sigma_histogram_cache.npz')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fixed seed for reproducibility of batches
    torch.manual_seed(0)
    np.random.seed(0)

    # Load fixed subset of features
    num_samples = args.num_batches * args.batch_size
    features = load_heldout_features(num_samples, args.batch_size, device)

    # Split into fixed batches
    batches = []
    for i in range(args.num_batches):
        batch = features[i * args.batch_size : (i + 1) * args.batch_size]
        if batch.size(0) == args.batch_size:
            batches.append(batch)
            
    if len(batches) < args.num_batches:
        print(f"Warning: requested {args.num_batches} batches but only got {len(batches)}.")
    actual_num_batches = len(batches)

    # Setup sigma grid
    sigmas = np.logspace(math.log10(args.sigma_min), math.log10(args.sigma_max), args.sigma_steps)

    # Universal fixed bin edges spanning [0, 1]
    bin_edges = np.linspace(0.0, 1.0, args.bins + 1)

    counts = np.zeros((len(sigmas), actual_num_batches, args.bins), dtype=np.int32)

    print(f"Precomputing histograms for {len(sigmas)} sigmas and {actual_num_batches} batches...")

    for i, sigma in enumerate(sigmas):
        for b in range(actual_num_batches):
            batch_feats = batches[b]
            D = torch.cdist(batch_feats, batch_feats, p=2)
            off_diag = ~torch.eye(batch_feats.size(0), dtype=torch.bool, device=device)

            D_sq = D**2
            D_sq_off_diag = D_sq.masked_fill(~off_diag, float('inf'))
            row_min = D_sq_off_diag.min(dim=1, keepdim=True).values
            W = torch.exp(- (D_sq_off_diag - row_min) / sigma**2)
            
            W_masked = W * off_diag.float()
            W_sum = W_masked.sum(dim=1, keepdim=True) + 1e-8
            W_norm = W_masked / W_sum

            W_norm_neighbors = W_norm[off_diag]

            hist, _ = np.histogram(W_norm_neighbors.cpu().numpy(), bins=bin_edges)
            counts[i, b] = hist

    # Save to cache file
    np.savez(
        args.out,
        sigmas=sigmas,
        bin_edges=bin_edges,
        counts=counts
    )
    print(f"Cache saved to {args.out}")

if __name__ == "__main__":
    main()
