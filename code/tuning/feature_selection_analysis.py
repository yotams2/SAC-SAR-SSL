"""
Unsupervised Feature Selection Pipeline for SAC Loss via Effective Rank Optimization.
Steps: 1. Synthetic Data Gen | 2. Effective Rank | 3. Greedy Selection | 4. Ground Truth Validation | 5. Kernel Sanity | 6. Visualization
"""

import numpy as np
import scipy.spatial.distance as dist
from sklearn.cross_decomposition import CCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# Named Constants
# ==========================================
N_SAMPLES = 5000       # N samples
N_FEATURES = 100      # D candidate features
R_TRUE = 8             # Ground truth latent rank
SIGMA = 1.0            # Gaussian kernel bandwidth
SEED = 42              # Random seed
N_SUBSAMPLE = 200      # Kernel check subsample size


# ==========================================
# Step 1: Synthetic Data Generation
# ==========================================
def generate_synthetic_data(n_samples=N_SAMPLES, n_features=N_FEATURES, r_true=R_TRUE, seed=SEED):
    """Generates synthetic matrix X (N, D) from latent Z (N, r_true) drawn from Gaussian mixture."""
    rng = np.random.RandomState(seed)
    n_clusters = 4
    cluster_ids = rng.choice(n_clusters, size=n_samples)
    means, stds = rng.uniform(-2.0, 2.0, size=(n_clusters, r_true)), rng.uniform(0.5, 1.5, size=(n_clusters, r_true))
    
    Z = np.zeros((n_samples, r_true))
    for i in range(n_samples):
        Z[i] = rng.normal(means[cluster_ids[i]], stds[cluster_ids[i]])
    
    M = rng.randn(r_true, n_features)
    X_noisy = Z @ M + rng.normal(0, 0.1, size=(n_samples, n_features))
    X_norm = (X_noisy - np.mean(X_noisy, axis=0)) / (3.0 * np.std(X_noisy, axis=0) + 1e-8)
    return np.clip(X_norm, -1.0, 1.0), Z


# ==========================================
# Step 2: Effective Rank Estimation
# ==========================================
def compute_effective_rank(X):
    """Roy & Vetterli entropy-based effective rank: exp(H(p))."""
    s = np.linalg.svd(X - np.mean(X, axis=0), compute_uv=False)
    s_sum = np.sum(s)
    if s_sum <= 0:
        return 1.0
    p = s / s_sum
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def compute_pca_95_rank(X, threshold=0.95):
    """PCA 95% explained variance rank threshold."""
    s = np.linalg.svd(X - np.mean(X, axis=0), compute_uv=False)
    cum_var = np.cumsum((s ** 2) / np.sum(s ** 2))
    return int(np.searchsorted(cum_var, threshold) + 1)


# ==========================================
# Step 3: Greedy Interpretable Feature Selection
# ==========================================
def greedy_feature_selection(X, k_target, criterion='explained_variance'):
    """Greedy forward selection maximizing effective_rank or explained_variance."""
    selected, candidates = [], list(range(X.shape[1]))
    history_ranks = []
    total_var = np.sum(np.var(X, axis=0)) + 1e-8
    
    for _ in range(k_target):
        best_score, best_feat = -1.0, -1
        for feat in candidates:
            X_sub = X[:, selected + [feat]]
            score = compute_effective_rank(X_sub) if criterion == 'effective_rank' else np.sum(np.var(X_sub, axis=0)) / total_var
            if score > best_score:
                best_score, best_feat = score, feat
        selected.append(best_feat)
        candidates.remove(best_feat)
        history_ranks.append(compute_effective_rank(X[:, selected]))
    return selected, history_ranks


# ==========================================
# Step 4: Validation Against Ground Truth
# ==========================================
def validate_with_ground_truth(X_selected, Z):
    """Unsupervised validation against Z using Canonical Correlation Analysis (CCA)."""
    n_comp = min(X_selected.shape[1], Z.shape[1])
    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(X_selected, Z)
    X_c, Z_c = cca.transform(X_selected, Z)
    corrs = [float(np.corrcoef(X_c[:, i], Z_c[:, i])[0, 1]) for i in range(n_comp)]
    return {'cca_correlations': corrs, 'mean_cca_corr': float(np.mean(corrs)), 'n_components': n_comp}


# ==========================================
# Step 5: Degenerate Kernel Sanity Check
# ==========================================
def evaluate_kernel_sanity(X_selected, sigma=SIGMA, n_subsample=N_SUBSAMPLE, seed=SEED):
    """Gaussian kernel sanity check w_ij = exp(-||x_i - x_j||^2 / sigma^2)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(X_selected.shape[0], size=min(n_subsample, X_selected.shape[0]), replace=False)
    W = np.exp(-dist.cdist(X_selected[idx], X_selected[idx], metric='sqeuclidean') / (sigma ** 2))
    w_off = W[~np.eye(W.shape[0], dtype=bool)]
    
    return {
        'mean': float(np.mean(W)), 'std': float(np.std(W)),
        'off_mean': float(np.mean(w_off)), 'off_std': float(np.std(w_off)),
        'frac_zero': float(np.mean(W < 0.01)), 'frac_one': float(np.mean(W > 0.99)),
        'is_degenerate': bool(np.mean(W < 0.01) > 0.9 or np.mean(W > 0.99) > 0.9 or np.std(W) < 0.01)
    }


# ==========================================
# Step 6: Visualization & Main Execution
# ==========================================
def plot_results(s_vals, erank, pca_95_rank, ranks_history):
    """Plots singular value spectrum and effective rank trajectory."""
    # Plot 1: Singular Value Spectrum
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(s_vals) + 1), s_vals, 'b.-', lw=1.5, label='Singular Values')
    plt.axvline(x=erank, color='r', linestyle='--', label=f'Effective Rank ({erank:.2f})')
    plt.title('Singular Value Spectrum of Candidate Matrix X')
    plt.xlabel('Component Index'); plt.ylabel('Singular Value')
    plt.grid(True, ls=':', alpha=0.6); plt.legend(); plt.tight_layout()
    plt.savefig('singular_value_spectrum.png', dpi=300)
    plt.close()

    # Plot 2: Effective Rank Trajectory
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(ranks_history) + 1), ranks_history, 'go-', lw=1.5, label='Selected Subset Rank')
    plt.axhline(y=erank, color='r', linestyle='--', label=f'Full Matrix Rank ({erank:.2f})')
    plt.axhline(y=pca_95_rank, color='m', linestyle='-.', label=f'PCA 95% Rank (target: {pca_95_rank})')
    plt.axhline(y=R_TRUE, color='k', linestyle=':', label=f'Ground-Truth Latent Rank ({R_TRUE})')
    plt.title('Greedy Feature Selection: Effective Rank Trajectory')
    plt.xlabel('Number of Greedily Selected Features (k)'); plt.ylabel('Effective Rank e^H')
    plt.grid(True, ls=':', alpha=0.6); plt.legend(); plt.tight_layout()
    plt.savefig('effective_rank_vs_features.png', dpi=300)
    plt.close()


def main():
    print("=== Step 1: Synthetic Data Generation ===")
    X, Z = generate_synthetic_data(N_SAMPLES, N_FEATURES, R_TRUE, SEED)
    print(f"Generated X matrix shape: {X.shape}, Ground Truth Z shape: {Z.shape}")

    print("\n=== Step 2: Effective Rank Estimation ===")
    erank = compute_effective_rank(X)
    pca_95_rank = compute_pca_95_rank(X)
    k_target = pca_95_rank
    print(f"Roy & Vetterli Effective Rank (e^H): {erank:.4f}")
    print(f"PCA 95% Explained Variance Rank:    {pca_95_rank}")
    print(f"Target Selection Count (k_target):  {k_target}")

    print("\n=== Step 3: Greedy Forward Feature Selection ===")
    selected_indices, ranks_history = greedy_feature_selection(X, k_target, criterion='explained_variance')
    print(f"Selected {len(selected_indices)} feature indices: {selected_indices}")

    print("\n=== Step 4: Validation Against Ground Truth ===")
    X_selected = X[:, selected_indices]
    val_results = validate_with_ground_truth(X_selected, Z)
    print(f"Recovered Effective Rank k: {k_target} vs Ground Truth r_true: {R_TRUE}")
    print(f"CCA Canonical Components:   {val_results['n_components']}")
    print(f"Mean Canonical Correlation:  {val_results['mean_cca_corr']:.4f}")
    print(f"Canonical Correlations:      {[round(c, 4) for c in val_results['cca_correlations']]}")

    print("\n=== Step 5: SAC Gaussian Kernel Sanity Check ===")
    kernel_stats = evaluate_kernel_sanity(X_selected, SIGMA, N_SUBSAMPLE, SEED)
    print(f"Kernel Matrix (200x200 subsample, sigma={SIGMA}):")
    print(f"  Mean Weight w_ij:          {kernel_stats['mean']:.4f} (Off-diag: {kernel_stats['off_mean']:.4f})")
    print(f"  Std Dev w_ij:              {kernel_stats['std']:.4f} (Off-diag: {kernel_stats['off_std']:.4f})")
    print(f"  Fraction close to 0 (<0.01): {kernel_stats['frac_zero']:.4f}")
    print(f"  Fraction close to 1 (>0.99): {kernel_stats['frac_one']:.4f}")
    print(f"  Degenerate Kernel Flag:    {kernel_stats['is_degenerate']}")

    print("\n=== Step 6: Saving Plots & Summary Output ===")
    s_vals = np.linalg.svd(X - np.mean(X, axis=0), compute_uv=False)
    plot_results(s_vals, erank, pca_95_rank, ranks_history)
    print("Saved plots: 'singular_value_spectrum.png' and 'effective_rank_vs_features.png'")


if __name__ == "__main__":
    main()
