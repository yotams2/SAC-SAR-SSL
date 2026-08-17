"""
Unsupervised Principal Feature Analysis (PFA) Pipeline & Bootstrap Stability Check.
Steps: 1. Load Data (CSV/NPY/Fallback) | 2. PCA-95% Corr Rank | 3. Greedy Selection | 4. Bootstrap Stability | 5. Handpicked Comparison | 6. Output & Plots
"""

import os
import argparse
import numpy as np
import scipy.spatial.distance as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# Named Constants
# ==========================================
CORRELATION_MODE = True       # Use correlation matrix
VARIANCE_THRESHOLD = 0.95     # PCA explained variance target (95%)
N_BOOTSTRAP = 50              # Bootstrap resamples
DATA_PATH = ""                # Path to CSV or NPY file (empty = synthetic fallback)
NAMES_PATH = ""               # Optional separate text/json file for feature names
HANDPICKED_FEATURES = ['tdoa', 'gcc_phat_max', 'coherence_low', 'coherence_mid', 'coherence_high']
OUTPUT_DIR = "analysis_results/feature_selection"  # Organized output directory
SEED = 42                     # Random seed


# ==========================================
# Step 1: Data Loading & Synthetic Fallback
# ==========================================
def _extract_single_file(w):
    import os, sys, soundfile, torch
    torch.set_num_threads(1)
    code_dir = os.path.dirname(os.path.abspath(__file__))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import common.utils_module as utils
    try:
        sig, fs = soundfile.read(w)
        if sig.ndim == 2 and sig.shape[1] == 2:
            sig_t = torch.tensor(sig.T, dtype=torch.float32).unsqueeze(0)
            spat_dict = utils.compute_spatial_features(sig_t, sample_rate=fs)
            vec, names_curr = [], []
            for k, v in sorted(spat_dict.items()):
                val = v.squeeze().detach().numpy()
                if val.ndim == 0: vec.append(float(val)); names_curr.append(k)
                elif val.ndim == 1:
                    for i, elem in enumerate(val): vec.append(float(elem)); names_curr.append(f"{k}_{i}")
            return vec, names_curr
    except Exception:
        pass
    return None, None


def load_feature_matrix(data_path=DATA_PATH, names_path=NAMES_PATH, seed=SEED):
    """Loads feature matrix from CSV/NPY/Directory or falls back to placeholder synthetic generator."""
    if data_path and os.path.exists(data_path):
        print(f"[INFO] Loading real feature matrix from: {data_path}")
        if os.path.isdir(data_path):
            import glob
            from concurrent.futures import ProcessPoolExecutor
            wav_files = sorted(glob.glob(os.path.join(data_path, '*.wav')))
            wav_files = [w for w in wav_files if not w.endswith('_dp.wav')][:1500]
            print(f"[INFO] Extracting features for {len(wav_files)} dataset files...")
            
            with ProcessPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(_extract_single_file, wav_files))
                
            spat_list, names = [], []
            for vec, names_curr in results:
                if vec is not None:
                    spat_list.append(vec)
                    names = names_curr
                    
            X = np.array(spat_list)
            X_norm = (X - np.mean(X, axis=0)) / (3.0 * np.std(X, axis=0) + 1e-8)
            return np.clip(X_norm, -1.0, 1.0), names, False
        elif data_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_path)
            return df.values.astype(np.float64), list(df.columns), False
        elif data_path.endswith('.npy'):
            X = np.load(data_path).astype(np.float64)
            names = [line.strip() for line in open(names_path)] if names_path and os.path.exists(names_path) else [f"feat_{i}" for i in range(X.shape[1])]
            return X, names, False

    print("\n" + "!" * 70)
    print("[WARNING] Real feature file not found or empty path specified!")
    print("[WARNING] Running on PLACEHOLDER synthetic candidate matrix (N=5000, D=100, r_true=8).")
    print("!" * 70 + "\n")
    
    rng = np.random.RandomState(seed)
    n_samples, n_features, r_true = 5000, 100, 8
    c_ids = rng.choice(4, size=n_samples)
    means, stds = rng.uniform(-2.0, 2.0, (4, r_true)), rng.uniform(0.5, 1.5, (4, r_true))
    Z = np.array([rng.normal(means[c], stds[c]) for c in c_ids])
    X_raw = Z @ rng.randn(r_true, n_features) + rng.normal(0, 0.1, (n_samples, n_features))
    X_norm = (X_raw - np.mean(X_raw, axis=0)) / (3.0 * np.std(X_raw, axis=0) + 1e-8)
    return np.clip(X_norm, -1.0, 1.0), [f"candidate_feat_{i:03d}" for i in range(n_features)], True


# ==========================================
# Step 2: PCA-95% on Correlation Matrix
# ==========================================
def compute_pca_95_corr_rank(X, threshold=VARIANCE_THRESHOLD):
    """Computes correlation matrix R and minimum components k for 95% variance."""
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    R = (X_std.T @ X_std) / (X.shape[0] - 1)
    eigvals = np.maximum(np.linalg.eigvalsh(R)[::-1], 0)
    return int(np.searchsorted(np.cumsum(eigvals) / np.sum(eigvals), threshold) + 1), R


# ==========================================
# Step 3: Greedy Original-Feature Selection
# ==========================================
def greedy_pfa_selection(R, k_target):
    """Greedily selects k original feature columns maximizing full correlation variance explained."""
    selected, candidates, n_features = [], list(range(R.shape[0])), R.shape[0]
    for _ in range(k_target):
        best_score, best_feat = -1.0, -1
        for feat in candidates:
            sub = selected + [feat]
            R_full_sub = R[:, sub]
            score = float(np.sum((R_full_sub @ np.linalg.pinv(R[np.ix_(sub, sub)])) * R_full_sub) / n_features)
            if score > best_score:
                best_score, best_feat = score, feat
        selected.append(best_feat)
        candidates.remove(best_feat)
    return selected


# ==========================================
# Step 4: Bootstrap Stability Check
# ==========================================
def bootstrap_stability_check(X, n_bootstrap=N_BOOTSTRAP, threshold=VARIANCE_THRESHOLD, seed=SEED):
    """Resamples N rows with replacement 50 times to check stability of k and feature selection."""
    n_samples, n_features = X.shape
    rng = np.random.RandomState(seed)
    k_list, counts = [], np.zeros(n_features, dtype=int)
    
    for _ in range(n_bootstrap):
        X_b = X[rng.choice(n_samples, size=n_samples, replace=True)]
        k_b, R_b = compute_pca_95_corr_rank(X_b, threshold)
        k_list.append(k_b)
        counts[greedy_pfa_selection(R_b, k_b)] += 1
        
    k_arr = np.array(k_list)
    k_u, k_c = np.unique(k_arr, return_counts=True)
    return {
        'k_list': k_list, 'k_mean': float(np.mean(k_arr)), 'k_std': float(np.std(k_arr)),
        'k_min': int(np.min(k_arr)), 'k_max': int(np.max(k_arr)), 'k_dist': dict(zip(k_u.tolist(), k_c.tolist())),
        'selection_frequencies': counts / n_bootstrap
    }


# ==========================================
# Step 5: Comparison Against Hand-Picked Features
# ==========================================
def compare_handpicked_set(selected_indices, feature_names, boot_freqs, R_full, handpicked=HANDPICKED_FEATURES):
    """Compares new selection to current production 5 features."""
    if not handpicked:
        return None
    hp_indices = [i if isinstance(i, int) and 0 <= i < len(feature_names) else feature_names.index(i) for i in handpicked if i in feature_names or (isinstance(i, int) and 0 <= i < len(feature_names))]
    if not hp_indices:
        return None
    
    indiv_var = np.sum(R_full ** 2, axis=0) / R_full.shape[0]
    rank_order = np.argsort(-indiv_var)
    overlap = set(selected_indices).intersection(set(hp_indices))
    
    details = [{
        'index': idx, 'name': feature_names[idx], 'in_selected': idx in selected_indices,
        'boot_freq': boot_freqs[idx], 'var_rank': int(np.where(rank_order == idx)[0][0]) + 1
    } for idx in hp_indices]
    
    return {'overlap_count': len(overlap), 'overlap_fraction': len(overlap) / len(hp_indices), 'details': details}


# ==========================================
# Step 6: Output & Visualization
# ==========================================
def save_stability_plots(boot_results, selected_indices, feature_names, output_dir=OUTPUT_DIR):
    """Saves organized PNG plots for bootstrap frequency and k distribution."""
    os.makedirs(output_dir, exist_ok=True)
    freqs = boot_results['selection_frequencies']
    sorted_idx = np.argsort(-freqs)
    colors = ['#2ca02c' if sel else '#1f77b4' for sel in np.isin(sorted_idx, selected_indices)]
    
    plt.figure(figsize=(11, 4.5))
    plt.bar(range(len(freqs)), freqs[sorted_idx], color=colors, alpha=0.85)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Stability Threshold')
    plt.title('Bootstrap Feature Selection Frequency (50 Resamples)'); plt.xlabel('Features (Sorted)'); plt.ylabel('Frequency')
    plt.legend(); plt.grid(True, axis='y', ls=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_selection_frequency.png'), dpi=300); plt.close()
    
    plt.figure(figsize=(6.5, 4))
    plt.hist(boot_results['k_list'], bins=np.arange(min(boot_results['k_list']) - 0.5, max(boot_results['k_list']) + 1.5, 1), color='#9467bd', edgecolor='black', rwidth=0.8)
    plt.title('Distribution of Target Rank k (50 Resamples)'); plt.xlabel('Target Rank k'); plt.ylabel('Count')
    plt.grid(True, axis='y', ls=':', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bootstrap_k_distribution.png'), dpi=300); plt.close()


def main():
    parser = argparse.ArgumentParser(description="Unsupervised PFA Feature Selection & Stability Analysis")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to input feature CSV or NPY file")
    parser.add_argument("--names-path", type=str, default=NAMES_PATH, help="Path to optional feature names file")
    args = parser.parse_args()
    
    X, feature_names, is_synthetic = load_feature_matrix(args.data_path, args.names_path, SEED)
    k_full, R_full = compute_pca_95_corr_rank(X, VARIANCE_THRESHOLD)
    selected_indices = greedy_pfa_selection(R_full, k_full)
    selected_names = [feature_names[i] for i in selected_indices]
    
    boot_res = bootstrap_stability_check(X, N_BOOTSTRAP, VARIANCE_THRESHOLD, SEED)
    freqs = boot_res['selection_frequencies']
    stable_count = np.sum(freqs[selected_indices] >= 0.8)
    k_var_flag = (boot_res['k_max'] - boot_res['k_min']) > 4
    stab_flag = stable_count < (len(selected_indices) / 2.0)
    
    hp_comp = compare_handpicked_set(selected_indices, feature_names, freqs, R_full, HANDPICKED_FEATURES)
    save_stability_plots(boot_res, selected_indices, feature_names, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print(" UNSUPERVISED FEATURE SELECTION (PFA) - ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Data Source:               {'PLACEHOLDER SYNTHETIC DATA' if is_synthetic else args.data_path}")
    print(f"Matrix Shape:              {X.shape[0]} samples x {X.shape[1]} candidate features")
    print(f"PCA-95% Target Rank k:     {k_full}")
    print(f"Selected Feature Indices:  {selected_indices}")
    print(f"Selected Feature Names:    {selected_names[:5]}{'...' if len(selected_names) > 5 else ''}")
    
    print("\n--- Bootstrap Stability Statistics (50 Resamples) ---")
    print(f"k Distribution:            Mean={boot_res['k_mean']:.2f}, Std={boot_res['k_std']:.2f}, Range=[{boot_res['k_min']}, {boot_res['k_max']}]")
    print(f"k Frequency Counts:        {boot_res['k_dist']}")
    print(f"Selected Features >=80% Stable: {stable_count} / {len(selected_indices)} ({stable_count / len(selected_indices):.1%})")
    
    if k_var_flag:
        print("[WARNING] Target rank k varies by more than +/-2 across resamples!")
    if stab_flag:
        print("[WARNING] Fewer than half of selected features appear in >=80% of bootstrap runs!")
    if not k_var_flag and not stab_flag:
        print("[STABILITY CHECK PASSED] Rank k and feature selection are highly stable across resamples.")
        
    print("\n--- Comparison with Hand-Picked Production Features ---")
    if hp_comp:
        print(f"Overlap with Production Set: {hp_comp['overlap_count']} / {len(HANDPICKED_FEATURES)} ({hp_comp['overlap_fraction']:.1%})")
        for d in hp_comp['details']:
            print(f"  Feature {d['index']:02d} ({d['name']}): Status={'SELECTED' if d['in_selected'] else 'Not selected'}, BootFreq={d['boot_freq']:.1%}, Indiv Var Rank={d['var_rank']}")
    else:
        print("Hand-picked production feature comparison skipped.")
        
    print("\n--- Output Directory & Figures ---")
    print(f"Saved PNG plots to organized directory: '{OUTPUT_DIR}/'")
    print("  - bootstrap_selection_frequency.png\n  - bootstrap_k_distribution.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
