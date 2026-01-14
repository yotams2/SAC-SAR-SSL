
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import FixMicSigDataset
from common.utils_module import spatial_features_from_stft, STFT
from torch.utils.data import DataLoader

def get_stats(vals):
    return np.mean(vals), np.std(vals)

def calculate_feature_stats(dataloader, device, fs, max_samples=200):
    """
    Calculates mean and std for spatial features from the dataset.
    """
    print("Calculating feature stats...")
    ild_values = []
    ipd_cos_values, ipd_sin_values = [], []
    coherence_mean_values, coherence_low_values, coherence_mid_values = [], [], []
    coherence_high_values, coherence_var_values = [], []
    gcc_phat_max_values = []
    hf_lf_ratio_values = []
    tdoa_values = []
    iacc_max_values = []
    # New features
    decay_slope_median_values = []
    decay_slope_p10_values = []
    decay_slope_p90_values = []
    mod_low_high_ratio_values = []
    early_late_ratio_mean_values = []
    early_late_ratio_median_values = []

    stft_module = STFT(win_len=512, win_shift_ratio=0.5, nfft=512).to(device)

    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=min(len(dataloader), max_samples)):
            if count >= max_samples:
                break
            
            # batch is typically (sig, labels) or just sig depending on dataset
            # FixMicSigDataset returns (sig, label_dict)
            sig = batch[0].to(device) # (B, M, T)
            
            # STFT
            # shape: (B, F, T, C) -> (B, C, F, T)
            stft_out = stft_module(sig) 
            stft_out = stft_out.permute(0, 3, 1, 2)
            
            # Extract features
            # Pass hop_length=256 matching STFT params above (win_len=512, win_shift_ratio=0.5 -> hop=256)
            features = spatial_features_from_stft(stft_out, sample_rate=fs, hop_length=256)
            
            # Accumulate
            # Features are typically (B, T_frames) or similar, we flatten them
            ild_values.append(features['ild_mean'].cpu().numpy().flatten())
            # ipd_cos_values.append(features['ipd_cos_mean'].cpu().numpy().flatten())
            # ipd_sin_values.append(features['ipd_sin_mean'].cpu().numpy().flatten())
            # coherence_mean_values.append(features['coherence_mean'].cpu().numpy().flatten())
            coherence_low_values.append(features['coherence_low'].cpu().numpy().flatten())
            coherence_mid_values.append(features['coherence_mid'].cpu().numpy().flatten())
            coherence_high_values.append(features['coherence_high'].cpu().numpy().flatten())
            coherence_var_values.append(features['coherence_var'].cpu().numpy().flatten())
            gcc_phat_max_values.append(features['gcc_phat_max'].cpu().numpy().flatten())
            hf_lf_ratio_values.append(features['hf_lf_ratio'].cpu().numpy().flatten())
            tdoa_values.append(features['tdoa'].cpu().numpy().flatten())
            iacc_max_values.append(features['iacc_max'].cpu().numpy().flatten())
            decay_slope_median_values.append(features['decay_slope_median'].cpu().numpy().flatten())
            decay_slope_p10_values.append(features['decay_slope_p10'].cpu().numpy().flatten())
            decay_slope_p90_values.append(features['decay_slope_p90'].cpu().numpy().flatten())
            mod_low_high_ratio_values.append(features['mod_low_high_ratio'].cpu().numpy().flatten())
            early_late_ratio_mean_values.append(features['early_late_ratio_mean'].cpu().numpy().flatten())
            early_late_ratio_median_values.append(features['early_late_ratio_median'].cpu().numpy().flatten())
            
            count += 1

    feature_stats = {}
    
    def add_stat(name, vals_list):
        if len(vals_list) > 0:
            vals = np.concatenate(vals_list)
            m, s = get_stats(vals)
            feature_stats[name] = {'mean': float(m), 'std': float(s)}

    add_stat('ild_mean', ild_values)
    # add_stat('ipd_cos_mean', ipd_cos_values)
    # add_stat('ipd_sin_mean', ipd_sin_values)
    # add_stat('coherence_mean', coherence_mean_values)
    add_stat('coherence_low', coherence_low_values)
    add_stat('coherence_mid', coherence_mid_values)
    add_stat('coherence_high', coherence_high_values)
    add_stat('coherence_var', coherence_var_values)
    add_stat('gcc_phat_max', gcc_phat_max_values)
    add_stat('hf_lf_ratio', hf_lf_ratio_values)
    add_stat('tdoa', tdoa_values)
    add_stat('iacc_max', iacc_max_values)
    add_stat('decay_slope_median', decay_slope_median_values)
    add_stat('decay_slope_p10', decay_slope_p10_values)
    add_stat('decay_slope_p90', decay_slope_p90_values)
    add_stat('mod_low_high_ratio', mod_low_high_ratio_values)
    add_stat('early_late_ratio_mean', early_late_ratio_mean_values)
    add_stat('early_late_ratio_median', early_late_ratio_median_values)
    
    return feature_stats

def calculate_weights(features_batch, feature_stats, keys, sigma=1.0):
    """
    Replicates the weight calculation logic from SARSSL model.
    W = exp( - (dist / sigma)^2 )
    Dist is Euclidean distance between normalized feature vectors.
    """
    # Create feature vector
    # Corresponding to 'cues' in model.py
    # We need to stack selected features and normalize them
    
    vecs = []
    for k in keys:
        if k in features_batch and k in feature_stats:
            val = features_batch[k] # Shape (B, ...)
            mu = feature_stats[k]['mean']
            std = feature_stats[k]['std']
            
            # Debug NaNs
            mu_isnan = np.isnan(mu).any() if not isinstance(mu, torch.Tensor) else torch.isnan(mu).any()
            std_isnan = np.isnan(std).any() if not isinstance(std, torch.Tensor) else torch.isnan(std).any()
            val_isnan = torch.isnan(val).any()

            if mu_isnan or std_isnan:
                 print(f"WARNING: Feature stats for '{k}' contain NaNs! Mean: {mu_isnan}, Std: {std_isnan}")
            if val_isnan:
                 print(f"WARNING: Feature '{k}' contains NaNs in the batch!")
            
            # Match model.py normalization: (val - mean) / (3 * std + 1e-6)
            norm_val = (val - mu) / (3 * std + 1e-6)
            # Match model.py clamping
            norm_val = torch.clamp(norm_val, -1.0, 1.0)
            vecs.append(norm_val)
            
    if not vecs:
        return None, None

    # Stack features: Shape (B, N_features, ...)
    # Flatten spatial/temporal dims for simpler distance calc if needed, 
    # but usually we want pairwise weights between samples in the batch.
    
    # Let's assume we want weights between items in the batch.
    # Stack: (B, FeatureDim) - need to pool over time if features are time-varying
    # Or if features are already pooled. spatial_features_from_stft returns pooled or per-frame?
    # Checking utils_module: usually returns pooled means if keys are like 'ild_mean'.
    # Yes, keys have 'mean' suffix.
    
    cues = torch.stack(vecs, dim=1) # (B, F_dim)
    
    # Calculate pairwise distance matrix
    # dist[i, j] = ||cues[i] - cues[j]||
    dist = torch.cdist(cues, cues, p=2) # (B, B)
    
    weights = torch.exp(- (dist / sigma) ** 2)
    
    return weights, dist

def plot_histograms(m_data, output_dir, prefix=''):
    """
    Plot histograms for sim_weights and cue_dist.
    Adapted from run_pretrain.py
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Similarity Weights
    if 'sim_weights' in m_data:
        w_list = m_data['sim_weights']
        if len(w_list) > 0:
            # Exclude diagonal (self-weights) for global histogram
            w_all_list = []
            for x in w_list:
                # x is (B, B)
                B = x.shape[0]
                if B > 1:
                    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
                    w_all_list.append(torch.masked_select(x, mask))
            
            if w_all_list:
                w_all = torch.cat(w_all_list).cpu().numpy()
                plt.figure()
                plt.hist(w_all, bins=50)
                plt.title(f'{prefix}Similarity Weights (w) - Global (No Diag)')
                plt.savefig(os.path.join(output_dir, f'{prefix}_sim_weights_global.png'))
                plt.close()
            
            # Anchor Sim Weights: Exclude self (index 0)
            if w_list[0].shape[0] > 1:
                w_anchor = w_list[0][0, 1:].cpu().numpy() # [B-1]
                plt.figure()
                plt.hist(w_anchor, bins=50)
                plt.title(f'{prefix}Similarity Weights (w) - Anchor (No Self)')
                plt.savefig(os.path.join(output_dir, f'{prefix}_sim_weights_anchor.png'))
                plt.close()

    # 2. Cue Dist
    if 'cue_dist' in m_data:
        cd_list = m_data['cue_dist']
        if len(cd_list) > 0:
            # Exclude diagonal (self-dist) for global histogram
            cd_all_list = []
            for x in cd_list:
                # x is (B, B)
                B = x.shape[0]
                if B > 1:
                    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
                    cd_all_list.append(torch.masked_select(x, mask))
            
            if cd_all_list:
                cd_all = torch.cat(cd_all_list).cpu().numpy()
                plt.figure()
                plt.hist(cd_all, bins=50)
                plt.title(f'{prefix}Cue Distance - Global (No Diag)')
                plt.savefig(os.path.join(output_dir, f'{prefix}_cue_dist_global.png'))
                plt.close()
            
            # Anchor Cue Dist: Exclude self (index 0)
            if cd_list[0].shape[0] > 1:
                cd_anchor = cd_list[0][0, 1:].cpu().numpy() # [B-1]
                plt.figure()
                plt.hist(cd_anchor, bins=50)
                plt.title(f'{prefix}Cue Distance - Anchor (No Self)')
                plt.savefig(os.path.join(output_dir, f'{prefix}_cue_dist_anchor.png'))
                plt.close()

def analyze(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    print(f"Using device: {device}")

    # 1. Dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = FixMicSigDataset(
        data_dir=args.data_path,
        fs=16000,
        load_anno=True,
        dataset_sz=None
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. Stats
    feature_stats = calculate_feature_stats(dataloader, device, fs=16000, max_samples=args.num_stat_samples)
    print("Feature stats calculated:", feature_stats.keys())
    
    # 3. Collect Weights and Diffs
    stft_module = STFT(win_len=512, win_shift_ratio=0.5, nfft=512).to(device)
    
    # Define Feature Sets
    # Original keys
    combined_keys = [
        # 'ild_mean', 
        'coherence_low', 'coherence_mid', 'coherence_high',
        'gcc_phat_max', 
        # 'hf_lf_ratio',
        'tdoa',
        'iacc_max',
        'decay_slope_median',
        'decay_slope_p10',
        'decay_slope_p90',
        # 'mod_low_high_ratio',
        # 'early_late_ratio_mean',
        # 'early_late_ratio_median'
    ]
    
    # Hierarchical keys (from model.py)
    doa_keys = ['tdoa']
    room_keys = [
        'coherence_low', 'coherence_mid', 'coherence_high', 
        'gcc_phat_max', 
        'iacc_max', 
        'decay_slope_median', 'decay_slope_p10', 'decay_slope_p90', 
        # 'mod_low_high_ratio', 
        # 'early_late_ratio_mean', 'early_late_ratio_median'
    ]
    
    analysis_sets = {
        'weight': combined_keys,
        'hierarchical_weight_doa': doa_keys,
        'hierarchical_weight_room': room_keys
    }
    
    all_weights = {k: [] for k in analysis_sets.keys()}
    hist_data = {k: {'sim_weights': [], 'cue_dist': []} for k in analysis_sets.keys()}
    
    acoustic_diffs = {
        'tdoa': [],
        't60': [],
        'c50': [],
        'drr': [],
        'abs': [] # Absorption if available
    }
    
    # Storage for correlation analysis
    all_feature_keys = [
        'ild_mean', 
        'ipd_cos_mean', 'ipd_sin_mean',
        'coherence_mean', 'coherence_var',
        'coherence_low', 'coherence_mid', 'coherence_high',
        'gcc_phat_max', 
        'hf_lf_ratio',
        'tdoa',
        'iacc_max',
        'decay_slope_median',
        'decay_slope_p10',
        'decay_slope_p90',
        'mod_low_high_ratio',
        'early_late_ratio_mean',
        'early_late_ratio_median'
    ]
    feature_keys_corr = all_feature_keys
    gt_keys_corr = ['TDOA', 'T60', 'DRR', 'C50', 'ABS']
    
    collected_features = {k: [] for k in feature_keys_corr}
    collected_gt = {k: [] for k in gt_keys_corr}
    
    print("Analyzing relationships...")
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=min(len(dataloader), args.num_analysis_batches)):
            if count >= args.num_analysis_batches:
                break
                
            sig, info, _ = batch
            sig = sig.to(device)
            
            # Compute Weights
            stft_out = stft_module(sig)
            stft_out = stft_out.permute(0, 3, 1, 2)
            features = spatial_features_from_stft(stft_out, sample_rate=16000, hop_length=256)
            
            # Calculate weights for each set
            for set_name, keys in analysis_sets.items():
                if set_name == 'hierarchical_weight_doa':
                    sigma = 0.4
                else:
                    sigma = 0.5 * np.sqrt(len(keys))
                if count == 0:
                    print(f"Set: {set_name}, Sigma: {sigma}")
                weights, dist = calculate_weights(features, feature_stats, keys=keys, sigma=sigma)
                
                # Store for histogram
                hist_data[set_name]['sim_weights'].append(weights.cpu())
                hist_data[set_name]['cue_dist'].append(dist.cpu())

                # For the weight vs diff plot, we essentially want to see if High Weight => Low Diff
                # We can store the weights and the GT diffs
                # We need the pairwise weights. 
                # This could be large: B*B pairs. 
                # Let's sample or just flatten everything? 
                # Doing B*B for every batch might be too much memory if we keep it all.
                # Let's keep it restricted.
                
                # Flatten weights and keep them only if we are also keeping the diffs?
                # Actually, we can just compute the correlation/scatter on the fly or accumulate 
                # a subset. 
                
                # For simplicity, let's accumulate a flattened sampling
                # We take the upper triangle to avoid duplicates and self-loops (dist 0)
                # But self-loops are important for "same sample" logic? 
                # Usually we care about positive vs negative pairs.
                
                # Use upper triangle indices to match acoustic diff pairs
                batch_size = weights.shape[0]
                indices = np.triu_indices(batch_size, k=1)
                w_flat = weights.cpu().numpy()[indices]
                
                # Check for NaNs and replace
                if np.isnan(w_flat).any():
                   w_flat = np.nan_to_num(w_flat, nan=0.0)
                    
                all_weights[set_name].append(w_flat)
            
            # Collect features for correlation
            for k in feature_keys_corr:
                if k in features:
                    collected_features[k].extend(features[k].cpu().numpy().flatten())
            
            # Collect GT for correlation
            for k in gt_keys_corr:
                if k in info:
                    val = info[k]
                    if isinstance(val, torch.Tensor):
                        val = val.cpu().numpy()
                    collected_gt[k].extend(np.reshape(val, -1))
            
            # Compute Acoustic Diffs
            available_keys = []
            if 'TDOA' in info: available_keys.append(('TDOA', 'tdoa'))
            if 'T60' in info: available_keys.append(('T60', 't60'))
            if 'C50' in info: available_keys.append(('C50', 'c50'))
            if 'DRR' in info: available_keys.append(('DRR', 'drr'))
            if 'ABS' in info: available_keys.append(('ABS', 'abs'))
            
            batch_size = sig.shape[0] # Re-use batch size from input
            indices = np.triu_indices(batch_size, k=1)
            
            for info_key, store_key in available_keys:
                vals = info[info_key] # (B, ...)
                if isinstance(vals, torch.Tensor):
                    vals = vals.cpu().numpy()
                
                # If vals has extra dims, flatten or squeeze
                if vals.ndim > 1:
                    vals = vals.squeeze()
                
                # Pairwise absolute difference
                # shape (B, 1) - (1, B) = (B, B)
                diff_mat = np.abs(vals[:, None] - vals[None, :])
                acoustic_diffs[store_key].extend(diff_mat[indices])
                
            count += 1
            
    # 4. Plotting
    print("Generating plots...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot Histograms
    print("Plotting histograms...")
    for set_name, data in hist_data.items():
        plot_histograms(data, args.output_dir, prefix=set_name)
    
    # Plot weights vs acoustic diffs for each set
    for weight_name, weights_list in all_weights.items():
        if len(weights_list) == 0:
            continue
        weights_arr = np.array(weights_list)
        
        for key, diffs in acoustic_diffs.items():
            if len(diffs) == 0:
                continue
                
            diffs_arr = np.array(diffs)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(diffs_arr, weights_arr, alpha=0.3, s=1)
            plt.xlabel(f"Difference in {key.upper()}")
            plt.ylabel("Model Weight")
            title_prefix = weight_name.replace('_', ' ').title()
            plt.title(f"{title_prefix} vs {key.upper()} Difference")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(args.output_dir, f"{weight_name}_vs_{key}.png"))
            plt.close()
        
    # 4. Correlation Analysis
    print("Calculating Pearson Correlation Coefficient matrix...")
    
    corr_feature_names = feature_keys_corr
    corr_gt_names = gt_keys_corr
    
    # Filter out empty collected lists
    active_feat_keys = [k for k in corr_feature_names if len(collected_features[k]) > 0]
    active_gt_keys = [k for k in corr_gt_names if len(collected_gt[k]) > 0]
    
    if active_feat_keys and active_gt_keys:
        # Check alignment
        n_samples = len(collected_features[active_feat_keys[0]])
        # Truncate to min length just in case, though they should be equal
        
        X = np.stack([collected_features[k][:n_samples] for k in active_feat_keys], axis=1) # (N, F)
        Y = np.stack([collected_gt[k][:n_samples] for k in active_gt_keys], axis=1) # (N, G)
        
        # We want correlation between each col of X and each col of Y
        # Result matrix: (F, G)
        corr_matrix = np.zeros((len(active_feat_keys), len(active_gt_keys)))
        
        for i, f_key in enumerate(active_feat_keys):
            for j, g_key in enumerate(active_gt_keys):
                # Pearson corr
                if np.std(X[:, i]) == 0 or np.std(Y[:, j]) == 0:
                    c = 0
                else:
                    c = np.corrcoef(X[:, i], Y[:, j])[0, 1]
                corr_matrix[i, j] = c
                
        print("Correlation Matrix:")
        print("Rows (Features):", active_feat_keys)
        print("Cols (GT):", active_gt_keys)
        print(corr_matrix)
        
        # Plotting
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Pearson Correlation')
        
        plt.xticks(np.arange(len(active_gt_keys)), active_gt_keys, rotation=45)
        plt.yticks(np.arange(len(active_feat_keys)), active_feat_keys)
        
        # Add text annotations
        for i in range(len(active_feat_keys)):
            for j in range(len(active_gt_keys)):
                val = corr_matrix[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                plt.text(j, i, f"{val:.2f}", 
                         ha="center", va="center", color=color)
                 
        plt.title("Correlation: Analytical Features vs GT Labels")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "feature_gt_correlation.png"))
        plt.close()
    else:
        print("Skipping correlation analysis due to missing data.")

    print(f"Analysis complete. Plots saved to {args.output_dir}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/storage/yotam/SAR-SSL/SAR-SSL/data/MicSig/simu/pretrain/", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", help="Directory to save plots")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_stat_samples", type=int, default=100, help="Number of batches for calculating stats")
    parser.add_argument("--num_analysis_batches", type=int, default=50, help="Number of batches to analyze")
    
    args = parser.parse_args()
    analyze(args)
