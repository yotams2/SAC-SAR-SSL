import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, '/storage/yotam/SAR-SSL/SAR-SSL/code')
from opt import opt_pretrain
import dataset as at_dataset
from common.utils_module import STFT, spatial_features_from_stft

def load_random_features(num_samples=5000, batch_size=32, device='cuda'):
    print(f"Loading up to {num_samples} samples from simulated pretrain set...")
    sys.argv = ['search_sigma_entropy.py', '--pretrain', '--simu-exp']
    opts = opt_pretrain()
    args = opts.parse()
    dirs = opts.dir()
    
    fs = args.acoustic_setting['fs']
    dataset = at_dataset.FixMicSigDataset(
        data_dir=dirs['micsig_simu_pretrain'],
        fs=fs,
        load_anno=False,
        dataset_sz=num_samples,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers)
    
    do_stft = STFT(win_len=512, win_shift_ratio=0.5, nfft=512).to(device)
    
    feature_keys = ['coherence_low', 'coherence_mid', 'coherence_high', 'gcc_phat_max', 'tdoa']
    all_features = []
    
    loaded_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list) and len(batch) >= 2:
                waveform = batch[0]
            else:
                waveform = batch
                
            waveform = waveform.to(device)
            # AddChToBatch if needed? In SAR-SSL waveform is [B, N, C]. 
            # In FixMicSigDataset it returns [B, T, C].
            if waveform.dim() == 3: # [B, T, C]
                stft_out = do_stft(waveform) # [B, F, T, C]
                stft_out = stft_out.permute(0, 3, 1, 2) # [B, C, F, T]
                
                feats_dict = spatial_features_from_stft(stft_out, sample_rate=fs)
                
                batch_feats = []
                for k in feature_keys:
                    batch_feats.append(feats_dict[k].unsqueeze(1))
                batch_feats = torch.cat(batch_feats, dim=1) # [B, 5]
                
                all_features.append(batch_feats)
                loaded_count += batch_feats.size(0)
                
                if loaded_count % (batch_size * 10) < batch_size:
                    print(f"Extracted {loaded_count} / {num_samples}...")
                    
                if loaded_count >= num_samples:
                    break
            else:
                print(f"Skipping unexpected waveform shape: {waveform.shape}")
                
    all_features = torch.cat(all_features, dim=0)[:num_samples]
    
    # Global normalization across all samples
    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True)
    all_features = (all_features - mean) / (3.0 * std + 1e-6)
    all_features = all_features.clamp(-1.0, 1.0)
    
    print(f"Successfully extracted globally normalized features of shape: {all_features.shape}")
    return all_features, feature_keys

def search_sigma_entropy():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    N = 5000
    try:
        features, features_list = load_random_features(num_samples=N, device=device)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return
        
    print(f"Feature list: {features_list}")
    
    # We only have 1 group of 5 features
    groups = {'spatial_features': list(range(len(features_list)))}
    
    sigmas = np.logspace(-2, 1, 50)
    batch_sizes = [16, 32, 64, 128]
    
    results = {}
    
    out_dir = '/storage/yotam/SAR-SSL/SAR-SSL/metrics/sigma_search'
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(24, 6))
    if len(batch_sizes) == 1:
        axes = [axes]
        
    for idx, bs in enumerate(batch_sizes):
        print(f"\n{'='*50}\nEvaluating for Batch Size = {bs}\n{'='*50}")
        max_entropy = np.log(bs - 1)
        target_entropy = 0.625 * max_entropy
        print(f"Theoretical max entropy for N={bs}: {max_entropy:.4f}")
        
        bs_results = {g: [] for g in groups.keys()}
        recommendations = {}
        
        for g_name, indices in groups.items():
            g_feats = features[:, indices]
            
            entropies = []
            best_sigma = None
            min_diff_to_target = float('inf')
            
            for sigma in sigmas:
                batch_entropies = []
                for i in range(0, N - bs + 1, bs):
                    batch_feats = g_feats[i:i+bs]
                    D = torch.cdist(batch_feats, batch_feats, p=2)
                    off_diag = ~torch.eye(bs, dtype=torch.bool, device=device)
                    
                    W = torch.exp(- (D / sigma)**2)
                    W_masked = W * off_diag.float()
                    
                    W_sum = W_masked.sum(dim=1, keepdim=True) + 1e-8
                    W_norm = W_masked / W_sum
                    
                    H = - (W_norm * torch.log(W_norm + 1e-8)).sum(dim=1).mean().item()
                    batch_entropies.append(H)
                    
                mean_H = np.mean(batch_entropies)
                entropies.append(mean_H)
                
                diff = abs(mean_H - target_entropy)
                if diff < min_diff_to_target and 0.5 * max_entropy <= mean_H <= 0.75 * max_entropy:
                    min_diff_to_target = diff
                    best_sigma = sigma
                    
            if best_sigma is None:
                best_sigma = sigmas[np.argmin([abs(h - target_entropy) for h in entropies])]
                
            bs_results[g_name] = entropies
            recommendations[g_name] = best_sigma
            print(f"{g_name:<15} | Recommended sigma: {best_sigma:.4f}")
            
        results[bs] = {
            'entropies': bs_results,
            'recommendations': recommendations
        }
        
        ax = axes[idx]
        for g_name, ent in bs_results.items():
            ax.plot(sigmas, ent, label=f"{g_name} ({recommendations[g_name]:.3f})", marker='.', markersize=4)
        ax.axhline(y=max_entropy, color='k', linestyle='--', label=f"Max H ({max_entropy:.2f})")
        ax.axhline(y=target_entropy, color='gray', linestyle=':', label="Target 62.5%")
        ax.set_xscale('log')
        ax.set_xlabel('Sigma (Log Scale)')
        ax.set_ylabel('Mean Shannon Entropy')
        ax.set_title(f'Batch Size = {bs}')
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'sigma_entropy_search_multibatch.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    
    np.save(os.path.join(out_dir, 'search_results_multibatch.npy'), results)
    
    print("\n\n=== Final Optimal Sigma Dictionary ===")
    final_dict = {bs: results[bs]['recommendations'] for bs in batch_sizes}
    import pprint
    pprint.pprint(final_dict)

if __name__ == '__main__':
    search_sigma_entropy()
