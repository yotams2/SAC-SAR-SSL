
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import json

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

import dataset as at_dataset
import model as at_model
from common.utils import get_nparams



def get_latest_exp_dir(base_dir):
    exps = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    if not exps:
        return None
    exps.sort(key=lambda x: int(x))
    return os.path.join(base_dir, exps[-1])

class VisDataset(at_dataset.FixMicSigDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure numerical sort so index i corresponds to file i.wav (if feasible)
        # The files are named "0.wav", "1.wav", etc.
        try:
            self.files.sort(key=lambda p: int(p.stem))
        except ValueError:
            print("Warning: Could not sort files numerically. Using default sort.")
            self.files.sort()

    def __getitem__(self, idx):
        # reuse parent logic to get data
        # parent __getitem__ returns: [mic_sig] or [mic_sig, annos] or [mic_sig, dp_sig] etc + [idx] always at end?
        # Let's check FixMicSigDataset.__getitem__
        # It returns [mic_sig, (optional annos), (optional dp_sig), idx]
        
        data = super().__getitem__(idx)
        
        # Extract file index from filename
        file_path = self.files[idx]
        file_name = file_path.stem 
        try:
            file_idx = int(file_name)
        except ValueError:
            file_idx = idx 
            
        # Check if we have annotations (data length > 2 usually, since min is [sig, idx])
        # If load_anno=True, returns [sig, annos, idx] (length 3)
        # If load_dp=True, returns [sig, dp, idx] (length 3)
        # If both, returns [sig, annos, dp, idx] (length 4)
        
        # We need to return annotations if present. 
        # The original code assumed only [0] (sig) and file_idx was needed.
        # Now we might need data[1] if it's annos.
        
        if len(data) > 2 and isinstance(data[1], dict):
             return data[0], data[1], file_idx
            
        return data[0], file_idx


def run_visualization(exp_dir, regressive=False):
    print(f"Using experiment directory: {exp_dir}")
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if regressive:
        data_dir = '../data_cl/MicSig/simu/pretest'
    else:
        data_dir = '../data_cl/MicSig/simu/pretest_vis'

    batch_size = 32
    
    # Model parameters
    acoustic_setting = { 
        'nmic': 2, 'sound_speed': 343.0, 'fs': 16000, 'T': 4.112, 'mic_dist_range': [0.03, 0.20] 
    }
    
    # STFT parameters
    win_len = 512
    nfft = 512
    win_shift_ratio = 0.5
    fs = acoustic_setting['fs']
    T = 4.112 
    nf = nfft//2
    nt = int((T * fs - win_len*(1-win_shift_ratio)) / (win_len*win_shift_ratio))
    
    # Load Config from config.json if available
    config_path = os.path.join(exp_dir, 'config.json')
    spec_dembed = 512
    spat_dembed = 256
    spectral_enc_input = 'all'
    hierarchical = False

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                # config_data is a list, first element is args dict
                if isinstance(config_data, list) and len(config_data) > 0:
                    args_dict = config_data[0]
                    spec_dembed = args_dict.get('spec_dembed', 512)
                    spat_dembed = args_dict.get('spat_dembed', 256)
                    spectral_enc_input = args_dict.get('spectral_enc_input', 'all')
                    hierarchical = args_dict.get('hierarchical', False)
                    split_spatial_encoder = args_dict.get('split_spatial_encoder', False)
                    diff_feat_selection = args_dict.get('diff_feat_selection', False)
                    diff_feat_selection_lambda = args_dict.get('diff_feat_selection_lambda', 0.01)
                    print(f"Loaded config: spec_dembed={spec_dembed}, spat_dembed={spat_dembed}, spectral_enc_input={spectral_enc_input}, hierarchical={hierarchical}, split_spatial_encoder={split_spatial_encoder}, diff_feat={diff_feat_selection}")
                else:
                    print("Warning: config.json structure unexpected. Using defaults.")
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}. Using defaults.")
    else:
        print("Warning: config.json not found. Using defaults.")

    print(f"Initializing model with shape: nf={nf}, nt={nt}, spec_dembed={spec_dembed}, spat_dembed={spat_dembed}, spectral_enc_input={spectral_enc_input}")
    
    net = at_model.SARSSL(
        sig_shape=(nf, nt, 2, 2), 
        pretrain=True, 
        device=device,
        contrastive_learning=False,
        spec_dembed=spec_dembed,
        spat_dembed=spat_dembed,
        spectral_enc_input=spectral_enc_input,
        hierarchical=hierarchical,
        split_spatial_encoder=split_spatial_encoder,
        diff_feat_selection=diff_feat_selection,
        diff_feat_selection_lambda=diff_feat_selection_lambda
    )
    
    # Load Checkpoint
    checkpoint_path = os.path.join(exp_dir, 'best_model.tar')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
        
    net.load_state_dict(new_state_dict, strict=False)
    net.to(device)
    net.eval()
    
    # Initialize Dataset
    print(f"Loading data from {data_dir}")
    # Load all available files first, then we subset
    dataset = VisDataset(
        data_dir=data_dir,
        fs=fs,
        load_anno=regressive, # Load annotations if regressive mode
        dataset_sz=None, # Load all found files so we can pick indices freely
        sort=False
    )
    
    # Define specific indices to visualize
    target_indices = []
    
    if regressive:
        # Use 4000 random samples
        np.random.seed(42) # For reproducibility
        target_indices = np.random.choice(len(dataset), 4000, replace=False).tolist()
        target_indices.sort()
    else:
        # "different permutations of 10 rooms and 10 utterances"
        # Logic: Samples 0-9 (Room 0), 300-309 (Room 1), ..., 2700-2709 (Room 9)
        # Assumes idx maps to Room = idx // 300, Utt = idx % 300
        num_rooms = 10
        num_utts = 10
        samples_per_room = 10 
        
        for r in range(num_rooms):
            for u in range(num_utts):
                idx = r * samples_per_room + u
                target_indices.append(idx)
            
    print(f"Selected {len(target_indices)} samples for visualization.")
    
    # Create Subset
    subset = torch.utils.data.Subset(dataset, target_indices)
    
    data_loader = torch.utils.data.DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Extraction Loop
    embeddings_spec = []
    embeddings_spat = []
    rir_indices = []
    utt_indices = []
    
    # Store regression labels dynamically based on annos
    reg_labels = {}

    
    import common.utils_module as at_module
    stft_module = at_module.STFT(win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft).to(device)
    addbatch = at_module.AddChToBatch(ch_mode='M')
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            if regressive:
                 mic_sig_batch, annos, file_indices = batch_data
            else:
                 mic_sig_batch, file_indices = batch_data
            
            mic_sig_batch = mic_sig_batch.to(device)
            # Preprocessing 
            stft = stft_module(signal=mic_sig_batch) 
            stft = stft.permute(0, 3, 1, 2) 
            
            mag = torch.abs(stft[:, 0:1, :, :])
            mean_value = torch.mean(mag.reshape(mag.shape[0], -1), dim=1)
            mean_value = mean_value[:, None, None, None].expand(mag.shape)
            stft = stft / (mean_value + 1e-6)
            
            stft_rebatch = addbatch(stft)
            reim_rebatch = torch.view_as_real(stft_rebatch)
            reim_rebatch = reim_rebatch[:, :, 1:(nf+1), :, :]
            
            x = reim_rebatch
            
            # Use Encoders
            data = x.permute(0, 2, 3, 4, 1) 
            vec_patch = net.patch_split(data) 
            
            nbatch = vec_patch.shape[0]
            npatch = vec_patch.shape[1]
            vec_patch_flat = vec_patch.reshape(nbatch, npatch, -1) 
            
            if spectral_enc_input == 'mono':
                 # vec_patch: (nbatch, npatch, dpatch, nreim, nmic)
                 # Take first channel (mic 0)
                 vec_patch_flat_spec = vec_patch[:, :, :, :, 0].reshape(nbatch, npatch, -1)
                 emb_spec = net.spec_encoder(vec_patch_flat_spec)[0]
            else:
                 emb_spec = net.spec_encoder(vec_patch_flat)[0]
            emb_spat = net.spat_encoder(vec_patch_flat)[0]
            
            emb_spec_mean = torch.mean(emb_spec, dim=1) 
            emb_spat_mean = torch.mean(emb_spat, dim=1) 
            
            embeddings_spec.append(emb_spec_mean.float().cpu().numpy())
            embeddings_spat.append(emb_spat_mean.float().cpu().numpy())
            
            indices = file_indices.numpy()
            if not regressive: 
                rir_indices.extend(indices // samples_per_room)
                utt_indices.extend(indices % samples_per_room)
            else:
                for k in annos.keys():
                    if k not in reg_labels:
                        reg_labels[k] = []
                    reg_labels[k].extend(annos[k].numpy() if hasattr(annos[k], 'numpy') else annos[k])

            
    embeddings_spec = np.concatenate(embeddings_spec, axis=0)
    embeddings_spat = np.concatenate(embeddings_spat, axis=0)
    if not regressive:
        rir_indices = np.array(rir_indices)
        utt_indices = np.array(utt_indices)

    
    print(f"Collected embeddings: Spec {embeddings_spec.shape}, Spat {embeddings_spat.shape}")
    
    # t-SNE
    print("Running t-SNE...")
    # Low perplexity because we have small clusters (10 points per group)
    tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, random_state=42)
    
    vis_spec = tsne.fit_transform(embeddings_spec)
    
    if (hierarchical or split_spatial_encoder) and embeddings_spat.shape[1] == spat_dembed * 2:
        print("Handling Hierarchical/Split Embeddings...")
        # Split Spat into DOA and Room
        dim = spat_dembed
        embeddings_doa = embeddings_spat[:, :dim]
        embeddings_room = embeddings_spat[:, dim:]
        
        vis_doa = tsne.fit_transform(embeddings_doa)
        vis_room = tsne.fit_transform(embeddings_room)
        vis_spat = None # Not used
        
    else:
        vis_spat = tsne.fit_transform(embeddings_spat)
        vis_doa = None
        vis_room = None
    
    # Plotting
    os.makedirs(exp_dir, exist_ok=True)
    if regressive:
        os.makedirs(os.path.join(exp_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'spectral_metrics'), exist_ok=True)
        save_path = os.path.join(exp_dir, 'metrics', 'embedding_vis.png')
    else:
        save_path = os.path.join(exp_dir, 'embedding_vis.png')
    
    if (hierarchical or split_spatial_encoder) and vis_doa is not None:
         # Plot layout for hierarchical: Spec, DOA, Room
         
         if regressive:
             # 2x5 grid. Top row: DOA, Bottom row: Room
             plt.figure(figsize=(25, 10))
             metrics = [m for m in ['TDOA', 'T60', 'C50', 'ABS', 'DRR'] if m in reg_labels]
             
             for i, metric in enumerate(metrics):
                 vals = np.array(reg_labels[metric])
                 
                 # DOA Row (Top)
                 plt.subplot(2, 5, i+1)
                 scatter_doa = plt.scatter(vis_doa[:, 0], vis_doa[:, 1], alpha=0.8, s=20, c=vals, cmap='viridis')
                 plt.title(f'DOA Encoder - {metric}')
                 plt.colorbar(scatter_doa, label=metric)
                 plt.grid(True, alpha=0.3)
                 
                 # Room Row (Bottom)
                 plt.subplot(2, 5, i+6)
                 scatter_room = plt.scatter(vis_room[:, 0], vis_room[:, 1], alpha=0.8, s=20, c=vals, cmap='viridis')
                 plt.title(f'Room Encoder - {metric}')
                 plt.colorbar(scatter_room, label=metric)
                 plt.grid(True, alpha=0.3)
         
         else:
             plt.figure(figsize=(18, 6))
             
             # 1. Spec
             plt.subplot(1, 3, 1)
             scatter_spec = plt.scatter(vis_spec[:, 0], vis_spec[:, 1], alpha=0.8, s=30, c=utt_indices, cmap='tab10')
             plt.title('Spectral Encoder Embeddings\n(Color: Utterance Index)')
             plt.colorbar(scatter_spec, ticks=range(10), label='Utterance Index')
             plt.grid(True, alpha=0.3)
             
             # 2. DOA
             plt.subplot(1, 3, 2)
             # In ideal hierarchical, DOA embeds primarily capture TDOA/Loc (RIR Index partly, but maybe better correlated with TDOA metric if regressive)
             # For classification viz (RIR index), DOA might cluster by room (since room determines mic pos/src pos somewhat) or just be confused if positions vary.
             # Let's stick to Room Index for color for now as per original.
             scatter_doa = plt.scatter(vis_doa[:, 0], vis_doa[:, 1], alpha=0.8, s=30, c=rir_indices, cmap='tab10')
             plt.title('Spatial Encoder (DOA)\n(Color: Room/RIR Index)')
             plt.colorbar(scatter_doa, ticks=range(10), label='Room Index')
             plt.grid(True, alpha=0.3)
             
             # 3. Room
             plt.subplot(1, 3, 3)
             scatter_room = plt.scatter(vis_room[:, 0], vis_room[:, 1], alpha=0.8, s=30, c=rir_indices, cmap='tab10')
             plt.title('Spatial Encoder (Room)\n(Color: Room/RIR Index)')
             plt.colorbar(scatter_room, ticks=range(10), label='Room Index')
             plt.grid(True, alpha=0.3)

    else:
        # Original Plotting Logic
        plt.figure(figsize=(18, 10) if regressive else (14, 6))
        
        if regressive:
            # 5 plots for ['TDOA', 'T60', 'C50', 'ABS', 'DRR']
            # Only spatial embeddings
            metrics = [m for m in ['TDOA', 'T60', 'C50', 'ABS', 'DRR'] if m in reg_labels]
            for i, metric in enumerate(metrics):
                plt.subplot(2, 3, i+1)
                vals = np.array(reg_labels[metric])
                
                # Scatter plot with continuous color
                scatter = plt.scatter(vis_spat[:, 0], vis_spat[:, 1], alpha=0.8, s=30, c=vals, cmap='viridis')
                plt.title(f'Spatial Encoder - {metric}')
                plt.colorbar(scatter, label=metric)
                plt.grid(True, alpha=0.3)
                
        else:
            # Spectral Plot - Color by Utterance Index
            # Use discrete colormap (tab10)
            plt.subplot(1, 2, 1)
            scatter_spec = plt.scatter(vis_spec[:, 0], vis_spec[:, 1], alpha=0.8, s=30, c=utt_indices, cmap='tab10')
            plt.title('Spectral Encoder Embeddings\n(Color: Utterance Index 0-9)')
            plt.colorbar(scatter_spec, ticks=range(10), label='Utterance Index')
            plt.grid(True, alpha=0.3)
            
            # Spatial Plot - Color by RIR Index
            plt.subplot(1, 2, 2)
            scatter_spat = plt.scatter(vis_spat[:, 0], vis_spat[:, 1], alpha=0.8, s=30, c=rir_indices, cmap='tab10')
            plt.title('Spatial Encoder Embeddings\n(Color: Room/RIR Index 0-9)')
            plt.colorbar(scatter_spat, ticks=range(10), label='Room Index')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()

    # Generate Spectral Embeddings Plot in Regressive Mode
    if regressive:
        spectral_metrics = ['src_idx', 'SNR']
        valid_spec_metrics = [m for m in spectral_metrics if m in reg_labels]
        if valid_spec_metrics:
            plt.figure(figsize=(6 * len(valid_spec_metrics), 5))
            for i, metric in enumerate(valid_spec_metrics):
                plt.subplot(1, len(valid_spec_metrics), i+1)
                vals = np.array(reg_labels[metric])
                # src_idx is better with discrete colormap, SNR with continuous
                cmap = 'tab10' if metric == 'src_idx' else 'viridis'
                scatter = plt.scatter(vis_spec[:, 0], vis_spec[:, 1], alpha=0.8, s=30, c=vals, cmap=cmap)
                plt.title(f'Spectral Encoder - {metric}')
                plt.colorbar(scatter, label=metric)
                plt.grid(True, alpha=0.3)
            
            spectral_save_path = os.path.join(exp_dir, 'spectral_metrics', 'spectral_embedding_vis.png')
            plt.tight_layout()
            plt.savefig(spectral_save_path)
            print(f"Saved spectral visualization to {spectral_save_path}")
            plt.close()

    # ---------------------------------------------------------
    # Correlation Analysis (New)
    # ---------------------------------------------------------
    if regressive:
        print("\nRunning Correlation Analysis...")
        
        # Determine representations for Correlation Analysis
        embeddings_to_analyze = {}
        if (hierarchical or split_spatial_encoder) and embeddings_spat.shape[1] == spat_dembed * 2:
            dim = spat_dembed
            embeddings_to_analyze['DOA'] = embeddings_spat[:, :dim]
            embeddings_to_analyze['Room'] = embeddings_spat[:, dim:]
        else:
            embeddings_to_analyze['Spatial'] = embeddings_spat
            
        for enc_name, X in embeddings_to_analyze.items():
            print(f"Running for Encoder: {enc_name}")
            
            # Y: Labels (N, 5)
            # Order: TDOA, T60, C50, ABS, DRR
            metrics = [m for m in ['TDOA', 'T60', 'C50', 'ABS', 'DRR'] if m in reg_labels]
            Y = np.stack([np.array(reg_labels[m]) for m in metrics], axis=1)
            
            print(f"Data shapes - {enc_name} Embeddings: {X.shape}, Labels: {Y.shape}")
            
            n_features = X.shape[1]
            n_labels = Y.shape[1]
            
            # Calculate Pearson Correlation
            # We want matrix (Features, Labels) - (D, 5)
            corr_matrix = np.zeros((n_features, n_labels))
            
            # Normalize to avoid division by zero issues logic inside loops or efficiently
            # Center data
            X_centered = X - X.mean(axis=0)
            Y_centered = Y - Y.mean(axis=0)
            
            # Std dev
            X_std = X.std(axis=0)
            Y_std = Y.std(axis=0)
            
            # Avoid division by zero
            X_std[X_std == 0] = 1e-8
            Y_std[Y_std == 0] = 1e-8
            
            # Fast correlation: Covariance / (std_x * std_y)
            # Cov = (X_c.T @ Y_c) / N
            N = X.shape[0]
            cov = (X_centered.T @ Y_centered) / N
            corr_matrix = cov / (X_std[:, None] @ Y_std[None, :])
            
            # Save raw matrix
            matrix_path = os.path.join(exp_dir, 'metrics', f'correlation_matrix_{enc_name}.npy')
            np.save(matrix_path, corr_matrix)
            print(f"Saved correlation matrix to {matrix_path}")
            
            # 1. Bar Chart: Max Absolute Correlation per Label
            max_abs_corr = np.max(np.abs(corr_matrix), axis=0) # (5,)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(metrics, max_abs_corr, color='skyblue')
            plt.ylim(0, 1.0)
            plt.title(f'Maximum Absolute Correlation with {enc_name} Features')
            plt.ylabel('Max |Pearson Correlation|')
        
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2f}',
                         ha='center', va='bottom')
                
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(exp_dir, 'metrics', f'correlation_max_bar_{enc_name}.png'))
            plt.close()
            
            # 2. Heatmap: Top Correlated Dimensions vs Labels
            # We select top K dimensions for each label, take the union, and plot
            top_k = 5
            top_dims = set()
            for i in range(n_labels):
                # Sort by abs correlation
                sorted_indices = np.argsort(np.abs(corr_matrix[:, i]))[::-1]
                top_dims.update(sorted_indices[:top_k])
                
            top_dims = sorted(list(top_dims))
            print(f"Selected {len(top_dims)} unique dimensions for heatmap (Top-{top_k} per label).")
            
            if len(top_dims) > 50:
                print("Warning: Too many dimensions for clear heatmap, truncating to top 50 by max overall correlation.")
                # Fallback: sort top_dims by their max correlation across any label
                top_dims.sort(key=lambda d: np.max(np.abs(corr_matrix[d, :])), reverse=True)
                top_dims = top_dims[:50]
                top_dims.sort() # Resort indices for sanity
                
            sub_matrix = corr_matrix[top_dims, :] # (K, 5)
            
            plt.figure(figsize=(8, max(6, len(top_dims)*0.3)))
            plt.imshow(sub_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(label='Pearson Correlation')
            
            plt.xticks(range(n_labels), metrics)
            plt.yticks(range(len(top_dims)), top_dims)
            plt.ylabel('Embedding Dimension Index')
            plt.title(f'Correlation Heatmap (Top Dimensions, {enc_name})')
            
            # Annotate if small enough
            if len(top_dims) < 30:
                for i in range(len(top_dims)):
                    for j in range(n_labels):
                        val = sub_matrix[i, j]
                        color = "white" if abs(val) > 0.5 else "black"
                        plt.text(j, i, f"{val:.2f}", 
                                 ha="center", va="center", color=color, fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, 'metrics', f'correlation_heatmap_{enc_name}.png'))
            plt.close()


def visualize_embeddings():
    parser = argparse.ArgumentParser(description='Visualize Embeddings')
    parser.add_argument('-t', '--time', nargs='+', default=None, help='List of timestamps of the models to use. If None, uses latest.')
    args = parser.parse_args()

    base_exp_dir = '../exp/pretrain'
    
    if args.time:
        timestamps = args.time
    else:
        # Default behavior: run on latest only
        latest_exp = get_latest_exp_dir(base_exp_dir)
        if latest_exp:
            timestamps = [os.path.basename(latest_exp)]
        else:
            print("No experiments found in", base_exp_dir)
            return

    for t in timestamps:
        exp_dir = os.path.join(base_exp_dir, t)
        if not os.path.exists(exp_dir):
            print(f"Experiment directory not found: {exp_dir}")
            continue
            
        print(f"\nProcessing experiment: {t} (Non-Regressive Mode)")
        run_visualization(exp_dir, regressive=False)
        
        print(f"\nProcessing experiment: {t} (Regressive Mode)")
        run_visualization(exp_dir, regressive=True)

if __name__ == '__main__':
    visualize_embeddings()
