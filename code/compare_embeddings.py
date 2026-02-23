import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import json
import json
from pathlib import Path
import textwrap
from datetime import datetime


# Add current directory to path to allow imports
sys.path.append(os.getcwd())

import dataset as at_dataset
import model as at_model
import common.utils_module as at_module 

# Predefined names for specific timestamps (from gen_summary_table.py)
MODEL_NAME_MAP = {
        "12281023": "Baseline (CCSR)",
        "12281234": "Original + Feat Sim",
        "12281809": "Autoenc + Feat Sim",
        "12291627": "Autoenc + Feat Sim w. New Normalization",
        "12291841": "Autoenc + Feat Sim w. New Normalization * 4",
        "01041159": "Autoenc + Feat Sim w. Percentile Normalization",
        "01041536": "Autoenc + Feat Sim w. Mean+STD+Clip Normalization",
        "01051314": "Autoenc + Feat Sim w. New Features, temp 0.07",
        "01051625": "Autoenc + Feat Sim w. New Features, temp 0.1",
        "01061055": "Autoenc + Feat Sim w. New Features, temp 0.3",
        "01061410": "Autoenc + Pseudo-Hierarchical Feat Sim",
        "01081544": "Autoenc + Feat Sim wo. ILD and HF_LF, temp 0.1",
        "01111059": "Autoenc + Pseudo-Hierarchical Feat Sim wo. ILD and HF_LF, doa_temp 0.3, room_temp 0.07",
        "01111117": "Autoenc + Pseudo-Hierarchical Feat Sim, mono spectral,  wo. ILD and HF_LF, doa_temp 0.3, room_temp 0.07",
        "01122345": "Autoenc + Feat Sim w. Newest Features (decay slope etc), temp 0.1",
        "01131440": "Autoenc + Feat Sim w. Newest Features (decay slope etc), Hierarchical, sigma 1/1, temp 0.3/0.07",
        "01141100": "Autoenc + Feat Sim w. Newest Features (decay slope etc), Hierarchical, sigma 0.4/1.6, temp 0.3/0.07",
        "01151136": "Autoenc + Feat Sim w. Some Of The Newest Features (decay slope etc), Hierarchical, sigma 0.4/0.5*sqrt(dim), temp 0.1/0.07",
        "01161203": "Autoenc + Feat Sim w. Some Of The Newest Features (decay slope etc), Hierarchical, sigma 0.4/0.5*sqrt(dim), temp 0.4/0.2",
        "01190956": "Autoenc + Feat Sim w. Some Of The Newest Features (decay slope etc), OBT, Hierarchical, sigma 0.4/0.5*sqrt(dim), temp 0.4/0.2",
        "01191734": "Autoenc + Feat Sim w. Some Of The Newest Features (decay slope etc), sigma 0.5*sqrt(dim), temp 0.1",
        "01200950": "Autoenc + Feat Sim w. 5 features (coherences, gcc max, tdoa), sigma 1, temp 0.1",
        "01201345": "Autoenc + Feat Sim w. 5 features (coherences, gcc max, tdoa), sigma 1, temp 0.5",
        "01211117": "Autoenc + Feat Sim w. 5 features (coherences, gcc max, tdoa), BN, sigma 1, temp 0.5",
        "01212123": "Autoenc + Feat Sim w. 5 features (coherences, gcc max, tdoa), BN, sigma 1, temp 0.3",
        "01221548": "Autoenc + Feat Sim w. 5 features (coherences, gcc max, tdoa), BN, mono spectral, sigma 1, temp 0.5",
        "01231932": "Autoenc ONLY",
        "01261622": "CCSR + Feat Sim w. 5 features (coherences, gcc max, tdoa), BN, sigma 1, temp 0.5",
        "01271707": "CCSR + New Version of Feat Sim w. 5 features (coherences, gcc max, tdoa), BN, sigma 1, temp 0.07",
        "01272122": "CCSR + New Version of Feat Sim w. minus sim_max, 5 features (coherences, gcc max, tdoa), BN, sigma 1, temp 0.07",
        "01301633": "CCSR + New Version of Feat Sim w. 5 features (coherences, gcc max, tdoa), BN, sigma 0.1, temp 0.5",
        "01311252": "CCSR + New Version of Feat Sim w. 5 features (coherences, gcc max, tdoa), BN, sigma 1, temp 0.3",
        "02010901": "Ours (CCSR + SAC Loss)",
        "02011216": "CCSR + New Version of Feat Sim w. 5 features (coherences, gcc max, tdoa), BN, sigma 1, temp 0.5",
        "02021006": "AE + FeatSim",
        "02041034": "CCSR Run 2",
        "02041438": "No Pretrain",
        "02121359": "Only FeatSim"
    }

class VisDataset(at_dataset.FixMicSigDataset):
    """
    Dataset class mostly copied from visualize_embeddings.py to ensure compatibility.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure numerical sort so index i corresponds to file i.wav (if feasible)
        try:
            self.files.sort(key=lambda p: int(p.stem))
        except ValueError:
            print("Warning: Could not sort files numerically. Using default sort.")
            self.files.sort()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # Extract file index from filename
        file_path = self.files[idx]
        file_name = file_path.stem 
        try:
            file_idx = int(file_name)
        except ValueError:
            file_idx = idx 
            
        # Return annotations if present (FixMicSigDataset returns [sig, anno, idx] or [sig, dp, idx] etc)
        # We need data[1] which are annotations
        if len(data) > 2 and isinstance(data[1], dict):
             return data[0], data[1], file_idx
            
        return data[0], file_idx

def get_model_label(timestamp):
    return MODEL_NAME_MAP.get(timestamp, f"Model {timestamp}")

def load_and_extract(exp_dir, device, tasks):
    """
    Loads model and extracts embeddings + labels for specified tasks.
    """
    print(f"Loading from {exp_dir}...")
    
    # Configuration
    data_dir = '../data/MicSig/simu/pretest' # Using regressive data path as we need labels
    batch_size = 32
    
    # STFT parameters (standard)
    win_len = 512
    nfft = 512
    win_shift_ratio = 0.5
    fs = 16000
    T = 4.112 
    nf = nfft//2
    nt = int((T * fs - win_len*(1-win_shift_ratio)) / (win_len*win_shift_ratio))
    
    # Load Config
    config_path = os.path.join(exp_dir, 'config.json')
    spec_dembed = 512
    spat_dembed = 256
    spectral_enc_input = 'all'
    hierarchical = False

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                if isinstance(config_data, list) and len(config_data) > 0:
                    args_dict = config_data[0]
                    spec_dembed = args_dict.get('spec_dembed', 512)
                    spat_dembed = args_dict.get('spat_dembed', 256)
                    spectral_enc_input = args_dict.get('spectral_enc_input', 'all')
                    hierarchical = args_dict.get('hierarchical', False)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")

    # Initialize Model
    net = at_model.SARSSL(
        sig_shape=(nf, nt, 2, 2), 
        pretrain=True, 
        device=device,
        contrastive_learning=False,
        spec_dembed=spec_dembed,
        spat_dembed=spat_dembed,
        spectral_enc_input=spectral_enc_input,
        hierarchical=hierarchical
    )
    
    # Load Checkpoint
    checkpoint_path = os.path.join(exp_dir, 'best_model.tar')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
        
    net.load_state_dict(new_state_dict, strict=False)
    net.to(device)
    net.eval()
    
    # Dataset
    # Load 4000 random samples (fixed seed for consistency across models if we reset seed, 
    # but here we follow visualize_embeddings logic roughly)
    dataset = VisDataset(
        data_dir=data_dir,
        fs=fs,
        load_anno=True, 
        dataset_sz=None, 
        sort=False
    )
    
    # Use 4000 random samples
    np.random.seed(42) 
    target_indices = np.random.choice(len(dataset), 4000, replace=False).tolist()
    target_indices.sort()
    
    subset = torch.utils.data.Subset(dataset, target_indices)
    data_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Extraction
    embeddings_spat = []
    labels = {t: [] for t in tasks}
    
    stft_module = at_module.STFT(win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft).to(device)
    addbatch = at_module.AddChToBatch(ch_mode='M')
    
    print(f"  Extracting embeddings from {len(subset)} samples...")
    
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            if i % 10 == 0:
                 print(f"    Batch {i}/{len(data_loader)}", end='\r')
            # Expecting: mic_sig, annos, idx
            mic_sig_batch, annos_batch, _ = batch_data
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
            
            # Encoder
            data_in = x.permute(0, 2, 3, 4, 1) 
            vec_patch = net.patch_split(data_in) 
            nbatch, npatch = vec_patch.shape[0], vec_patch.shape[1]
            vec_patch_flat = vec_patch.reshape(nbatch, npatch, -1) 
            
            emb_spat = net.spat_encoder(vec_patch_flat)
            emb_spat_mean = torch.mean(emb_spat, dim=1) 
            
            embeddings_spat.append(emb_spat_mean.cpu().numpy())
            
            for t in tasks:
                if t in annos_batch:
                    labels[t].extend(annos_batch[t].numpy())
                else:
                    # Fallback or error if task missing? 
                    # Assuming standard tasks exist in annos
                    pass
    
    print("") # Newline after progress
    embeddings_spat = np.concatenate(embeddings_spat, axis=0)
    for t in tasks:
        labels[t] = np.array(labels[t])
        
    return embeddings_spat, labels

def main():
    parser = argparse.ArgumentParser(description='Compare Embeddings of Different Models')
    parser.add_argument('--m', nargs='+', required=True, help='List of model timestamps')
    parser.add_argument('--t', nargs='+', required=True, help='List of downstream tasks')
    parser.add_argument('--tsne_mode', type=int, choices=[1, 2], required=True, help='TSNE mode: 1 (utils style) or 2 (visualize_embeddings style)')
    
    args = parser.parse_args()
    
    models = args.m
    tasks = args.t
    tsne_mode = args.tsne_mode
    
    base_exp_dir = '../exp/pretrain'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # LaTeX Support (using mathtext, no external LaTeX required)
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm", # Computer Modern font for LaTeX look
    })
    
    # Prepare Plot
    n_models = len(models)
    n_tasks = len(tasks)
    
    fig, axes = plt.subplots(n_models, n_tasks, figsize=(5 * n_tasks, 5 * n_models), squeeze=False)
    
    for i, model_ts in enumerate(models):
        exp_dir = os.path.join(base_exp_dir, model_ts)
        model_name = get_model_label(model_ts)
        print(f"Processing {model_name} ({model_ts})...")
        
        if not os.path.exists(exp_dir):
            print(f"Error: {exp_dir} does not exist.")
            continue
            
        try:
            embeddings, labels_dict = load_and_extract(exp_dir, device, tasks)
            
            print(f"  Computing t-SNE for {model_ts}...")
            if tsne_mode == 1:
                # Mode 1: utils.vis_TSNE style
                # learning_rate=100, n_components=2 (defaults: init='random' implied by sklearn unless specified otherwise, but utils doesn't specify init)
                tsne = TSNE(n_components=2, learning_rate=100, random_state=42)
                vis_data = tsne.fit_transform(embeddings)
            else:
                # Mode 2: visualize_embeddings style
                # perplexity=5, n_iter=1000
                tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, random_state=42)
                vis_data = tsne.fit_transform(embeddings)
                
            # Plotting columns (Tasks)
            for j, task in enumerate(tasks):
                ax = axes[i, j]
                label_vals = labels_dict[task]
                
                # Check for NaNs
                if len(label_vals) == 0:
                    ax.text(0.5, 0.5, "No Data", ha='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # Prepare Plotting Params
                cmap = 'plasma' 
                
                # Task-specific overrides
                c_vals = label_vals
                clabel_text = ""
                
                if task == 'TDOA':
                    c_vals = label_vals * 1000 # Convert s to ms
                    clabel_text = " (ms)"
                elif task == 'T60':
                    clabel_text = " (s)"
                elif task == 'DRR':
                    clabel_text = " (dB)"

                # bigger dots
                s_size = 25

                scatter = ax.scatter(vis_data[:, 0], vis_data[:, 1], c=c_vals, s=s_size, marker='o', cmap=cmap, alpha=0.8)
                
                if i == 0:
                     # Use Task + Unit for title if present, otherwise just Task
                     title_text = task + clabel_text if task not in ['TDOA', 'T60', 'DRR'] else task
                     ax.set_title(title_text, fontsize=20, fontweight='bold')
                
                if j == 0:
                    ylabel = textwrap.fill(model_name, width=30)
                    ax.set_ylabel(ylabel, rotation=90, fontsize=20, fontweight='bold')

                # Colorbar with units
                cbar = plt.colorbar(scatter, ax=ax)
                if clabel_text:
                     cbar.set_label(clabel_text.strip(), rotation=90, fontsize=18)
                cbar.ax.tick_params(labelsize=18)
                
                ax.grid(True, alpha=0.3)
                ax.set_xticks([])
                ax.set_yticks([])
            
            print(f"  Finished {model_name}.")
                
        except Exception as e:
            print(f"Failed to process {model_ts}: {e}")
            import traceback
            traceback.print_exc()
            
    plt.tight_layout()
    models_suffix = "_".join(models)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"figures/compare_embeddings_{models_suffix}_mode_{tsne_mode}_{timestamp}.png"
    plt.savefig(output_filename)
    print(f"Saved comparison to {output_filename}")

if __name__ == '__main__':
    main()
