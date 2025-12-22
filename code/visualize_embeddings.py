
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
        data = super().__getitem__(idx)
        
        # Extract file index from filename
        file_path = self.files[idx]
        file_name = file_path.stem 
        try:
            file_idx = int(file_name)
        except ValueError:
            file_idx = idx 
            
        return data[0], file_idx


def run_visualization(exp_dir):
    print(f"Using experiment directory: {exp_dir}")
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
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
                    print(f"Loaded config: spec_dembed={spec_dembed}, spat_dembed={spat_dembed}, spectral_enc_input={spectral_enc_input}")
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
        spectral_enc_input=spectral_enc_input
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
        load_anno=False,
        dataset_sz=None, # Load all found files so we can pick indices freely
        sort=False
    )
    
    # Define specific indices to visualize
    # "different permutations of 10 rooms and 10 utterances"
    # Logic: Samples 0-9 (Room 0), 300-309 (Room 1), ..., 2700-2709 (Room 9)
    # Assumes idx maps to Room = idx // 300, Utt = idx % 300
    target_indices = []
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
    
    import common.utils_module as at_module
    stft_module = at_module.STFT(win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft).to(device)
    addbatch = at_module.AddChToBatch(ch_mode='M')
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for i, (mic_sig_batch, file_indices) in enumerate(data_loader):
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
                 emb_spec = net.spec_encoder(vec_patch_flat_spec)
            else:
                 emb_spec = net.spec_encoder(vec_patch_flat)
            emb_spat = net.spat_encoder(vec_patch_flat)
            
            emb_spec_mean = torch.mean(emb_spec, dim=1) 
            emb_spat_mean = torch.mean(emb_spat, dim=1) 
            
            embeddings_spec.append(emb_spec_mean.cpu().numpy())
            embeddings_spat.append(emb_spat_mean.cpu().numpy())
            
            indices = file_indices.numpy()
            rir_indices.extend(indices // samples_per_room)
            utt_indices.extend(indices % samples_per_room)
            
    embeddings_spec = np.concatenate(embeddings_spec, axis=0)
    embeddings_spat = np.concatenate(embeddings_spat, axis=0)
    rir_indices = np.array(rir_indices)
    utt_indices = np.array(utt_indices)
    
    print(f"Collected embeddings: Spec {embeddings_spec.shape}, Spat {embeddings_spat.shape}")
    
    # t-SNE
    print("Running t-SNE...")
    # Low perplexity because we have small clusters (10 points per group)
    tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, random_state=42)
    
    vis_spec = tsne.fit_transform(embeddings_spec)
    vis_spat = tsne.fit_transform(embeddings_spat)
    
    # Plotting
    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, 'embedding_vis.png')
    
    plt.figure(figsize=(14, 6))
    
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
            
        print(f"\nProcessing experiment: {t}")
        run_visualization(exp_dir)

if __name__ == '__main__':
    visualize_embeddings()
