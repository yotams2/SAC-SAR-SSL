import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from model import SARSSL
from learner import STFTLearner

def generate_pairs(csv_path, num_positive=1000, num_negative=1000):
    """
    Generates positive and negative audio pairs for verification.
    """
    print(f"Reading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Group by Micro_Class_ID
    classes = df['Micro_Class_ID'].unique()
    class_to_files = {cls: df[df['Micro_Class_ID'] == cls]['File_Path'].tolist() for cls in classes}
    
    # Generate Positive Pairs
    pos_pairs = []
    print(f"Generating {num_positive} positive pairs...")
    # Only keep classes that have at least 2 files
    valid_pos_classes = [cls for cls in classes if len(class_to_files[cls]) >= 2]
    
    while len(pos_pairs) < num_positive:
        cls = random.choice(valid_pos_classes)
        file1, file2 = random.sample(class_to_files[cls], 2)
        pos_pairs.append((file1, file2, 1))
        
    # Generate Negative Pairs
    neg_pairs = []
    print(f"Generating {num_negative} negative pairs...")
    while len(neg_pairs) < num_negative:
        cls1, cls2 = random.sample(list(classes), 2)
        file1 = random.choice(class_to_files[cls1])
        file2 = random.choice(class_to_files[cls2])
        neg_pairs.append((file1, file2, 0))
        
    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)
    return all_pairs


class VerificationPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        file1, file2, label = self.pairs[idx]
        
        sig1, fs1 = sf.read(file1)
        sig2, fs2 = sf.read(file2)
        
        return sig1.astype(np.float32), sig2.astype(np.float32), label, file1, file2


class VerificationModel(nn.Module):
    def __init__(self, spec_dembed=512, spat_dembed=256, downstream_embed='spat'):
        super().__init__()
        
        self.encoder = SARSSL(
            sig_shape=(256, 249, 2, 2), 
            pretrain=False, 
            downstream_token='all', 
            downstream_head='mlp', 
            downstream_embed=downstream_embed, 
            spec_dembed=spec_dembed, 
            spat_dembed=spat_dembed,
            spectral_enc_input='all',
            hierarchical=False,
            split_spatial_encoder=False
        )
        
        self.stft_learner = STFTLearner(
            model=None, 
            win_len=512, 
            win_shift_ratio=0.5, 
            nfft=512, 
            fre_used_ratio=1, 
            fs=16000, 
            task='TDOA', 
            ch_mode='M'
        )
        # Set a default device for stft_learner; will be updated in forward() if model moves
        self.stft_learner.device = 'cpu'
        
        self.cos_sim = nn.CosineSimilarity(dim=1)
        
    def get_embedding(self, x):
        """
        x: raw audio (B, 64000, 2)
        """
        # Keep stft_learner device in sync with the model's device
        self.stft_learner.device = next(self.parameters()).device
        
        preprocessed = self.stft_learner.data_preprocess(x, None)
        in_batch = preprocessed[0]
        
        _, pooled_embed, _ = self.encoder(in_batch)
        return pooled_embed
        
    def forward(self, x1, x2):
        emb1 = self.get_embedding(x1)
        emb2 = self.get_embedding(x2)
        return self.cos_sim(emb1, emb2)


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configure output isolation per pretrained model timestamp and run timestamp
    from datetime import datetime as _dt
    import shlex as _shlex
    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    exp_dir = work_dir + '/SAR-SSL/' + args.exp_dir
    run_timestamp = _dt.now().strftime('%m%d%H%M')
    save_dir = os.path.join(exp_dir, 'room_verification', args.time, run_timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    csv_path = os.path.join(args.data_dir, 'metadata_test.csv')
    pairs = generate_pairs(csv_path, args.num_pos, args.num_neg)
    
    dataset = VerificationPairDataset(pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    downstream_embed = getattr(args, 'downstream_embed', 'spat')
    model = VerificationModel(downstream_embed=downstream_embed)
    
    # Load pretrained or finetuned weights
    if getattr(args, 'finetuned_model_dir', None):
        best_model_path = os.path.join(args.finetuned_model_dir, 'best_classifier.pth')
        if os.path.exists(best_model_path):
            print(f"Loading finetuned downstream weights from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location='cpu')
            
            # The finetuned model (ClassifierWrapper) state_dict has keys like 'encoder.spec_encoder...'
            # We want to load these into VerificationModel which also has 'encoder' as an attribute.
            # So the keys match perfectly.
            model.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded finetuned encoder parameters.")
        else:
            print(f"WARNING: Finetuned weights not found at {best_model_path}.")
    else:
        best_model_path = exp_dir + '/pretrain/' + args.time + '/best_model.tar'
        
        if os.path.exists(best_model_path):
            print(f"Loading zero-shot pretrained weights from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location='cpu')
            pretrained_dict = checkpoint['model']
            
            # Strip 'module.' prefix added by DataParallel when checkpoint was saved from multi-GPU training
            cleaned_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in pretrained_dict.items()}
            
            # Only load keys that exist in the downstream encoder (ignores decoder/loss-head weights)
            encoder_dict = model.encoder.state_dict()
            matched_dict = {k: v for k, v in cleaned_dict.items() if k in encoder_dict}
            
            print(f"Matched {len(matched_dict)} keys out of {len(encoder_dict)} encoder parameters.")
            assert len(matched_dict) > 0, "Failed to match any keys from pretrained checkpoint!"
            
            encoder_dict.update(matched_dict)
            model.encoder.load_state_dict(encoder_dict)
        else:
            print(f"WARNING: Pretrained weights not found at {best_model_path}.")
        
    model = model.to(device)
    model.eval()
    
    all_scores = []
    all_labels = []
    all_files1 = []
    all_files2 = []
    
    with torch.no_grad():
        for sig1, sig2, labels, files1, files2 in tqdm(loader, desc="Evaluating Pairs"):
            sig1, sig2 = sig1.to(device), sig2.to(device)
            scores = model(sig1, sig2)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_files1.extend(files1)
            all_files2.extend(files2)
            
    # Metrics
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER: find the operating point where FPR ≈ FNR, then average the two
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    
    print(f"\n--- Results ---")
    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")
    
    # Plot ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f}, EER = {eer:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Zero-Shot Room Verification')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {roc_path}")

    # Comprehensive Debug Output Generation
    debug_dir = os.path.join(save_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    optimal_threshold = thresholds[eer_threshold_idx]
    
    # Calculate parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    # 1. Overall verification report with complete self-contained context
    # Build a copy-pasteable command string
    _cmd_parts = ['python eval_verification.py']
    for _k, _v in vars(args).items():
        _cmd_parts.append(f'--{_k} {_shlex.quote(str(_v))}')
    reproduced_command = ' '.join(_cmd_parts)

    report = {
        'task': 'Zero-Shot Acoustic Environment Verification',
        'command': reproduced_command,
        'execution_args': vars(args),
        'model_context': {
            'architecture_summary': str(model),
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params
        },
        'dataset_context': {
            'evaluated_pairs': len(all_labels),
            'positive_pairs_requested': args.num_pos,
            'negative_pairs_requested': args.num_neg
        },
        'verification_results': {
            'auc': float(roc_auc),
            'eer': float(eer),
            'optimal_cosine_threshold': float(optimal_threshold)
        }
    }
    with open(os.path.join(debug_dir, 'verification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    # 2. Score statistics for true positive vs true negative pairs
    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    
    pos_scores = scores_arr[labels_arr == 1]
    neg_scores = scores_arr[labels_arr == 0]
    
    stats = {
        'positive_pairs': {
            'count': int(len(pos_scores)),
            'mean': float(np.mean(pos_scores)) if len(pos_scores) > 0 else 0.0,
            'std': float(np.std(pos_scores)) if len(pos_scores) > 0 else 0.0,
            'min': float(np.min(pos_scores)) if len(pos_scores) > 0 else 0.0,
            'q25': float(np.percentile(pos_scores, 25)) if len(pos_scores) > 0 else 0.0,
            'median': float(np.median(pos_scores)) if len(pos_scores) > 0 else 0.0,
            'q75': float(np.percentile(pos_scores, 75)) if len(pos_scores) > 0 else 0.0,
            'max': float(np.max(pos_scores)) if len(pos_scores) > 0 else 0.0
        },
        'negative_pairs': {
            'count': int(len(neg_scores)),
            'mean': float(np.mean(neg_scores)) if len(neg_scores) > 0 else 0.0,
            'std': float(np.std(neg_scores)) if len(neg_scores) > 0 else 0.0,
            'min': float(np.min(neg_scores)) if len(neg_scores) > 0 else 0.0,
            'q25': float(np.percentile(neg_scores, 25)) if len(neg_scores) > 0 else 0.0,
            'median': float(np.median(neg_scores)) if len(neg_scores) > 0 else 0.0,
            'q75': float(np.percentile(neg_scores, 75)) if len(neg_scores) > 0 else 0.0,
            'max': float(np.max(neg_scores)) if len(neg_scores) > 0 else 0.0
        }
    }
    with open(os.path.join(debug_dir, 'score_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    # 3. False negatives sample (Positive pairs with lowest cosine similarity)
    pos_indices = np.where(labels_arr == 1)[0]
    # Sort positive indices by score ascending
    sorted_pos_indices = pos_indices[np.argsort(scores_arr[pos_indices])]
    
    false_negatives = []
    for idx in sorted_pos_indices[:50]:
        false_negatives.append({
            'file1': all_files1[idx],
            'file2': all_files2[idx],
            'cosine_similarity': float(scores_arr[idx])
        })
    with open(os.path.join(debug_dir, 'false_negatives_sample.json'), 'w') as f:
        json.dump(false_negatives, f, indent=4)
        
    # 4. False positives sample (Negative pairs with highest cosine similarity)
    neg_indices = np.where(labels_arr == 0)[0]
    # Sort negative indices by score descending
    sorted_neg_indices = neg_indices[np.argsort(scores_arr[neg_indices])[::-1]]
    
    false_positives = []
    for idx in sorted_neg_indices[:50]:
        false_positives.append({
            'file1': all_files1[idx],
            'file2': all_files2[idx],
            'cosine_similarity': float(scores_arr[idx])
        })
    with open(os.path.join(debug_dir, 'false_positives_sample.json'), 'w') as f:
        json.dump(false_positives, f, indent=4)
        
    print(f"Comprehensive verification debug files saved to {debug_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Zero-Shot Room Verification')
    parser.add_argument('--data_dir', type=str, default='../../data_room_classification', help='Directory containing metadata CSVs and wavs')
    parser.add_argument('--work_dir', type=str, default=os.path.abspath(os.path.join(os.getcwd(), '../../..')), help='Work directory')
    parser.add_argument('--exp_dir', type=str, default='exp', help='Experiment directory name')
    parser.add_argument('--time', type=str, default='', help='Time string for the pretrained model checkpoint (e.g. "0711_1512")')
    parser.add_argument('--save_dir', type=str, default='exp/room_verification/', help='Directory to save verification plots')
    parser.add_argument('--num_pos', type=int, default=1000, help='Number of positive pairs')
    parser.add_argument('--num_neg', type=int, default=1000, help='Number of negative pairs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--downstream_embed', type=str, choices=['spat', 'spec_spat'], default='spat', help='Which embedding to extract from encoder')
    parser.add_argument('--finetuned_model_dir', type=str, default=None, help='Path to a finetuned model directory (e.g., from train_classifier.py) to evaluate instead of the pretrained model')
    
    args = parser.parse_args()
    
    if not args.finetuned_model_dir and not args.time:
        parser.error("You must provide either --time (for zero-shot pretrained evaluation) or --finetuned_model_dir (for downstream fine-tuned evaluation).")
        
    if args.finetuned_model_dir:
        report_path = os.path.join(args.finetuned_model_dir, 'debug', 'classification_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
            saved_embed = report.get('execution_args', {}).get('downstream_embed')
            if saved_embed:
                print(f"Auto-detected downstream_embed='{saved_embed}' from finetuned model configuration.")
                args.downstream_embed = saved_embed
                
    evaluate(args)
