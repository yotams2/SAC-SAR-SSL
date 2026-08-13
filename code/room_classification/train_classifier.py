import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import itertools

from model import SARSSL
from learner import STFTLearner

def get_room_classes():
    classes = []
    unique_rooms = set()
    
    # Small Rooms
    widths = np.arange(1.5, 3.51, 1.0)
    depths = np.arange(2.5, 4.51, 1.0)
    heights = np.arange(2.5, 3.01, 0.5)
    for w, d, h in itertools.product(widths, depths, heights):
        shape = (min(w, d), max(w, d), h)
        if shape not in unique_rooms:
            unique_rooms.add(shape)
            classes.append(('Small', (float(w), float(d), float(h))))
            
    # Large Rooms
    widths = np.arange(6.0, 13.01, 1.0)
    depths = np.arange(6.0, 12.01, 2.0)
    heights = np.arange(2.5, 3.51, 1.0)
    for w, d, h in itertools.product(widths, depths, heights):
        shape = (min(w, d), max(w, d), h)
        if shape not in unique_rooms:
            unique_rooms.add(shape)
            classes.append(('Large', (float(w), float(d), float(h))))
            
    # Halls
    widths = np.arange(1.0, 3.01, 1.0)
    depths = np.arange(7.0, 13.01, 1.0)
    heights = np.arange(2.5, 3.51, 1.0)
    for w, d, h in itertools.product(widths, depths, heights):
        shape = (min(w, d), max(w, d), h)
        if shape not in unique_rooms:
            unique_rooms.add(shape)
            classes.append(('Hall', (float(w), float(d), float(h))))
        
    return classes

class RoomClassificationDataset(Dataset):
    def __init__(self, data_dir, split, mode='micro'):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.mode = mode
        self.csv_path = os.path.join(data_dir, f'metadata_{split}.csv')
        self.df = pd.read_csv(self.csv_path)
        self.room_classes = get_room_classes()
        self.macro_to_id = {'Small': 0, 'Large': 1, 'Hall': 2}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row['File_Path']
        micro_class_id = int(row['Micro_Class_ID'])
        
        mic_sig, fs = sf.read(wav_path)
        # Expected shape from generation is (64000, 2)
        mic_sig = mic_sig.astype(np.float32)
        
        # Data Augmentation: Random L/R Channel Swapping
        if self.split == 'train' and np.random.rand() > 0.5:
            mic_sig = mic_sig[:, ::-1].copy()
        
        if self.mode == 'macro':
            macro_label = self.room_classes[micro_class_id][0]
            target_id = self.macro_to_id[macro_label]
        else:
            target_id = micro_class_id
            
        return mic_sig, target_id, micro_class_id, wav_path


class ClassifierWrapper(nn.Module):
    def __init__(self, num_classes=110, spec_dembed=512, spat_dembed=256, classifier_hidden=0, downstream_embed='spat'):
        super().__init__()
        self.downstream_embed = downstream_embed
        
        # Initialize SARSSL model with exact parameters as run_downstream.py
        # Setting pretrain=False to get downstream spatial encoder
        self.encoder = SARSSL(
            sig_shape=(256, 249, 2, 2), # Standard STFT shape for 4s audio at 16kHz
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
        
        # We need the STFT Learner to handle data preprocessing
        # It's not a nn.Module parameter, just a utility class
        self.stft_learner = STFTLearner(
            model=None, # Dummy model since we just need the preprocessor
            win_len=512, 
            win_shift_ratio=0.5, 
            nfft=512, 
            fre_used_ratio=1, 
            fs=16000, 
            task='TDOA', # Dummy task
            ch_mode='M'
        )
        # Set a default device; will be kept in sync with the model in forward()
        self.stft_learner.device = 'cpu'
        
        embed_dim = spat_dembed if downstream_embed == 'spat' else (spec_dembed + spat_dembed)
        
        # Classification head: single linear (classifier_hidden=0) or 2-layer MLP
        if classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, classifier_hidden),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(classifier_hidden, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes)
            )
        
    def forward(self, x):
        """
        x: raw audio (B, 64000, 2)
        """
        # Keep stft_learner in sync with the model's current device
        self.stft_learner.device = next(self.parameters()).device
        
        # STFTLearner's data_preprocess expects (B, 64000, 2) -> returns list where [0] is in_batch
        preprocessed = self.stft_learner.data_preprocess(x, None)
        in_batch = preprocessed[0]
        
        # Data Augmentation: SpecAugment (Time and Frequency Masking)
        if self.training:
            B, nmic, nf, nt, nreim = in_batch.shape
            # Frequency masking
            for _ in range(2):
                f_width = torch.randint(0, 20, (1,)).item()
                if f_width > 0:
                    f0 = torch.randint(0, nf - f_width, (1,)).item()
                    in_batch[:, :, f0:f0+f_width, :, :] = 0
            # Time masking
            for _ in range(2):
                t_width = torch.randint(0, 20, (1,)).item()
                if t_width > 0:
                    t0 = torch.randint(0, nt - t_width, (1,)).item()
                    in_batch[:, :, :, t0:t0+t_width, :] = 0
        
        # Forward through encoder directly (mirrors run_downstream.py exactly)
        # When pretrain=False and downstream_embed='spat', SARSSL.forward returns:
        # pred, pooled_embed, hidden_states
        # where pooled_embed is torch.mean(embed_spat, dim=1) of shape (B, spat_dembed)
        _, pooled_embed, _ = self.encoder(in_batch)
        
        logits = self.classifier(pooled_embed) # (B, num_classes)
        return logits

    def get_embedding(self, x):
        """Extract spatial encoder pooled embeddings directly for visualization/t-SNE."""
        self.stft_learner.device = next(self.parameters()).device
        
        preprocessed = self.stft_learner.data_preprocess(x, None)
        in_batch = preprocessed[0]
        
        _, pooled_embed, _ = self.encoder(in_batch)
        return pooled_embed

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configure output isolation per mode, pretrained model timestamp, and run timestamp
    from datetime import datetime as _dt
    import shlex as _shlex
    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    exp_dir = work_dir + '/SAR-SSL/' + args.exp_dir
    
    if getattr(args, 'test_only_mode', None):
        save_dir = os.path.abspath(args.test_only_mode)
        print(f"Running in test_only_mode using directory: {save_dir}")
    else:
        run_timestamp = _dt.now().strftime('%m%d%H%M')
        save_dir = os.path.join(
            exp_dir, 'room_classification', args.mode, args.time,
            'frozen' if args.probe_mode else 'finetune',
            run_timestamp
        )
        os.makedirs(save_dir, exist_ok=True)
    
    train_dataset = RoomClassificationDataset(args.data_dir, 'train', mode=args.mode)
    val_dataset = RoomClassificationDataset(args.data_dir, 'val', mode=args.mode)
    test_dataset = RoomClassificationDataset(args.data_dir, 'test', mode=args.mode)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    num_classes = 3 if args.mode == 'macro' else 110
    model = ClassifierWrapper(
        num_classes=num_classes,
        classifier_hidden=args.classifier_hidden,
        downstream_embed=args.downstream_embed
    ).to(device)
    
    if not getattr(args, 'test_only_mode', None):
        # Load pretrained weights reliably mapping keys from DataParallel checkpoints
        best_model_path = exp_dir + '/pretrain/' + args.time + '/best_model.tar'
        
        if os.path.exists(best_model_path):
            print(f"Loading pretrained weights from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location='cpu')
            pretrained_dict = checkpoint['model']
            
            # Clean keys by stripping 'module.' prefix if saved from DataParallel
            cleaned_dict = {}
            for k, v in pretrained_dict.items():
                clean_key = k[7:] if k.startswith('module.') else k
                cleaned_dict[clean_key] = v
                
            # Load matched keys into encoder
            encoder_dict = model.encoder.state_dict()
            matched_dict = {k: v for k, v in cleaned_dict.items() if k in encoder_dict}
            
            print(f"Matched {len(matched_dict)} keys out of {len(encoder_dict)} encoder parameters.")
            assert len(matched_dict) > 0, "Failed to match any keys from pretrained checkpoint! Verify pretrain path and model configuration."
            
            encoder_dict.update(matched_dict)
            model.encoder.load_state_dict(encoder_dict)
        else:
            print(f"WARNING: Pretrained weights not found at {best_model_path}. Initializing from scratch.")
        
    model = model.to(device)
    
    if args.probe_mode:
        print("Probe mode: Freezing spatial encoder parameters.")
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        print("Finetune mode: Training all parameters.")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Build the main LR scheduler
    decay_epochs = max(1, args.epochs - args.warmup_epochs)
    if args.lr_scheduler == 'polynomial':
        main_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=decay_epochs, power=0.1)
    elif args.lr_scheduler == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_epochs)
    elif args.lr_scheduler == 'smoothed_plateau':
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}. Choose 'polynomial', 'cosine', or 'smoothed_plateau'.")

    # Optionally prepend a linear warmup phase
    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs
        )
        if args.lr_scheduler == 'smoothed_plateau':
            scheduler = main_scheduler
            print(f"LR schedule: {args.warmup_epochs}-epoch linear warmup → smoothed_plateau decay (custom handling)")
        else:
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[args.warmup_epochs]
            )
            print(f"LR schedule: {args.warmup_epochs}-epoch linear warmup → {args.lr_scheduler} decay over {decay_epochs} epochs")
    else:
        warmup_scheduler = None
        scheduler = main_scheduler
        print(f"LR schedule: {args.lr_scheduler} decay over {args.epochs} epochs (no warmup)")
    
    save_model_path = os.path.join(save_dir, 'best_classifier.pth')
    
    if getattr(args, 'test_only_mode', None):
        report_path = os.path.join(save_dir, 'debug', 'classification_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                old_report = json.load(f)
            best_val_acc = old_report.get('training_results', {}).get('best_validation_accuracy', 0.0)
            history = old_report.get('training_results', {}).get('history_per_epoch', {})
        else:
            best_val_acc = 0.0
            history = {}
    else:
        best_val_acc = 0.0
        import collections
        checkpoint_queue = collections.deque(maxlen=5)
        lr_drop_count = 0
        smoothed_val_loss = None
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    for epoch in range(1, args.epochs + 1):
        if getattr(args, 'test_only_mode', None):
            break
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for audio, labels, _, _ in pbar:
            audio, labels = audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(audio)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=args.grad_clip
                )
            optimizer.step()
            
            train_loss += loss.item() * audio.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'Loss': loss.item(), 'Acc': train_correct / train_total})
            
        if args.lr_scheduler == 'smoothed_plateau':
            if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
                warmup_scheduler.step()
        else:
            scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
            for audio, labels, _, _ in pbar_val:
                audio, labels = audio.to(device), labels.to(device)
                logits = model(audio)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * audio.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        train_loss_epoch = train_loss / train_total
        train_acc_epoch = train_correct / train_total
        val_loss_epoch = val_loss / val_total
        val_acc_epoch = val_correct / val_total
        
        history['train_loss'].append(float(train_loss_epoch))
        history['train_acc'].append(float(train_acc_epoch))
        history['val_loss'].append(float(val_loss_epoch))
        history['val_acc'].append(float(val_acc_epoch))
        
        if args.lr_scheduler == 'smoothed_plateau':
            if smoothed_val_loss is None:
                smoothed_val_loss = val_loss_epoch
            else:
                smoothed_val_loss = 0.8 * smoothed_val_loss + 0.2 * val_loss_epoch
                
            if epoch > args.warmup_epochs:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(smoothed_val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < old_lr:
                    lr_drop_count += 1
                    
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f} | Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f} | LR: {current_lr:.2e}")
        
        current_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        checkpoint_queue.append(current_state_dict)
        
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            print(f"--> New best validation accuracy: {best_val_acc:.4f}. Saving model (ensembled over last {len(checkpoint_queue)} epochs).")
            ensembled_state_dict = {}
            for k in checkpoint_queue[0].keys():
                stack = torch.stack([ckpt[k] for ckpt in checkpoint_queue])
                if stack.is_floating_point():
                    ensembled_state_dict[k] = torch.mean(stack, dim=0)
                else:
                    ensembled_state_dict[k] = checkpoint_queue[-1][k]
            torch.save(ensembled_state_dict, save_model_path)
            
        if args.lr_scheduler == 'smoothed_plateau' and lr_drop_count >= 2:
            print("Early stopping triggered: LR dropped twice.")
            break
            
    # Save training curves plot
    if not getattr(args, 'test_only_mode', None):
        plt.figure(figsize=(12, 5))
        epochs_range = range(1, len(history['train_loss']) + 1)
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
        plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o')
        plt.plot(epochs_range, history['val_acc'], label='Val Acc', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)
        
        curves_path = os.path.join(save_dir, 'training_curves.png')
        plt.tight_layout()
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {curves_path}")
            
    # Final Test Evaluation
    print("\n--- Starting Final Test Evaluation ---")
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    model.eval()
    
    test_preds = []
    test_labels = []
    test_micro_ids = []
    test_paths = []
    test_embeds = []
    with torch.no_grad():
        for audio, labels, micro_ids, paths in tqdm(test_loader, desc="Testing"):
            audio = audio.to(device)
            # Single encoder pass: reuse the embedding for both t-SNE and the classifier head
            embeds = model.get_embedding(audio)
            logits = model.classifier(embeds)
            preds = logits.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_micro_ids.extend(micro_ids.cpu().numpy())
            test_paths.extend(paths)
            test_embeds.append(embeds.cpu().numpy())
            
    test_embeds = np.concatenate(test_embeds, axis=0)
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Final Test Top-1 Accuracy: {test_acc:.4f}")
    
    # --- Extended Diagnostics ---
    room_classes_list = get_room_classes()
    macro_to_id = {'Small': 0, 'Large': 1, 'Hall': 2}
    
    # Top-5 accuracy
    with torch.no_grad():
        all_logits = []
        for audio, _, _, _ in tqdm(test_loader, desc="Re-extracting logits for Top-5"):
            audio = audio.to(device)
            embeds_batch = model.get_embedding(audio)
            all_logits.append(model.classifier(embeds_batch).cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    top5_correct = sum(
        test_labels[i] in np.argsort(all_logits[i])[-5:]
        for i in range(len(test_labels))
    )
    test_top5_acc = top5_correct / len(test_labels)
    print(f"Final Test Top-5 Accuracy: {test_top5_acc:.4f}")

    # Macro accuracy derived from micro predictions (always computed, even in micro mode)
    # Map both true micro labels and predicted micro labels → macro category
    test_labels_arr = np.array(test_labels)
    test_preds_arr  = np.array(test_preds)
    test_micro_ids_arr = np.array(test_micro_ids)

    def micro_to_macro_id(micro_id):
        return macro_to_id[room_classes_list[micro_id][0]]

    if args.mode == 'micro':
        true_macro = np.array([micro_to_macro_id(l) for l in test_micro_ids_arr])
        # Predicted micro class → macro
        pred_macro  = np.array([micro_to_macro_id(p) if 0 <= p < len(room_classes_list) else -1
                                 for p in test_preds_arr])
        macro_acc = accuracy_score(true_macro, pred_macro)
        print(f"Macro-Level Accuracy (derived from micro predictions): {macro_acc:.4f}")
        
        # Per-macro-subgroup micro accuracy
        print("\nPer-Macro-Subgroup Test Accuracy (Micro Mode):")
        macro_names = ['Small', 'Large', 'Hall']
        per_macro_results = {}
        for mac_id, mac_name in enumerate(macro_names):
            mask = (true_macro == mac_id)
            n = mask.sum()
            if n > 0:
                correct = (test_preds_arr[mask] == test_labels_arr[mask]).sum()
                acc = correct / n
                print(f"  {mac_name:6s}: {acc:.4f}  ({correct}/{n} correct)")
                per_macro_results[mac_name] = {'n': int(n), 'correct': int(correct), 'accuracy': float(acc)}
    else:
        macro_acc = test_acc  # already macro mode
        per_macro_results = {}

    # Embedding health stats
    embed_norms = np.linalg.norm(test_embeds, axis=1)
    embed_vars  = np.var(test_embeds, axis=1)
    # Inter-class centroid distance: mean pairwise distance between per-class centroids
    unique_classes_set = sorted(set(test_labels))
    centroids = np.array([
        test_embeds[test_labels_arr == c].mean(axis=0)
        for c in unique_classes_set
    ])
    # Pairwise L2 between centroids
    from itertools import combinations as _comb
    centroid_dists = [
        np.linalg.norm(centroids[i] - centroids[j])
        for i, j in _comb(range(len(centroids)), 2)
    ]
    print(f"\nEmbedding Health:")
    print(f"  L2 norm   — mean: {embed_norms.mean():.3f}, std: {embed_norms.std():.3f}")
    print(f"  Variance  — mean: {embed_vars.mean():.4f}, std: {embed_vars.std():.4f}")
    print(f"  Inter-class centroid L2 — mean: {np.mean(centroid_dists):.3f}, "
          f"min: {np.min(centroid_dists):.3f}, max: {np.max(centroid_dists):.3f}")

    
    # Generate Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(20, 16) if args.mode == 'micro' else (8, 6))
    
    if args.mode == 'macro':
        macro_names = ['Small', 'Large', 'Hall']
        sns.heatmap(cm, annot=True, cmap='Blues', cbar=True, fmt='d',
                    xticklabels=macro_names, yticklabels=macro_names)
    else:
        sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
        
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'Room Classification Confusion Matrix ({args.mode.upper()} Mode | Test Acc: {test_acc:.4f})')
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Comprehensive Debug Output Generation
    debug_dir = os.path.join(save_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save embeddings and physical dimensions for latent space analysis
    np.save(os.path.join(debug_dir, 'test_embeds.npy'), test_embeds)
    
    physics_data = []
    for micro_id in test_micro_ids:
        macro_class, dims = room_classes_list[micro_id]
        width, depth, height = dims
        volume = width * depth * height
        physics_data.append({
            'Width': width,
            'Depth': depth,
            'Height': height,
            'Volume': volume,
            'Macro_Class': macro_class
        })
    pd.DataFrame(physics_data).to_csv(os.path.join(debug_dir, 'test_physics.csv'), index=False)
    
    # Calculate parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    # 1. Overall metrics report with complete context
    # Build a copy-pasteable command string (shlex.quote handles spaces / special chars in --description)
    _cmd_parts = ['python train_classifier.py']
    for _k, _v in vars(args).items():
        if _k == 'probe_mode':
            if _v:
                _cmd_parts.append('--probe_mode')
        elif _k == 'description':
            _cmd_parts.append(f'--description {_shlex.quote(str(_v))}')
        else:
            _cmd_parts.append(f'--{_k} {_shlex.quote(str(_v))}')
    reproduced_command = ' '.join(_cmd_parts)

    report = {
        'task': f'Room Shape Classification ({args.mode.upper()} Mode - {num_classes} classes)',
        'command': reproduced_command,
        'execution_args': vars(args),
        'model_context': {
            'architecture_summary': str(model),
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params
        },
        'dataset_context': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        },
        'training_results': {
            'best_validation_accuracy': float(best_val_acc),
            'best_validation_epoch': int(np.argmax(history['val_acc']) + 1) if history['val_acc'] else 0,
            'final_test_accuracy': float(test_acc),
            'final_test_top5_accuracy': float(test_top5_acc),
            'final_test_macro_accuracy': float(macro_acc),
            'per_macro_subgroup_accuracy': per_macro_results,
            'zero_accuracy_class_count': int((test_labels_arr != test_preds_arr).all() or
                sum(1 for c in unique_classes_set
                    if (test_preds_arr[test_labels_arr == c] == c).sum() == 0)),
            'embedding_health': {
                'norm_mean': float(embed_norms.mean()),
                'norm_std': float(embed_norms.std()),
                'variance_mean': float(embed_vars.mean()),
                'inter_class_centroid_dist_mean': float(np.mean(centroid_dists)),
                'inter_class_centroid_dist_min': float(np.min(centroid_dists)),
                'inter_class_centroid_dist_max': float(np.max(centroid_dists)),
            },
            'history_per_epoch': history
        }
    }
    with open(os.path.join(debug_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    # 2. Per-class accuracy CSV — enriched with macro type and room dimensions
    unique_classes = sorted(list(set(test_labels)))
    
    per_class_data = []
    for cls in unique_classes:
        mask = (test_labels_arr == cls)
        total_cls = mask.sum()
        correct_cls = (test_preds_arr[mask] == cls).sum()
        acc_cls = correct_cls / total_cls if total_cls > 0 else 0.0
        
        if args.mode == 'macro':
            cls_name = ['Small', 'Large', 'Hall'][cls]
            macro_type = cls_name
            dims = ('', '', '')
        else:
            cls_name = str(cls)
            macro_type = room_classes_list[cls][0]
            dims = room_classes_list[cls][1]  # (w, d, h)

        per_class_data.append({
            'Class_ID': cls, 
            'Class_Name': cls_name,
            'Macro_Type': macro_type,
            'Width': dims[0] if args.mode == 'micro' else '',
            'Depth': dims[1] if args.mode == 'micro' else '',
            'Height': dims[2] if args.mode == 'micro' else '',
            'Total': int(total_cls), 
            'Correct': int(correct_cls), 
            'Accuracy': float(acc_cls)
        })
        
    pd.DataFrame(per_class_data).to_csv(os.path.join(debug_dir, 'per_class_accuracy.csv'), index=False)
    
    # 3. Top confused class pairs CSV
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                i_name = ['Small', 'Large', 'Hall'][i] if args.mode == 'macro' else str(i)
                j_name = ['Small', 'Large', 'Hall'][j] if args.mode == 'macro' else str(j)
                confused_pairs.append({
                    'True_Class': i_name, 
                    'Predicted_Class': j_name, 
                    'Count': int(cm[i, j])
                })
    if confused_pairs:
        confused_df = pd.DataFrame(confused_pairs).sort_values(by='Count', ascending=False)
        confused_df.to_csv(os.path.join(debug_dir, 'top_confused_pairs.csv'), index=False)
        
    # 4. Worst failure cases sample (up to 100 worst cases)
    incorrect_mask = (test_labels_arr != test_preds_arr)
    failure_indices = np.where(incorrect_mask)[0]
    
    failure_cases = []
    for idx in failure_indices[:100]:
        t_cls = int(test_labels_arr[idx])
        p_cls = int(test_preds_arr[idx])
        t_name = ['Small', 'Large', 'Hall'][t_cls] if args.mode == 'macro' else str(t_cls)
        p_name = ['Small', 'Large', 'Hall'][p_cls] if args.mode == 'macro' else str(p_cls)
        failure_cases.append({
            'True_Class': t_name,
            'Predicted_Class': p_name,
            'File_Path': test_paths[idx],
            'Micro_Class_ID': int(test_micro_ids[idx])
        })
    with open(os.path.join(debug_dir, 'worst_failure_cases.json'), 'w') as f:
        json.dump(failure_cases, f, indent=4)
        
    print(f"Comprehensive debug files saved to {debug_dir}")

    # --- Results Summary Table ---
    if getattr(args, 'test_only_mode', None):
        print("Skipping results_summary.csv update in test_only_mode.")
    else:
        # Append one row per run to a persistent CSV at the room_classification root.
        from datetime import datetime as _dt
        summary_path = os.path.join(exp_dir, 'room_classification', 'results_summary.csv')
        summary_cols = [
            'Run_Timestamp', 'Pretrained_Time', 'Description', 'Mode', 'Probe_Mode',
            'Epochs', 'LR', 'LR_Scheduler', 'Warmup_Epochs', 'Weight_Decay', 'Grad_Clip',
            'Classifier_Hidden', 'Train_Samples',
            'Val_Acc_Best', 'Val_Best_Epoch',
            'Test_Top1_Acc', 'Test_Top5_Acc', 'Macro_Acc_Derived',
            'Save_Dir'
        ]
        summary_row = {
            'Run_Timestamp': _dt.now().strftime('%Y-%m-%d %H:%M'),
            'Pretrained_Time': args.time,
            'Description': args.description,
            'Mode': args.mode,
            'Probe_Mode': 'frozen' if args.probe_mode else 'finetune',
            'Epochs': args.epochs,
            'LR': args.lr,
            'LR_Scheduler': args.lr_scheduler,
            'Warmup_Epochs': args.warmup_epochs,
            'Weight_Decay': args.weight_decay,
            'Grad_Clip': args.grad_clip,
            'Classifier_Hidden': args.classifier_hidden,
            'Train_Samples': len(train_dataset),
            'Val_Acc_Best': round(float(best_val_acc), 4),
            'Val_Best_Epoch': int(np.argmax(history['val_acc']) + 1) if history.get('val_acc') else 0,
            'Test_Top1_Acc': round(float(test_acc), 4),
            'Test_Top5_Acc': round(float(test_top5_acc), 4),
            'Macro_Acc_Derived': round(float(macro_acc), 4),
            'Save_Dir': save_dir,
        }
        summary_df_new = pd.DataFrame([summary_row], columns=summary_cols)
        if os.path.exists(summary_path):
            summary_df_existing = pd.read_csv(summary_path)
            # Prepend new row on top so most recent is first
            summary_df = pd.concat([summary_df_new, summary_df_existing], ignore_index=True)
        else:
            summary_df = summary_df_new
        summary_df.to_csv(summary_path, index=False)
        print(f"Results summary updated: {summary_path}")

    # Generate t-SNE Visualization colored continuously by Room Width, Height, Depth
    print("\n--- Generating t-SNE Visualization of Spatial Embeddings ---")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeds_2d = tsne.fit_transform(test_embeds)
    
    room_classes = get_room_classes()
    # Use actual micro class IDs to extract genuine continuous physical dimensions
    widths = np.array([room_classes[lbl][1][0] for lbl in test_micro_ids])
    depths = np.array([room_classes[lbl][1][1] for lbl in test_micro_ids])
    heights = np.array([room_classes[lbl][1][2] for lbl in test_micro_ids])
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # Plot 1: Width
    sc0 = axes[0].scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=widths, cmap='viridis', alpha=0.7, s=15)
    axes[0].set_title('Colored by Room Width (m)')
    axes[0].set_xlabel('t-SNE Dim 1')
    axes[0].set_ylabel('t-SNE Dim 2')
    fig.colorbar(sc0, ax=axes[0], label='Width (m)')
    
    # Plot 2: Height
    sc1 = axes[1].scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=heights, cmap='plasma', alpha=0.7, s=15)
    axes[1].set_title('Colored by Room Height (m)')
    axes[1].set_xlabel('t-SNE Dim 1')
    axes[1].set_ylabel('t-SNE Dim 2')
    fig.colorbar(sc1, ax=axes[1], label='Height (m)')
    
    # Plot 3: Depth
    sc2 = axes[2].scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=depths, cmap='inferno', alpha=0.7, s=15)
    axes[2].set_title('Colored by Room Depth (m)')
    axes[2].set_xlabel('t-SNE Dim 1')
    axes[2].set_ylabel('t-SNE Dim 2')
    fig.colorbar(sc2, ax=axes[2], label='Depth (m)')
    
    plt.suptitle(f't-SNE Visualization of Downstream Spatial Encoder Embeddings ({args.mode.upper()} Mode)', fontsize=16, y=1.02)
    tsne_path = os.path.join(save_dir, 'tsne_embeddings.png')
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE visualization saved to {tsne_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Room Shape Classifier')
    # --- Paths ---
    parser.add_argument('--data_dir', type=str, default='../../data_room_classification', help='Directory containing metadata CSVs and wavs')
    parser.add_argument('--work_dir', type=str, default=os.path.abspath(os.path.join(os.getcwd(), '../../..')), help='Work directory')
    parser.add_argument('--exp_dir', type=str, default='exp', help='Experiment directory name')
    parser.add_argument('--time', type=str, default='', help='Pretrained model timestamp (e.g. "0711_1512")')
    parser.add_argument('--save_dir', type=str, default='exp/room_classification/', help='Base directory to save classifier weights and plots')
    # --- Experiment identity ---
    parser.add_argument('--description', type=str, default='', help='Short human-readable description of this run (logged to results_summary.csv)')
    # --- Task ---
    parser.add_argument('--mode', type=str, choices=['micro', 'macro'], default='micro', help='Classification target: micro (110 classes) or macro (3 classes)')
    parser.add_argument('--probe_mode', action='store_true', help='Freeze encoder and only train the classification head')
    # --- Architecture ---
    parser.add_argument('--classifier_hidden', type=int, default=0, help='Hidden dim of 2-layer MLP head (0 = single linear layer, the original setting)')
    parser.add_argument('--downstream_embed', type=str, choices=['spat', 'spec_spat'], default='spat', help='Which embedding to extract from encoder')
    # --- Optimization ---
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Peak learning rate')
    parser.add_argument('--lr_scheduler', type=str, choices=['polynomial', 'cosine', 'smoothed_plateau'], default='polynomial', help='LR decay schedule after warmup')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Linear LR warmup epochs before the main scheduler (0 = disabled)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='AdamW weight decay')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='Gradient clipping max norm (0 = disabled)')
    # --- Dataloader ---
    parser.add_argument('--workers', type=int, default=8, help='Dataloader worker processes')
    parser.add_argument('--test_only_mode', type=str, default=None, help='Path to a saved model directory to test only')
    
    args = parser.parse_args()
    if getattr(args, 'test_only_mode', None):
        report_path = os.path.join(args.test_only_mode, 'debug', 'classification_report.json')
        if os.path.exists(report_path):
            print(f"Loading arguments from {report_path}")
            with open(report_path, 'r') as f:
                report = json.load(f)
            # Update args with saved execution_args
            for k, v in report.get('execution_args', {}).items():
                if k not in ['test_only_mode', 'data_dir', 'work_dir']:
                    setattr(args, k, v)
        else:
            print(f"Warning: No classification_report.json found at {report_path}")
    train(args)
