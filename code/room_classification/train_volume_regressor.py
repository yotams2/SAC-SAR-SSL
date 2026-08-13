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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
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

class RoomRegressionDataset(Dataset):
    def __init__(self, data_dir, split, target='volume'):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.target = target
        self.csv_path = os.path.join(data_dir, f'metadata_{split}.csv')
        self.df = pd.read_csv(self.csv_path)
        self.room_classes = get_room_classes()
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row['File_Path']
        micro_class_id = int(row['Micro_Class_ID'])
        
        mic_sig, fs = sf.read(wav_path)
        mic_sig = mic_sig.astype(np.float32)
        
        if self.split == 'train' and np.random.rand() > 0.5:
            mic_sig = mic_sig[:, ::-1].copy()
        
        macro_label, dims = self.room_classes[micro_class_id]
        if self.target == 'volume':
            volume = dims[0] * dims[1] * dims[2]
            target_tensor = torch.tensor([volume], dtype=torch.float32)
        else:
            target_tensor = torch.tensor(dims, dtype=torch.float32)
            
        return mic_sig, target_tensor, micro_class_id, wav_path

class RegressorWrapper(nn.Module):
    def __init__(self, spec_dembed=512, spat_dembed=256, classifier_hidden=0, downstream_embed='spat', target='volume'):
        super().__init__()
        self.downstream_embed = downstream_embed
        self.target = target
        
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
        self.stft_learner.device = 'cpu'
        
        embed_dim = spat_dembed if downstream_embed == 'spat' else (spec_dembed + spat_dembed)
        out_dim = 1 if target == 'volume' else 3
        
        if classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, classifier_hidden),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(classifier_hidden, out_dim)
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, out_dim)
            )
        
    def forward(self, x):
        self.stft_learner.device = next(self.parameters()).device
        
        preprocessed = self.stft_learner.data_preprocess(x, None)
        in_batch = preprocessed[0]
        
        if self.training:
            B, nmic, nf, nt, nreim = in_batch.shape
            for _ in range(2):
                f_width = torch.randint(0, 20, (1,)).item()
                if f_width > 0:
                    f0 = torch.randint(0, nf - f_width, (1,)).item()
                    in_batch[:, :, f0:f0+f_width, :, :] = 0
            for _ in range(2):
                t_width = torch.randint(0, 20, (1,)).item()
                if t_width > 0:
                    t0 = torch.randint(0, nt - t_width, (1,)).item()
                    in_batch[:, :, :, t0:t0+t_width, :] = 0
        
        _, pooled_embed, _ = self.encoder(in_batch)
        
        logits = self.classifier(pooled_embed) 
        return logits

    def get_embedding(self, x):
        self.stft_learner.device = next(self.parameters()).device
        preprocessed = self.stft_learner.data_preprocess(x, None)
        in_batch = preprocessed[0]
        _, pooled_embed, _ = self.encoder(in_batch)
        return pooled_embed

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
            exp_dir, 'room_regression', args.target, args.time,
            'frozen' if args.probe_mode else 'finetune',
            run_timestamp
        )
        os.makedirs(save_dir, exist_ok=True)
    
    train_dataset = RoomRegressionDataset(args.data_dir, 'train', target=args.target)
    val_dataset = RoomRegressionDataset(args.data_dir, 'val', target=args.target)
    test_dataset = RoomRegressionDataset(args.data_dir, 'test', target=args.target)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    model = RegressorWrapper(
        classifier_hidden=args.classifier_hidden,
        downstream_embed=args.downstream_embed,
        target=args.target
    ).to(device)
    
    if not getattr(args, 'test_only_mode', None):
        best_model_path = exp_dir + '/pretrain/' + args.time + '/best_model.tar'
        
        if os.path.exists(best_model_path):
            print(f"Loading pretrained weights from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location='cpu')
            pretrained_dict = checkpoint['model']
            
            cleaned_dict = {}
            for k, v in pretrained_dict.items():
                clean_key = k[7:] if k.startswith('module.') else k
                cleaned_dict[clean_key] = v
                
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
        
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    decay_epochs = max(1, args.epochs - args.warmup_epochs)
    if args.lr_scheduler == 'polynomial':
        main_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=decay_epochs, power=0.1)
    elif args.lr_scheduler == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_epochs)
    elif args.lr_scheduler == 'smoothed_plateau':
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}. Choose 'polynomial', 'cosine', or 'smoothed_plateau'.")

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
    
    save_model_path = os.path.join(save_dir, 'best_regressor.pth')
    out_dim = 1 if args.target == 'volume' else 3
    
    if getattr(args, 'test_only_mode', None):
        report_path = os.path.join(save_dir, 'debug', 'regression_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                old_report = json.load(f)
            best_val_mae = old_report.get('training_results', {}).get('best_validation_mae', float('inf'))
            history = old_report.get('training_results', {}).get('history_per_epoch', {})
        else:
            best_val_mae = float('inf')
            history = {}
    else:
        best_val_mae = float('inf')
        import collections
        checkpoint_queue = collections.deque(maxlen=5)
        lr_drop_count = 0
        smoothed_val_loss = None
        
        history = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': []
        }
    
    for epoch in range(1, args.epochs + 1):
        if getattr(args, 'test_only_mode', None):
            break
        model.train()
        train_loss = 0.0
        train_mae_sum = 0.0
        train_total_elements = 0
        
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
            
            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            
            mae = torch.abs(logits - labels).sum().item()
            train_mae_sum += mae
            train_total_elements += (batch_size * out_dim)
            
            pbar.set_postfix({'Loss (MSE)': loss.item(), 'MAE': train_mae_sum / train_total_elements})
            
        if args.lr_scheduler == 'smoothed_plateau':
            if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
                warmup_scheduler.step()
        else:
            scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_mae_sum = 0.0
        val_total_elements = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
            for audio, labels, _, _ in pbar_val:
                audio, labels = audio.to(device), labels.to(device)
                logits = model(audio)
                loss = criterion(logits, labels)
                
                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                
                mae = torch.abs(logits - labels).sum().item()
                val_mae_sum += mae
                val_total_elements += (batch_size * out_dim)
                
        train_loss_epoch = train_loss / len(train_dataset)
        train_mae_epoch = train_mae_sum / train_total_elements
        val_loss_epoch = val_loss / len(val_dataset)
        val_mae_epoch = val_mae_sum / val_total_elements
        
        history['train_loss'].append(float(train_loss_epoch))
        history['train_mae'].append(float(train_mae_epoch))
        history['val_loss'].append(float(val_loss_epoch))
        history['val_mae'].append(float(val_mae_epoch))
        
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
        print(f"Epoch {epoch}: Train Loss(MSE): {train_loss_epoch:.4f}, Train MAE: {train_mae_epoch:.4f} | Val Loss(MSE): {val_loss_epoch:.4f}, Val MAE: {val_mae_epoch:.4f} | LR: {current_lr:.2e}")
        
        current_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        checkpoint_queue.append(current_state_dict)
        
        if val_mae_epoch < best_val_mae:
            best_val_mae = val_mae_epoch
            print(f"--> New best validation MAE: {best_val_mae:.4f}. Saving model (ensembled over last {len(checkpoint_queue)} epochs).")
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
            
    if not getattr(args, 'test_only_mode', None):
        plt.figure(figsize=(12, 5))
        epochs_range = range(1, len(history['train_loss']) + 1)
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train Loss (MSE)', marker='o')
        plt.plot(epochs_range, history['val_loss'], label='Val Loss (MSE)', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_mae'], label='Train MAE', marker='o')
        plt.plot(epochs_range, history['val_mae'], label='Val MAE', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE Curves')
        plt.legend()
        plt.grid(True)
        
        curves_path = os.path.join(save_dir, 'training_curves.png')
        plt.tight_layout()
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {curves_path}")
            
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
            embeds = model.get_embedding(audio)
            logits = model.classifier(embeds)
            test_preds.append(logits.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_micro_ids.extend(micro_ids.cpu().numpy())
            test_paths.extend(paths)
            test_embeds.append(embeds.cpu().numpy())
            
    test_embeds = np.concatenate(test_embeds, axis=0)
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    test_mse = mean_squared_error(test_labels, test_preds)
    test_mae = mean_absolute_error(test_labels, test_preds)
    test_r2 = r2_score(test_labels, test_preds)
    test_mape = mean_absolute_percentage_error(test_labels, test_preds)
    
    print(f"Final Test MSE: {test_mse:.4f}")
    print(f"Final Test MAE: {test_mae:.4f}")
    print(f"Final Test MAPE: {test_mape:.4f}")
    print(f"Final Test R2: {test_r2:.4f}")
    
    if args.target == 'volume':
        plt.figure(figsize=(8, 8))
        plt.scatter(test_labels[:, 0], test_preds[:, 0], alpha=0.5, color='blue')
        max_val = max(np.max(test_labels), np.max(test_preds))
        min_val = min(np.min(test_labels), np.min(test_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
        plt.xlabel('True Volume')
        plt.ylabel('Predicted Volume')
        plt.title(f'Volume Regression | Test MAE: {test_mae:.2f} | R2: {test_r2:.4f} | MAPE: {test_mape:.4f}')
        plt.legend()
        
        scatter_path = os.path.join(save_dir, 'volume_scatter.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatter plot saved to {scatter_path}")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        dim_names = ['Width', 'Depth', 'Height']
        for i in range(3):
            axes[i].scatter(test_labels[:, i], test_preds[:, i], alpha=0.5)
            max_val = max(np.max(test_labels[:, i]), np.max(test_preds[:, i]))
            min_val = min(np.min(test_labels[:, i]), np.min(test_preds[:, i]))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[i].set_xlabel(f'True {dim_names[i]}')
            axes[i].set_ylabel(f'Predicted {dim_names[i]}')
            
            dim_mae = mean_absolute_error(test_labels[:, i], test_preds[:, i])
            dim_r2 = r2_score(test_labels[:, i], test_preds[:, i])
            axes[i].set_title(f'{dim_names[i]} | MAE: {dim_mae:.2f} | R2: {dim_r2:.4f}')
            
        plt.suptitle(f'Dimensions Regression | Overall Test MAE: {test_mae:.2f} | Overall R2: {test_r2:.4f} | MAPE: {test_mape:.4f}', fontsize=16)
        scatter_path = os.path.join(save_dir, 'dims_scatter.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatter plot saved to {scatter_path}")

    debug_dir = os.path.join(save_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    np.save(os.path.join(debug_dir, 'test_embeds.npy'), test_embeds)
    
    room_classes_list = get_room_classes()
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
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    _cmd_parts = ['python train_volume_regressor.py']
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
        'task': f'Room {args.target.capitalize()} Regression',
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
            'best_validation_mae': float(best_val_mae),
            'best_validation_epoch': int(np.argmin(history['val_mae']) + 1) if history.get('val_mae') else 0,
            'final_test_mae': float(test_mae),
            'final_test_mse': float(test_mse),
            'final_test_r2': float(test_r2),
            'final_test_mape': float(test_mape),
            'history_per_epoch': history
        }
    }
    with open(os.path.join(debug_dir, 'regression_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    if getattr(args, 'test_only_mode', None):
        print("Skipping regression_summary.csv update in test_only_mode.")
    else:
        from datetime import datetime as _dt
        summary_path = os.path.join(exp_dir, 'room_regression', 'regression_summary.csv')
        summary_cols = [
            'Run_Timestamp', 'Target', 'Pretrained_Time', 'Description', 'Probe_Mode',
            'Epochs', 'LR', 'LR_Scheduler', 'Warmup_Epochs', 'Weight_Decay', 'Grad_Clip',
            'Classifier_Hidden', 'Train_Samples',
            'Val_MAE_Best', 'Val_Best_Epoch',
            'Test_MAE', 'Test_MSE', 'Test_R2', 'Test_MAPE',
            'Save_Dir'
        ]
        summary_row = {
            'Run_Timestamp': _dt.now().strftime('%Y-%m-%d %H:%M'),
            'Target': args.target,
            'Pretrained_Time': args.time,
            'Description': args.description,
            'Probe_Mode': 'frozen' if args.probe_mode else 'finetune',
            'Epochs': args.epochs,
            'LR': args.lr,
            'LR_Scheduler': args.lr_scheduler,
            'Warmup_Epochs': args.warmup_epochs,
            'Weight_Decay': args.weight_decay,
            'Grad_Clip': args.grad_clip,
            'Classifier_Hidden': args.classifier_hidden,
            'Train_Samples': len(train_dataset),
            'Val_MAE_Best': round(float(best_val_mae), 4),
            'Val_Best_Epoch': int(np.argmin(history['val_mae']) + 1) if history.get('val_mae') else 0,
            'Test_MAE': round(float(test_mae), 4),
            'Test_MSE': round(float(test_mse), 4),
            'Test_R2': round(float(test_r2), 4),
            'Test_MAPE': round(float(test_mape), 4),
            'Save_Dir': save_dir,
        }
        summary_df_new = pd.DataFrame([summary_row], columns=summary_cols)
        if os.path.exists(summary_path):
            summary_df_existing = pd.read_csv(summary_path)
            # if previous csv didn't have Target or MAPE, fill with nan
            for col in summary_cols:
                if col not in summary_df_existing.columns:
                    summary_df_existing[col] = None
            summary_df = pd.concat([summary_df_new, summary_df_existing], ignore_index=True)
        else:
            summary_df = summary_df_new
        # Keep columns in order
        summary_df = summary_df[summary_cols]
        summary_df.to_csv(summary_path, index=False)
        print(f"Results summary updated: {summary_path}")

    print("\n--- Generating t-SNE Visualization of Spatial Embeddings ---")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeds_2d = tsne.fit_transform(test_embeds)
    
    widths = np.array([room_classes_list[lbl][1][0] for lbl in test_micro_ids])
    depths = np.array([room_classes_list[lbl][1][1] for lbl in test_micro_ids])
    heights = np.array([room_classes_list[lbl][1][2] for lbl in test_micro_ids])
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    sc0 = axes[0].scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=widths, cmap='viridis', alpha=0.7, s=15)
    axes[0].set_title('Colored by Room Width (m)')
    axes[0].set_xlabel('t-SNE Dim 1')
    axes[0].set_ylabel('t-SNE Dim 2')
    fig.colorbar(sc0, ax=axes[0], label='Width (m)')
    
    sc1 = axes[1].scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=heights, cmap='plasma', alpha=0.7, s=15)
    axes[1].set_title('Colored by Room Height (m)')
    axes[1].set_xlabel('t-SNE Dim 1')
    axes[1].set_ylabel('t-SNE Dim 2')
    fig.colorbar(sc1, ax=axes[1], label='Height (m)')
    
    sc2 = axes[2].scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=depths, cmap='inferno', alpha=0.7, s=15)
    axes[2].set_title('Colored by Room Depth (m)')
    axes[2].set_xlabel('t-SNE Dim 1')
    axes[2].set_ylabel('t-SNE Dim 2')
    fig.colorbar(sc2, ax=axes[2], label='Depth (m)')
    
    plt.suptitle(f't-SNE Visualization of Downstream Spatial Encoder Embeddings (Target: {args.target})', fontsize=16, y=1.02)
    tsne_path = os.path.join(save_dir, 'tsne_embeddings.png')
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE visualization saved to {tsne_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Room Regression Model')
    parser.add_argument('--data_dir', type=str, default='../../data_room_classification', help='Directory containing metadata CSVs and wavs')
    parser.add_argument('--work_dir', type=str, default=os.path.abspath(os.path.join(os.getcwd(), '../../..')), help='Work directory')
    parser.add_argument('--exp_dir', type=str, default='exp', help='Experiment directory name')
    parser.add_argument('--time', type=str, default='', help='Pretrained model timestamp (e.g. "0711_1512")')
    parser.add_argument('--save_dir', type=str, default='exp/room_regression/', help='Base directory to save regressor weights and plots')
    parser.add_argument('--description', type=str, default='', help='Short human-readable description of this run (logged to regression_summary.csv)')
    parser.add_argument('--target', type=str, choices=['volume', 'dims'], default='volume', help='Regression target: volume (1D) or dims (3D: w, d, h)')
    parser.add_argument('--probe_mode', action='store_true', help='Freeze encoder and only train the regression head')
    parser.add_argument('--classifier_hidden', type=int, default=0, help='Hidden dim of 2-layer MLP head (0 = single linear layer)')
    parser.add_argument('--downstream_embed', type=str, choices=['spat', 'spec_spat'], default='spat', help='Which embedding to extract from encoder')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Peak learning rate')
    parser.add_argument('--lr_scheduler', type=str, choices=['polynomial', 'cosine', 'smoothed_plateau'], default='polynomial', help='LR decay schedule after warmup')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Linear LR warmup epochs before the main scheduler (0 = disabled)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='AdamW weight decay')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='Gradient clipping max norm (0 = disabled)')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader worker processes')
    parser.add_argument('--test_only_mode', type=str, default=None, help='Path to a saved model directory to test only')
    
    args = parser.parse_args()
    if getattr(args, 'test_only_mode', None):
        report_path = os.path.join(args.test_only_mode, 'debug', 'regression_report.json')
        if os.path.exists(report_path):
            print(f"Loading arguments from {report_path}")
            with open(report_path, 'r') as f:
                report = json.load(f)
            for k, v in report.get('execution_args', {}).items():
                if k not in ['test_only_mode', 'data_dir', 'work_dir']:
                    setattr(args, k, v)
        else:
            print(f"Warning: No regression_report.json found at {report_path}")
    train(args)
