
import sys
import os
import torch
import numpy as np

# Add code directory to path
sys.path.append(os.getcwd() + '/../code')

from model import SARSSL

def verify_clda_logic():
    print("Verifying CLDA Logic...")
    
    # Mock parameters
    batch_size = 16 # 2 base samples * 8 augmentations
    augmentations_per_sample = 8
    device = 'cpu'
    
    # Create Model
    model = SARSSL(
        sig_shape=[256, 256, 2, 2], 
        pretrain=True, 
        contrastive_learning=False,
        contrastive_learning_data_aug=True,
        device=device
    )
    
    # Mock Indices: 2 groups of 8
    # [0-7] and [16-23] (simulating non-contiguous base indices if sampled randomly)
    # The sampler yields CONTINUOUS indices for the augmented block.
    # Base 0 -> 0..7. Base 2 -> 16..23.
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 
                            16, 17, 18, 19, 20, 21, 22, 23], dtype=torch.long)
    
    print(f"\nMock Indices: {indices.tolist()}")
    
    # Re-implement Logic from Model.py to debug
    base_indices = (indices // 8) * 8
    offset = indices % 8
    mask = offset < 4
    
    label_spat = torch.where(mask, indices, base_indices)
    label_spec = torch.where(mask, base_indices, indices)
    
    print(f"\nSpatial Labels: {label_spat.tolist()}")
    print("  Expected Grouping: {0,4,5,6,7} should match. {1}, {2}, {3} should be singletons.")
    
    print(f"Spectral Labels: {label_spec.tolist()}")
    print("  Expected Grouping: {0,1,2,3} should match. {4}, {5}, {6}, {7} should be singletons.")
    
    # --- Verify Neutral Masks ---
    print("\n--- Verifying Neutral Masks ---")
    
    # Spat Neutral Mask Logic
    # Any pair from same base signal that is NOT a positive pair should be neutral
    same_base_mask = (base_indices.unsqueeze(0) == base_indices.unsqueeze(1)).float()
    pos_mask_spat = (label_spat.unsqueeze(0) == label_spat.unsqueeze(1)).float()
    neutral_mask_spat = same_base_mask * (1 - pos_mask_spat)
    
    # Spec Neutral Mask Logic
    # Any pair from same base signal that is NOT a positive pair should be neutral
    pos_mask_spec = (label_spec.unsqueeze(0) == label_spec.unsqueeze(1)).float()
    neutral_mask_spec = same_base_mask * (1 - pos_mask_spec)
    
    print(f"Neutral Mask Spat (Sum): {neutral_mask_spat.sum().item()}")
    # Total pairs for one base (8 samples): 8*8 = 64.
    # Positives + Neutrals = Total (since all share base)
    # Spat Labels: 1 group of 5 (offset>=4 + base), 3 singletons (offset 1,2,3).
    # Positives: 5*5 + 1*1 + 1*1 + 1*1 = 25 + 3 = 28.
    # Expected Neutrals: 64 - 28 = 36.
    # Total Neutral for 2 bases: 36 * 2 = 72.
    print("Expected Spat Neutral Count (incl diag): 72")
    
    print(f"Neutral Mask Spec (Sum): {neutral_mask_spec.sum().item()}")
    # Spec Labels: 1 group of 4 (offset<4), 4 singletons (offset 4,5,6,7).
    # Positives: 4*4 + 4*1 = 16 + 4 = 20.
    # Expected Neutrals: 64 - 20 = 44.
    # Total Neutral for 2 bases: 44 * 2 = 88.
    print("Expected Spec Neutral Count (incl diag): 88")

    # Visualize Masks
    pos_mask_spat = (label_spat.unsqueeze(0) == label_spat.unsqueeze(1)).float()
    p_i_spat = pos_mask_spat.sum(dim=1) - 1
    p_i_spat = torch.max(p_i_spat, torch.ones_like(p_i_spat))
    
    print(f"\nSpatial Positives Count per sample:\n{p_i_spat.tolist()}")
    
    pos_mask_spec = (label_spec.unsqueeze(0) == label_spec.unsqueeze(1)).float()
    p_i_spec = pos_mask_spec.sum(dim=1) - 1
    p_i_spec = torch.max(p_i_spec, torch.ones_like(p_i_spec))
    
    print(f"Spectral Positives Count per sample:\n{p_i_spec.tolist()}")
    
    # Mock Embeddings (Random)
    torch.manual_seed(42)
    # Mock z_spat
    z_spat = torch.randn(batch_size, 256, requires_grad=True)
    
    # Run Loss WITH Neutral Mask
    loss_spat = model.cl_loss(z_spat, label_spat, neutral_mask=neutral_mask_spat)
    print(f"\nCalculated Spatial Loss (Random Init) WITH Neutral Mask: {loss_spat.item():.4f}")
    
    # Check Gradients
    loss_spat.backward()
    print(f"Gradients on z_spat exists: {z_spat.grad is not None}")
    if z_spat.grad is not None:
        grad_norm = z_spat.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.4f}")
        if grad_norm == 0:
            print("WARNING: Gradient is ZERO.")
    
    # Check Logic without neutral mask for comparison
    z_spat_2 = z_spat.clone().detach().requires_grad_(True)
    loss_spat_no_neutral = model.cl_loss(z_spat_2, label_spat, neutral_mask=None)
    print(f"Calculated Spatial Loss (Random Init) WITHOUT Neutral Mask: {loss_spat_no_neutral.item():.4f}")
    
    if loss_spat.item() != loss_spat_no_neutral.item():
        print("Verification SUCCESS: Neutral mask implies different loss value.")
    else:
        print("Verification WARNING: Loss is same. Might be because random vectors are already orthogonal enough or mask had no effect?")

if __name__ == "__main__":
    verify_clda_logic()
