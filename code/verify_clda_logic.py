
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
    
    # Visualize Masks
    pos_mask_spat = (label_spat.unsqueeze(0) == label_spat.unsqueeze(1)).float()
    self_mask = torch.eye(batch_size)
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
    
    # Run Loss
    loss_spat = model.cl_loss(z_spat, label_spat)
    print(f"\nCalculated Spatial Loss (Random Init): {loss_spat.item():.4f}")
    
    # Check Gradients
    loss_spat.backward()
    print(f"Gradients on z_spat exists: {z_spat.grad is not None}")
    if z_spat.grad is not None:
        grad_norm = z_spat.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.4f}")
        if grad_norm == 0:
            print("WARNING: Gradient is ZERO.")
    
    expected_loss = np.log(batch_size - 1)
    print(f"Note: Log({batch_size}-1) = {expected_loss:.4f}")
    
    # Check Collapse Case
    z_collapse = torch.ones(batch_size, 256, requires_grad=True)
    loss_collapse = model.cl_loss(z_collapse, label_spat)
    print(f"Calculated Spatial Loss (Collapsed): {loss_collapse.item():.4f}")
    
    if abs(loss_spat.item() - expected_loss) < 0.1:
        print("Observation: Random Init Loss is close to Log(B-1). This is normal for orthogonal vectors.")
        
    if abs(loss_collapse.item() - expected_loss) < 0.1:
        print("Observation: Collapsed Loss is ALSO close to Log(B-1).")
        print("CRITICAL: 'Stuck on constant value' likely means the model isn't learning to discriminate.")

if __name__ == "__main__":
    verify_clda_logic()
