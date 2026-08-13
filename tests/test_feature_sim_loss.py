
import sys
import os
from pathlib import Path

# Add 'code' directory to path to allow importing modules
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
code_dir = project_root / 'code'
sys.path.append(str(code_dir))

import torch
import torch.nn as nn
from model import SARSSL
from common import utils_module

def test_feature_sim_loss():
    print("Testing Feature Similarity Loss...")
    
    # 1. Setup Model
    device = 'cpu'
    model = SARSSL(
        sig_shape=[256, 100, 2, 2], # [F, T, Reim, Mic] ... wait, standard is [F, T, Reim, Mic] or [Mic, F, T, Reim]? 
        # model.py: nf, nt, nreim, nmic = sig_shape
        patch_shape=(256, 1),
        feature_sim_loss=True,
        spat_dembed=256,
        spec_dembed=256,
        device=device
    )
    
    # 2. Create Dummy Input
    # [Batch, Mic, F, T, Reim]
    B = 8
    # F=256, T=100
    x = torch.randn(B, 2, 256, 100, 2)
    
    # 3. Forward Pass
    # This should trigger feature extraction and loss calculation
    # We need indices for contrastive learning but here we are testing feat_sim_loss which doesn't strictly need indices unless cl is on.
    # But forward expects indices=None by default.
    
    # Mocking spatial features to ensure they are working
    print("Running forward pass...")
    loss, diff, vis, loss_cl_spat, loss_cl_spec, obt_loss, loss_gen = model(x)
    
    print(f"Total Loss: {loss.item()}")
    print("Feature Sim Loss check passed if code ran without error.")
    
    # 4. Check Gradients
    loss.backward()
    print("Backward pass successful.")
    
    # 5. Check if feat_sim_proj has grads
    assert model.feat_sim_proj[0].weight.grad is not None
    print("Gradients flowed to projection head.")

if __name__ == "__main__":
    test_feature_sim_loss()
