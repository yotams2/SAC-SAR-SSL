
import torch
import torch.nn as nn
from model import SARSSL

def test_contrastive_loss():
    print("Testing Contrastive Loss implementation...")
    
    # Setup
    batch_size = 8
    dim = 16
    nmic = 2
    nf = 32
    nt = 32
    num_utterances = 4
    
    device = 'cpu'
    
    # Instantiate model with CL enabled
    model = SARSSL(
        sig_shape=[nf, nt, 2, nmic], 
        contrastive_learning=True, 
        num_utterances=num_utterances,
        device=device
    )
    
    # Create Dummy Input
    # Shape: (nbatch, nmic, nf, nt, nreim)
    x = torch.randn(batch_size, nmic, nf, nt, 2)
    
    # Create Dummy Indices
    # Let's say we have 2 rooms, each with 4 utterances.
    # Room 0: Utterances 0, 1, 2, 3 -> Indices 0, 1, 2, 3
    # Room 1: Utterances 0, 1, 2, 3 -> Indices 4, 5, 6, 7
    # Batch indices: [0, 1, 4, 5, 0, 1, 4, 5] mixture
    # RIR indices:   [0, 0, 1, 1, 0, 0, 1, 1]
    indices = torch.tensor([0, 1, 4, 5, 0, 1, 4, 5], dtype=torch.long)
    
    print(f"Input indices: {indices}")
    print(f"Expected RIR indices: {indices // num_utterances}")
    
    # Forward Pass
    try:
        loss, diff, vis = model(x, indices=indices)
        print(f"Forward pass successful. Loss: {loss.item()}")
        
        # Backward Pass Check
        loss.backward()
        print("Backward pass successful.")
        
        # Check gradients
        grad_found = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                # print(f"Gradient found in {name}")
                if param.grad.abs().sum() > 0:
                    grad_found = True
                    break
        
        if grad_found:
            print("Gradients are flowing to parameters.")
        else:
            print("WARNING: No gradients found in parameters!")

        # Check spacial encoder grads specifically
        spat_grads = False
        for name, param in model.spat_encoder.named_parameters():
             if param.grad is not None and param.grad.abs().sum() > 0:
                 spat_grads = True
                 break
        
        if spat_grads:
             print("Spatial Encoder is receiving gradients.")
        else:
             print("WARNING: Spatial Encoder is NOT receiving gradients!")
             
        # Check projection head grads
        if hasattr(model, 'cl_proj'):
            proj_grads = False
            for name, param in model.cl_proj.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    proj_grads = True
                    break
            if proj_grads:
                print("Projection Head is receiving gradients.")
            else:
                 print("WARNING: Projection Head is NOT receiving gradients!")


        
    except Exception as e:
        print(f"Test Failed with error: {e}")
        raise e

    print("Contrastive Loss Test Passed (Logic check).")

if __name__ == "__main__":
    test_contrastive_loss()
