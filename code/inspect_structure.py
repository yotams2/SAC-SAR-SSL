
import sys
import os
import torch
sys.path.append('/storage/yotam/SAR-SSL/SAR-SSL/code')

# Mocking necessary parts if imports fail, but we assume environment is set up.
import model as at_model

def inspect_model():
    print("Initializing SARSSL model with hierarchical=True...")
    # Parameters from config
    nf = 256
    nt = 256
    spec_dembed = 512
    spat_dembed = 256
    spectral_enc_input = 'all'
    hierarchical = True
    
    net = at_model.SARSSL(
        sig_shape=(nf, nt, 2, 2), 
        pretrain=True, 
        device='cpu',
        contrastive_learning=False,
        spec_dembed=spec_dembed,
        spat_dembed=spat_dembed,
        spectral_enc_input=spectral_enc_input,
        hierarchical=hierarchical
    )
    
    print("\n--- Spatial Encoder Structure ---")
    spat_enc = net.spat_encoder
    print(spat_enc)
    
    print("\n--- Conformer Encoder Details ---")
    if hasattr(spat_enc, 'embed'):
        conf_enc = spat_enc.embed
        print(f"Hierarchical: {conf_enc.hierarchical}")
        
        # Check layers
        if hasattr(conf_enc, 'layers'):
            print(f"Number of Shared Layers: {len(conf_enc.layers)}")
            
        if hasattr(conf_enc, 'last_layer_doa'):
            print("Has last_layer_doa: Yes")
        else:
            print("Has last_layer_doa: No")
            
        if hasattr(conf_enc, 'last_layer_room'):
            print("Has last_layer_room: Yes")
        else:
            print("Has last_layer_room: No")
            
        # Total layers intended
        # In model.py: mhsa_nlayer = 3 for mode='spat'
        # So we expect 2 shared, 1 split.
        
inspect_model()
