import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root / 'code'))

import torch
from model import SARSSL


def test_feature_similarity_uses_banded_coherence():
    B = 6
    dembed = 128
    # Instantiate a model with projection head
    model = SARSSL(sig_shape=[256, 100, 2, 2], feature_sim_loss=True, feature_stats={
        'ild_mean':{'mean':0.,'std':1.},
        'coherence_low':{'mean':0.5,'std':0.1},
        'coherence_mid':{'mean':0.5,'std':0.1},
        'coherence_high':{'mean':0.5,'std':0.1},
        'gcc_phat_max':{'mean':0.5,'std':0.1},
        'hf_lf_ratio':{'mean':0.,'std':1.},
        'tdoa':{'mean':0.,'std':1.}
    })
    model.feat_sim_proj = torch.nn.Identity()

    z = torch.randn(B, dembed)
    spatial_features = {
        'ild_mean': torch.zeros(B),
        'coherence_low': torch.linspace(0.1,0.9,B),
        'coherence_mid': torch.linspace(0.2,0.8,B),
        'coherence_high': torch.linspace(0.3,0.7,B),
        'tdoa': torch.zeros(B),
        'gcc_phat_max': torch.ones(B) * 0.5,
        'hf_lf_ratio': torch.zeros(B)
    }

    loss, metrics = model.feature_similarity_loss(z, spatial_features, return_metrics=True)

    # Check raw and norm keys include banded coherence
    for k in ['coherence_low','coherence_mid','coherence_high','ild_mean','tdoa']:
        assert k in metrics['raw']
        assert k in metrics['norm']
        assert metrics['raw'][k].shape[0] == B
        assert metrics['norm'][k].shape[0] == B
    # Also check new features
    for k in ['gcc_phat_max','hf_lf_ratio']:
        assert k in metrics['raw']
        assert k in metrics['norm']
        assert metrics['raw'][k].shape[0] == B
        assert metrics['norm'][k].shape[0] == B

    print('Feature similarity cues test passed')

if __name__ == '__main__':
    test_feature_similarity_uses_banded_coherence()
    print('OK')
