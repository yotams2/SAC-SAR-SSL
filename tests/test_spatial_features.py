import torch
import sys
from pathlib import Path

# Add 'code' directory to path to allow importing modules
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
code_dir = project_root / 'code'
sys.path.append(str(code_dir))

from common.utils_module import spatial_features_from_stft


def test_spatial_features_coherence_bands():
    B = 4
    F = 257
    T = 10
    device = 'cpu'

    # Create random complex STFT-like tensor [B, 2, F, T]
    real = torch.randn(B, 2, F, T)
    imag = torch.randn(B, 2, F, T)
    X = torch.complex(real, imag)

    feats = spatial_features_from_stft(X, sample_rate=16000.0)

    # Check new keys exist
    assert 'coherence_low' in feats
    assert 'coherence_mid' in feats
    assert 'coherence_high' in feats
    assert 'coherence_mean' in feats
    assert 'hf_lf_ratio' in feats
    assert 'gcc_phat_max' in feats

    # Check shapes
    assert feats['coherence_low'].shape == (B,)
    assert feats['coherence_mid'].shape == (B,)
    assert feats['coherence_high'].shape == (B,)
    assert feats['coherence_mean'].shape == (B,)

    # Coherence values should be in [0,1]
    for k in ['coherence_low', 'coherence_mid', 'coherence_high', 'coherence_mean']:
        v = feats[k]
        assert torch.all(v >= 0.0 - 1e-6)
        assert torch.all(v <= 1.0 + 1e-6)

    # hf_lf_ratio and gcc_phat_max have different ranges
    assert feats['hf_lf_ratio'].shape == (B,)
    assert feats['gcc_phat_max'].shape == (B,)
    # gcc_phat_max should be non-negative, hf_lf_ratio finite
    assert torch.all(feats['gcc_phat_max'] >= 0.0 - 1e-6)
    assert torch.isfinite(feats['hf_lf_ratio']).all()

    # HF/LF ratio should be finite and have correct shape
    assert feats['hf_lf_ratio'].shape == (B,)
    assert torch.isfinite(feats['hf_lf_ratio']).all()

    # GCC-PHAT max should be finite and non-negative
    assert feats['gcc_phat_max'].shape == (B,)
    assert torch.isfinite(feats['gcc_phat_max']).all()
    assert torch.all(feats['gcc_phat_max'] >= 0.0 - 1e-6)


if __name__ == '__main__':
    test_spatial_features_coherence_bands()
    print('OK')
