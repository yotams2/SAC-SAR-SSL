
import torch
import unittest
from model import SARSSL

class TestFeatureNormalization(unittest.TestCase):
    def test_normalization(self):
        # Mock feature stats using mean/std expected by feature_similarity_loss
        # We pick stats so that p99/p1 mapping approximates the previous test expectations
        feature_stats = {
            'ild_mean': {'mean': 0.0, 'std': 5.0/3.0},
            'tdoa': {'mean': 0.0, 'std': 0.001/3.0},
            'coherence_low': {'mean': 0.5, 'std': 0.1},
            'coherence_mid': {'mean': 0.5, 'std': 0.1},
            'coherence_high': {'mean': 0.5, 'std': 0.1},
            'gcc_phat_max': {'mean': 0.0, 'std': 1.0},
            'hf_lf_ratio': {'mean': 0.0, 'std': 1.0}
        }
        
        model = SARSSL(sig_shape=[256, 100, 2, 2], feature_stats=feature_stats, feature_sim_loss=True)
        model.feat_sim_proj = torch.nn.Identity() # Mock projection to simple identity
        
        # Test ILD normalization
        # Range [-5, 5] -> [-1, 1]
        # 0 -> 0
        k = 'ild_mean'
        val = torch.tensor([0.0, 5.0, -5.0, 10.0, -10.0])
        
        stats = feature_stats[k]
        lo_f = stats['p1']
        hi_f = stats['p99']
        scale = 2.0 / (hi_f - lo_f)
        bias = -1.0 - scale * lo_f
        expected = torch.clamp(scale * val + bias, -1.0, 1.0)
        
        # Manually verify logic for checks
        # scale = 2 / 10 = 0.2
        # bias = -1 - 0.2 * (-5) = -1 + 1 = 0
        # 0 -> 0
        # 5 -> 1
        # -5 -> -1
        # 10 -> 2 -> 1
        # -10 -> -2 -> -1
        
        dummy_embed = torch.randn(5, 256)
        dummy_features = {
            'ild_mean': val,
            'tdoa': torch.zeros(5),
            'coherence_low': torch.zeros(5),
            'coherence_mid': torch.zeros(5),
            'coherence_high': torch.zeros(5)
        }
        
        _, metrics = model.feature_similarity_loss(dummy_embed, dummy_features, return_metrics=True)
        norm_vals = metrics['norm']['ild_mean']
        
        print(f"ILD Raw: {val}")
        print(f"ILD Norm: {norm_vals}")
        
        self.assertTrue(torch.allclose(norm_vals, expected), f"ILD normalization failed. Got {norm_vals}, expected {expected}")
        self.assertTrue(torch.all(norm_vals >= -1.0))
        self.assertTrue(torch.all(norm_vals <= 1.0))

        # Test TDOA normalization
        k = 'tdoa'
        val = torch.tensor([0.0, 0.001, -0.001])
        # scale = 2 / 0.002 = 1000
        # bias = -1 - 1000 * (-0.001) = -1 + 1 = 0
        dummy_features = {
            'ild_mean': torch.zeros(3),
            'tdoa': val,
            'coherence_low': torch.zeros(3),
            'coherence_mid': torch.zeros(3),
            'coherence_high': torch.zeros(3)
        }
        
        _, metrics = model.feature_similarity_loss(torch.randn(3, 256), dummy_features, return_metrics=True)
        norm_vals = metrics['norm']['tdoa']
        print(f"TDOA Raw: {val}")
        print(f"TDOA Norm: {norm_vals}")
        expected = torch.tensor([0.0, 1.0, -1.0])
        self.assertTrue(torch.allclose(norm_vals, expected), "TDOA normalization failed")
        
        print("Verification passed!")

if __name__ == '__main__':
    unittest.main()
