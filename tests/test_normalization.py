import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.normalization import LayerNorm

class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.embed_dim = 16
        self.ln = LayerNorm(self.embed_dim).to(self.device)

    def test_output_shape(self):
        # 1. OUTPUT SHAPE
        cases = [
            (1, 4),  # B=1
            (4, 1),  # T=1
            (8, 16)  # multiple
        ]
        for B, T in cases:
            x = torch.randn(B, T, self.embed_dim, device=self.device)
            out = self.ln(x)
            self.assertEqual(out.shape, (B, T, self.embed_dim))

    def test_independent_mathematical_reference(self):
        # 2. INDEPENDENT MATHEMATICAL REFERENCE
        B, T, D = 2, 3, 4
        ln = LayerNorm(D).to(self.device)
        
        # Change weights/bias to ensure they're tested
        with torch.no_grad():
            ln.ln.weight.copy_(torch.randn(D))
            ln.ln.bias.copy_(torch.randn(D))
            
        x = torch.randn(B, T, D, device=self.device)
        out = ln(x)
        
        # Independent calculation
        # PyTorch LayerNorm calculates mean and variance along the last D dimensions
        # variance is calculated without Bessel's correction (unbiased=False)
        eps = ln.ln.eps
        mean = x.mean(dim=-1, keepdim=True)
        # variance = mean((x - mean)^2)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        
        normalized = (x - mean) / torch.sqrt(variance + eps)
        
        expected = normalized * ln.ln.weight + ln.ln.bias
        
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_normalization_property(self):
        # 3. NORMALIZATION PROPERTY
        B, T, D = 4, 4, 16
        x = torch.randn(B, T, D, device=self.device) * 10 + 5 # mean 5, var 100
        
        # Use default affine parameters (weight=1, bias=0)
        ln = LayerNorm(D).to(self.device)
        
        out = ln(x)
        
        # Output mean should be approx 0, var approx 1
        out_mean = out.mean(dim=-1)
        # For variance, we use unbiased=False to match layer norm's internal calculation
        out_var = out.var(dim=-1, unbiased=False)
        
        self.assertTrue(torch.allclose(out_mean, torch.zeros_like(out_mean), atol=1e-5))
        # variance will be exactly 1.0 - eps if weight=1, so 1e-4 tolerance easily covers it
        self.assertTrue(torch.allclose(out_var, torch.ones_like(out_var), atol=1e-4))

    def test_position_wise_independence(self):
        # 4. POSITION-WISE INDEPENDENCE
        B, T, D = 1, 3, 16
        x1 = torch.randn(B, T, D, device=self.device)
        x2 = x1.clone()
        
        x2[:, 1, 0] += 10.0 # Modify position 1 (feature 0 only to ensure normalized output changes)
        
        out1 = self.ln(x1)
        out2 = self.ln(x2)
        
        # Positions 0 and 2 should remain identical
        self.assertTrue(torch.allclose(out1[:, 0, :], out2[:, 0, :]))
        self.assertTrue(torch.allclose(out1[:, 2, :], out2[:, 2, :]))
        self.assertFalse(torch.allclose(out1[:, 1, :], out2[:, 1, :]))

    def test_batch_independence(self):
        # 5. BATCH INDEPENDENCE
        B, T, D = 3, 2, 16
        x1 = torch.randn(B, T, D, device=self.device)
        x2 = x1.clone()
        
        x2[1, :, 0] += 10.0 # Modify batch 1 (feature 0 only)
        
        out1 = self.ln(x1)
        out2 = self.ln(x2)
        
        # Batches 0 and 2 should remain identical
        self.assertTrue(torch.allclose(out1[0, :, :], out2[0, :, :]))
        self.assertTrue(torch.allclose(out1[2, :, :], out2[2, :, :]))
        self.assertFalse(torch.allclose(out1[1, :, :], out2[1, :, :]))

    def test_affine_parameters(self):
        # 6. AFFINE PARAMETERS
        weight = self.ln.ln.weight
        bias = self.ln.ln.bias
        
        self.assertEqual(weight.shape, (self.embed_dim,))
        self.assertEqual(bias.shape, (self.embed_dim,))
        
        self.assertTrue(weight.requires_grad)
        self.assertTrue(bias.requires_grad)

    def test_gradients(self):
        # 7. GRADIENTS
        B, T, D = 2, 4, 16
        x = torch.randn(B, T, D, requires_grad=True, device=self.device)
        
        out = self.ln(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.ln.ln.weight.grad)
        self.assertIsNotNone(self.ln.ln.bias.grad)
        
        self.assertTrue(torch.any(x.grad != 0))
        self.assertTrue(torch.any(self.ln.ln.weight.grad != 0))
        self.assertTrue(torch.any(self.ln.ln.bias.grad != 0))

    def test_edge_cases(self):
        # 9. EDGE CASES
        cases = [
            (1, 4, 1),   # D=1
            (2, 4, 2),   # small D
            (1, 4, 16),  # B=1
            (2, 1, 16),  # T=1
            (2, 3, 5)    # multiple combos
        ]
        for B, T, D in cases:
            model = LayerNorm(D).to(self.device)
            x = torch.randn(B, T, D, device=self.device)
            out = model(x)
            self.assertEqual(out.shape, (B, T, D))

    def test_numerical_validity(self):
        # 10. NUMERICAL VALIDITY
        x = torch.randn(2, 4, self.embed_dim, device=self.device)
        out = self.ln(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_device_cpu(self):
        # 8. DEVICE CPU
        model = LayerNorm(16).to('cpu')
        x = torch.randn(2, 4, 16, device='cpu')
        out = model(x)
        self.assertEqual(out.device.type, 'cpu')

    def test_device_cuda(self):
        # 8. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        model = LayerNorm(16).to('cuda')
        x = torch.randn(2, 4, 16, device='cuda')
        out = model(x)
        self.assertEqual(out.device.type, 'cuda')

if __name__ == "__main__":
    unittest.main()
