import unittest
import torch
import os
import sys
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.feed_forward import FeedForward

class TestFeedForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.embed_dim = 16
        self.ff_dim = 64
        self.ff = FeedForward(self.embed_dim, self.ff_dim).to(self.device)

    def test_output_shape(self):
        # 1. OUTPUT SHAPE
        cases = [
            (1, 4),  # B=1
            (4, 1),  # T=1
            (8, 16)  # multiple
        ]
        for B, T in cases:
            x = torch.randn(B, T, self.embed_dim, device=self.device)
            out = self.ff(x)
            self.assertEqual(out.shape, (B, T, self.embed_dim))

    def test_internal_dimension(self):
        # 2. INTERNAL DIMENSION
        self.assertEqual(self.ff.linear1.out_features, self.ff_dim)
        self.assertEqual(self.ff.linear1.in_features, self.embed_dim)
        
        self.assertEqual(self.ff.linear2.out_features, self.embed_dim)
        self.assertEqual(self.ff.linear2.in_features, self.ff_dim)

    def test_independent_reference(self):
        # 3. INDEPENDENT REFERENCE
        B, T = 1, 2
        x = torch.randn(B, T, self.embed_dim, device=self.device)
        
        out = self.ff(x)
        
        with torch.no_grad():
            W1 = self.ff.linear1.weight
            b1 = self.ff.linear1.bias
            W2 = self.ff.linear2.weight
            b2 = self.ff.linear2.bias
            
            # Independent calculation
            hidden = x @ W1.T + b1
            hidden_act = F.gelu(hidden)
            expected = hidden_act @ W2.T + b2
            
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_position_wise_independence(self):
        # 4. POSITION-WISE PROPERTY
        B, T = 1, 3
        x1 = torch.randn(B, T, self.embed_dim, device=self.device)
        x2 = x1.clone()
        
        # Change position 1
        x2[:, 1, :] += 10.0
        
        out1 = self.ff(x1)
        out2 = self.ff(x2)
        
        # Position 0 and 2 should remain identical
        self.assertTrue(torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5))
        self.assertTrue(torch.allclose(out1[:, 2, :], out2[:, 2, :], atol=1e-5))
        # Position 1 should change
        self.assertFalse(torch.allclose(out1[:, 1, :], out2[:, 1, :]))

    def test_batch_independence(self):
        # 5. TOKEN/BATCH INDEPENDENCE
        B, T = 3, 2
        x1 = torch.randn(B, T, self.embed_dim, device=self.device)
        x2 = x1.clone()
        
        # Change batch 1
        x2[1, :, :] += 10.0
        
        out1 = self.ff(x1)
        out2 = self.ff(x2)
        
        # Batch 0 and 2 should remain identical
        self.assertTrue(torch.allclose(out1[0, :, :], out2[0, :, :], atol=1e-5))
        self.assertTrue(torch.allclose(out1[2, :, :], out2[2, :, :], atol=1e-5))
        # Batch 1 should change
        self.assertFalse(torch.allclose(out1[1, :, :], out2[1, :, :]))

    def test_gelu_actually_used(self):
        # 6. GELU IS ACTUALLY USED
        # Compare GELU with ReLU or Identity
        # Use inputs around 0 where GELU differs from ReLU significantly (e.g. -1.0)
        B, T = 1, 1
        x = torch.ones(B, T, self.embed_dim, device=self.device) * -1.0
        
        out = self.ff(x)
        
        with torch.no_grad():
            hidden = self.ff.linear1(x)
            
            # Identity expected
            expected_id = self.ff.linear2(hidden)
            
            # ReLU expected
            expected_relu = self.ff.linear2(F.relu(hidden))
            
            # GELU expected
            expected_gelu = self.ff.linear2(F.gelu(hidden))
            
        # Verify it DOES NOT match identity or ReLU
        self.assertFalse(torch.allclose(out, expected_id, atol=1e-3))
        self.assertFalse(torch.allclose(out, expected_relu, atol=1e-3))
        
        # Verify it DOES match GELU
        self.assertTrue(torch.allclose(out, expected_gelu, atol=1e-5))

    def test_gradients(self):
        # 7. GRADIENTS
        B, T = 2, 4
        x = torch.randn(B, T, self.embed_dim, requires_grad=True, device=self.device)
        
        out = self.ff(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(self.ff.linear1.weight.grad)
        self.assertIsNotNone(self.ff.linear1.bias.grad)
        self.assertIsNotNone(self.ff.linear2.weight.grad)
        self.assertIsNotNone(self.ff.linear2.bias.grad)
        self.assertIsNotNone(x.grad)
        
        self.assertTrue(torch.any(self.ff.linear1.weight.grad != 0))
        self.assertTrue(torch.any(self.ff.linear1.bias.grad != 0))
        self.assertTrue(torch.any(self.ff.linear2.weight.grad != 0))
        self.assertTrue(torch.any(self.ff.linear2.bias.grad != 0))
        self.assertTrue(torch.any(x.grad != 0))

    def test_edge_cases(self):
        # 9. EDGE CASES
        cases = [
            (1, 1, 1, 1),      # small D, B, T, ff_dim
            (1, 2, 2, 8),      # feed_forward > embedding_dim
            (2, 4, 16, 4)      # feed_forward < embedding_dim
        ]
        for B, T, D, ff_dim in cases:
            model = FeedForward(D, ff_dim).to(self.device)
            x = torch.randn(B, T, D, device=self.device)
            out = model(x)
            self.assertEqual(out.shape, (B, T, D))

    def test_numerical_validity(self):
        # 10. NUMERICAL VALIDITY
        x = torch.randn(2, 4, self.embed_dim, device=self.device)
        out = self.ff(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_parameter_count_structure(self):
        # 12. PARAMETER COUNT / STRUCTURE
        params = list(self.ff.parameters())
        self.assertEqual(len(params), 4) # 2 weights, 2 biases
        self.assertTrue(isinstance(self.ff.linear1, torch.nn.Linear))
        self.assertTrue(isinstance(self.ff.linear2, torch.nn.Linear))
        self.assertTrue(isinstance(self.ff.act, torch.nn.GELU))

    def test_device_cpu(self):
        # 8. DEVICE CPU
        model = FeedForward(self.embed_dim, self.ff_dim).to('cpu')
        x = torch.randn(2, 4, self.embed_dim, device='cpu')
        out = model(x)
        self.assertEqual(out.device.type, 'cpu')

    def test_device_cuda(self):
        # 8. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        model = FeedForward(self.embed_dim, self.ff_dim).to('cuda')
        x = torch.randn(2, 4, self.embed_dim, device='cuda')
        out = model(x)
        self.assertEqual(out.device.type, 'cuda')

if __name__ == "__main__":
    unittest.main()
