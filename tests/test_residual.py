import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.residual import ResidualAdd

class TestResidualAdd(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.res = ResidualAdd().to(self.device)

    def test_basic_addition(self):
        # 1. BASIC ADDITION
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        y = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
        expected = torch.tensor([[[6.0, 8.0], [10.0, 12.0]]])
        
        out = self.res(x, y)
        self.assertTrue(torch.allclose(out, expected))

    def test_shape_preservation(self):
        # 2. SHAPE
        B, T, D = 2, 4, 16
        x = torch.randn(B, T, D)
        y = torch.randn(B, T, D)
        out = self.res(x, y)
        self.assertEqual(out.shape, (B, T, D))

    def test_zero_behavior(self):
        # 3. ZERO BEHAVIOR
        x = torch.randn(2, 4, 16)
        y = torch.zeros(2, 4, 16)
        out = self.res(x, y)
        self.assertTrue(torch.allclose(out, x))

    def test_identity_behavior(self):
        # 4. IDENTITY BEHAVIOR
        x = torch.zeros(2, 4, 16)
        y = torch.randn(2, 4, 16)
        out = self.res(x, y)
        self.assertTrue(torch.allclose(out, y))

    def test_negative_values(self):
        # 5. NEGATIVE VALUES
        x = torch.tensor([[[-5.0, 2.0], [3.0, -10.0]]])
        y = torch.tensor([[[5.0, -4.0], [-1.0, 0.0]]])
        expected = torch.tensor([[[0.0, -2.0], [2.0, -10.0]]])
        out = self.res(x, y)
        self.assertTrue(torch.allclose(out, expected))

    def test_no_in_place_modification(self):
        # 6. NO IN-PLACE MODIFICATION
        x_orig = torch.randn(2, 3, 4)
        y_orig = torch.randn(2, 3, 4)
        
        x = x_orig.clone()
        y = y_orig.clone()
        
        _ = self.res(x, y)
        
        self.assertTrue(torch.equal(x, x_orig))
        self.assertTrue(torch.equal(y, y_orig))

    def test_no_learned_parameters(self):
        # 7. NO LEARNED PARAMETERS
        params = list(self.res.parameters())
        self.assertEqual(len(params), 0)

    def test_gradients(self):
        # 8. GRADIENTS
        x = torch.randn(2, 3, 4, requires_grad=True)
        y = torch.randn(2, 3, 4, requires_grad=True)
        
        out = self.res(x, y)
        
        # Backward pass from a sum (each grad should be exactly 1.0)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))
        self.assertTrue(torch.allclose(y.grad, torch.ones_like(y)))

    def test_batch_position_independence(self):
        # 9. BATCH/POSITION INDEPENDENCE
        x = torch.zeros(2, 3, 4)
        y = torch.zeros(2, 3, 4)
        
        y[1, 2, 3] = 10.0
        
        out = self.res(x, y)
        
        # Only [1, 2, 3] should be 10.0, rest 0.0
        self.assertEqual(out[1, 2, 3].item(), 10.0)
        out[1, 2, 3] = 0.0
        self.assertTrue(torch.all(out == 0.0))

    def test_shape_safety(self):
        # 12. SHAPE SAFETY
        x = torch.randn(2, 3, 4)
        
        # Should raise ValueError for broadcasting attempts
        with self.assertRaises(ValueError):
            self.res(x, torch.randn(2, 3, 1)) # wrong D
            
        with self.assertRaises(ValueError):
            self.res(x, torch.randn(1, 3, 4)) # wrong B
            
        with self.assertRaises(ValueError):
            self.res(x, torch.randn(3, 4))    # missing B

    def test_edge_cases(self):
        # 11. EDGE CASES
        cases = [
            (1, 1, 1),
            (2, 1, 1),
            (1, 4, 1),
            (1, 1, 16),
            (2, 4, 16)
        ]
        for B, T, D in cases:
            x = torch.randn(B, T, D)
            y = torch.randn(B, T, D)
            out = self.res(x, y)
            self.assertEqual(out.shape, (B, T, D))

    def test_device_cpu(self):
        # 10. DEVICE CPU
        model = ResidualAdd().to('cpu')
        x = torch.randn(2, 4, 16, device='cpu')
        y = torch.randn(2, 4, 16, device='cpu')
        out = model(x, y)
        self.assertEqual(out.device.type, 'cpu')

    def test_device_cuda(self):
        # 10. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        model = ResidualAdd().to('cuda')
        x = torch.randn(2, 4, 16, device='cuda')
        y = torch.randn(2, 4, 16, device='cuda')
        out = model(x, y)
        self.assertEqual(out.device.type, 'cuda')

if __name__ == "__main__":
    unittest.main()
