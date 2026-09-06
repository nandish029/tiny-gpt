import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.lm_head import LanguageModelHead

class TestLanguageModelHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.embed_dim = 16
        self.vocab_size = 100
        self.head = LanguageModelHead(self.embed_dim, self.vocab_size).to(self.device)

    def test_output_shape(self):
        # 1. OUTPUT SHAPE
        cases = [
            (1, 4, 16, 100),
            (2, 1, 16, 100),
            (3, 8, 16, 50),
            (1, 1, 4, 10),
        ]
        for B, T, D, V in cases:
            model = LanguageModelHead(D, V).to(self.device)
            x = torch.randn(B, T, D, device=self.device)
            out = model(x)
            self.assertEqual(out.shape, (B, T, V))

    def test_independent_reference(self):
        # 2. INDEPENDENT REFERENCE
        B, T = 2, 3
        x = torch.randn(B, T, self.embed_dim, device=self.device)
        out = self.head(x)

        with torch.no_grad():
            W = self.head.linear.weight
            b = self.head.linear.bias
            expected = x @ W.T + b

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_raw_logits(self):
        # 3. RAW LOGITS — output should NOT sum to 1
        x = torch.randn(2, 4, self.embed_dim, device=self.device)
        out = self.head(x)

        # If softmax were applied, each [V] vector would sum to 1.0
        sums = out.sum(dim=-1)
        ones = torch.ones_like(sums)
        self.assertFalse(torch.allclose(sums, ones, atol=1e-2))

        # Logits can be negative
        self.assertTrue((out < 0).any())

    def test_position_independence(self):
        # 4. POSITION INDEPENDENCE
        B, T = 1, 3
        x1 = torch.randn(B, T, self.embed_dim, device=self.device)
        x2 = x1.clone()
        x2[:, 1, :] += 10.0

        out1 = self.head(x1)
        out2 = self.head(x2)

        self.assertTrue(torch.allclose(out1[:, 0, :], out2[:, 0, :]))
        self.assertTrue(torch.allclose(out1[:, 2, :], out2[:, 2, :]))
        self.assertFalse(torch.allclose(out1[:, 1, :], out2[:, 1, :]))

    def test_batch_independence(self):
        # 5. BATCH INDEPENDENCE
        B, T = 3, 2
        x1 = torch.randn(B, T, self.embed_dim, device=self.device)
        x2 = x1.clone()
        x2[1, :, :] += 10.0

        out1 = self.head(x1)
        out2 = self.head(x2)

        self.assertTrue(torch.allclose(out1[0], out2[0]))
        self.assertTrue(torch.allclose(out1[2], out2[2]))
        self.assertFalse(torch.allclose(out1[1], out2[1]))

    def test_input_dependence(self):
        # 6. INPUT DEPENDENCE
        x1 = torch.randn(1, 2, self.embed_dim, device=self.device)
        x2 = x1 + 1.0

        out1 = self.head(x1)
        out2 = self.head(x2)

        self.assertFalse(torch.allclose(out1, out2))

    def test_gradients(self):
        # 7. GRADIENTS
        x = torch.randn(2, 4, self.embed_dim, requires_grad=True, device=self.device)
        out = self.head(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.head.linear.weight.grad)
        self.assertIsNotNone(self.head.linear.bias.grad)

        self.assertTrue(torch.any(x.grad != 0))
        self.assertTrue(torch.any(self.head.linear.weight.grad != 0))
        self.assertTrue(torch.any(self.head.linear.bias.grad != 0))

    def test_parameter_structure(self):
        # 8. PARAMETER STRUCTURE
        params = list(self.head.parameters())
        self.assertEqual(len(params), 2)  # weight and bias
        self.assertEqual(params[0].shape, (self.vocab_size, self.embed_dim))  # weight
        self.assertEqual(params[1].shape, (self.vocab_size,))  # bias

    def test_device_cpu(self):
        # 9. DEVICE CPU
        model = LanguageModelHead(16, 100).to('cpu')
        x = torch.randn(2, 4, 16, device='cpu')
        out = model(x)
        self.assertEqual(out.device.type, 'cpu')

    def test_device_cuda(self):
        # 9. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        model = LanguageModelHead(16, 100).to('cuda')
        x = torch.randn(2, 4, 16, device='cuda')
        out = model(x)
        self.assertEqual(out.device.type, 'cuda')

    def test_edge_cases(self):
        # 10. EDGE CASES
        cases = [
            (1, 1, 1, 1),
            (1, 1, 4, 10),
            (2, 3, 2, 5),
        ]
        for B, T, D, V in cases:
            model = LanguageModelHead(D, V).to(self.device)
            x = torch.randn(B, T, D, device=self.device)
            out = model(x)
            self.assertEqual(out.shape, (B, T, V))

    def test_numerical_validity(self):
        # 11. NUMERICAL VALIDITY
        x = torch.randn(2, 4, self.embed_dim, device=self.device)
        out = self.head(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_linear_behavior(self):
        # 13. LINEAR BEHAVIOR — verify linearity: f(ax) = a*f(x) - (a-1)*bias
        x = torch.randn(1, 2, self.embed_dim, device=self.device)
        a = 2.0

        with torch.no_grad():
            out_x = self.head(x)
            out_ax = self.head(a * x)
            bias = self.head.linear.bias
            # f(ax) = a * (x @ W.T) + b = a * (f(x) - b) + b = a*f(x) - (a-1)*b
            expected = a * out_x - (a - 1) * bias

        self.assertTrue(torch.allclose(out_ax, expected, atol=1e-4))

if __name__ == "__main__":
    unittest.main()
