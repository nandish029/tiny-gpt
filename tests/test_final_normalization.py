import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.final_normalization import FinalLayerNorm
from src.model.transformer_block import TransformerBlock

class TestFinalLayerNorm(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.embed_dim = 16
        self.fln = FinalLayerNorm(self.embed_dim).to(self.device)

    def test_output_shape(self):
        # 1. OUTPUT SHAPE
        cases = [(1, 4), (2, 1), (4, 8)]
        for B, T in cases:
            x = torch.randn(B, T, self.embed_dim, device=self.device)
            out = self.fln(x)
            self.assertEqual(out.shape, (B, T, self.embed_dim))

    def test_independent_reference(self):
        # 2. INDEPENDENT REFERENCE
        B, T, D = 2, 3, 4
        fln = FinalLayerNorm(D).to(self.device)
        with torch.no_grad():
            fln.ln.ln.weight.copy_(torch.randn(D))
            fln.ln.ln.bias.copy_(torch.randn(D))

        x = torch.randn(B, T, D, device=self.device)
        out = fln(x)

        eps = fln.ln.ln.eps
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + eps)
        expected = normalized * fln.ln.ln.weight + fln.ln.ln.bias

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_normalization_property(self):
        # 3. NORMALIZATION PROPERTY
        B, T, D = 4, 4, 16
        fln = FinalLayerNorm(D).to(self.device)
        x = torch.randn(B, T, D, device=self.device) * 10 + 5

        out = fln(x)
        out_mean = out.mean(dim=-1)
        out_var = out.var(dim=-1, unbiased=False)

        self.assertTrue(torch.allclose(out_mean, torch.zeros_like(out_mean), atol=1e-5))
        self.assertTrue(torch.allclose(out_var, torch.ones_like(out_var), atol=1e-4))

    def test_parameter_independence(self):
        # 4. PARAMETER INDEPENDENCE
        fln = FinalLayerNorm(self.embed_dim).to(self.device)
        block = TransformerBlock(self.embed_dim, 4, 64).to(self.device)

        # Record block LN weights before
        block_ln1_weight_before = block.ln_1.ln.weight.clone()
        block_ln2_weight_before = block.ln_2.ln.weight.clone()

        # Modify FinalLayerNorm parameters
        with torch.no_grad():
            fln.ln.ln.weight += 100.0
            fln.ln.ln.bias += 100.0

        # Block LN weights should be unchanged
        self.assertTrue(torch.equal(block.ln_1.ln.weight, block_ln1_weight_before))
        self.assertTrue(torch.equal(block.ln_2.ln.weight, block_ln2_weight_before))

    def test_gradients(self):
        # 5. GRADIENTS
        x = torch.randn(2, 4, self.embed_dim, requires_grad=True, device=self.device)
        out = self.fln(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.fln.ln.ln.weight.grad)
        self.assertIsNotNone(self.fln.ln.ln.bias.grad)
        self.assertTrue(torch.any(x.grad != 0))
        self.assertTrue(torch.any(self.fln.ln.ln.weight.grad != 0))
        self.assertTrue(torch.any(self.fln.ln.ln.bias.grad != 0))

    def test_position_independence(self):
        # 6. POSITION INDEPENDENCE
        B, T, D = 1, 3, 16
        x1 = torch.randn(B, T, D, device=self.device)
        x2 = x1.clone()
        x2[:, 1, 0] += 10.0

        out1 = self.fln(x1)
        out2 = self.fln(x2)

        self.assertTrue(torch.allclose(out1[:, 0, :], out2[:, 0, :]))
        self.assertTrue(torch.allclose(out1[:, 2, :], out2[:, 2, :]))
        self.assertFalse(torch.allclose(out1[:, 1, :], out2[:, 1, :]))

    def test_batch_independence(self):
        # 7. BATCH INDEPENDENCE
        B, T, D = 3, 2, 16
        x1 = torch.randn(B, T, D, device=self.device)
        x2 = x1.clone()
        x2[1, :, 0] += 10.0

        out1 = self.fln(x1)
        out2 = self.fln(x2)

        self.assertTrue(torch.allclose(out1[0], out2[0]))
        self.assertTrue(torch.allclose(out1[2], out2[2]))
        self.assertFalse(torch.allclose(out1[1], out2[1]))

    def test_device_cpu(self):
        # 8. DEVICE CPU
        model = FinalLayerNorm(16).to('cpu')
        x = torch.randn(2, 4, 16, device='cpu')
        out = model(x)
        self.assertEqual(out.device.type, 'cpu')

    def test_device_cuda(self):
        # 8. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        model = FinalLayerNorm(16).to('cuda')
        x = torch.randn(2, 4, 16, device='cuda')
        out = model(x)
        self.assertEqual(out.device.type, 'cuda')

    def test_edge_cases(self):
        # 9. EDGE CASES
        cases = [(1, 1, 1), (1, 4, 2), (2, 1, 16), (3, 5, 8)]
        for B, T, D in cases:
            model = FinalLayerNorm(D).to(self.device)
            x = torch.randn(B, T, D, device=self.device)
            out = model(x)
            self.assertEqual(out.shape, (B, T, D))

    def test_numerical_validity(self):
        # 10. NUMERICAL VALIDITY
        x = torch.randn(2, 4, self.embed_dim, device=self.device)
        out = self.fln(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_no_extra_parameters(self):
        # 12. NO EXTRA PARAMETERS
        params = list(self.fln.parameters())
        self.assertEqual(len(params), 2)  # weight and bias
        self.assertEqual(params[0].shape, (self.embed_dim,))
        self.assertEqual(params[1].shape, (self.embed_dim,))

if __name__ == "__main__":
    unittest.main()
