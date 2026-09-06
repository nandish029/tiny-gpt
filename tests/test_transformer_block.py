import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.transformer_block import TransformerBlock
from src.model.attention import MultiHeadSelfAttention
from src.model.feed_forward import FeedForward
from src.model.normalization import LayerNorm
from src.model.residual import ResidualAdd

class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.embed_dim = 16
        self.num_heads = 4
        self.ff_dim = 64
        self.block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim).to(self.device)

    def test_output_shape(self):
        # 1. OUTPUT SHAPE
        cases = [
            (1, 4),
            (2, 1),
            (3, 8)
        ]
        for B, T in cases:
            x = torch.randn(B, T, self.embed_dim, device=self.device)
            out = self.block(x)
            self.assertEqual(out.shape, (B, T, self.embed_dim))

    def test_constructor_validation(self):
        # 2. CONSTRUCTOR VALIDATION
        with self.assertRaises(ValueError):
            TransformerBlock(0, 4, 64)
        with self.assertRaises(ValueError):
            TransformerBlock(16, 0, 64)
        with self.assertRaises(ValueError):
            TransformerBlock(16, 3, 64) # Indivisible
        with self.assertRaises(ValueError):
            TransformerBlock(16, 4, -10)

    def test_component_structure(self):
        # 3. COMPONENT STRUCTURE
        self.assertTrue(isinstance(self.block.ln_1, LayerNorm))
        self.assertTrue(isinstance(self.block.attn, MultiHeadSelfAttention))
        self.assertTrue(isinstance(self.block.res_1, ResidualAdd))
        
        self.assertTrue(isinstance(self.block.ln_2, LayerNorm))
        self.assertTrue(isinstance(self.block.ffn, FeedForward))
        self.assertTrue(isinstance(self.block.res_2, ResidualAdd))
        
        # Verify causal is explicitly true in Attention
        self.assertTrue(self.block.attn.causal)

    def test_independent_reference_assembly(self):
        # 4. INDEPENDENT REFERENCE TEST
        B, T = 2, 4
        x = torch.randn(B, T, self.embed_dim, device=self.device)
        
        out = self.block(x)
        
        # Independently reproduce
        with torch.no_grad():
            norm1 = self.block.ln_1(x)
            attention = self.block.attn(norm1)
            x_attn = x + attention # Native py residual
            
            norm2 = self.block.ln_2(x_attn)
            ffn_out = self.block.ffn(norm2)
            expected = x_attn + ffn_out
            
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_residual_behavior(self):
        # 5. RESIDUAL BEHAVIOR
        B, T = 1, 2
        x = torch.randn(B, T, self.embed_dim, device=self.device)
        
        out = self.block(x)
        
        with torch.no_grad():
            # If we remove the residual paths, it should fail equality
            norm1 = self.block.ln_1(x)
            attention = self.block.attn(norm1)
            # Incorrectly skipping residual:
            norm2 = self.block.ln_2(attention)
            ffn_out = self.block.ffn(norm2)
            
        self.assertFalse(torch.allclose(out, ffn_out, atol=1e-3))
        
        # If we just passed identity through attention/FFN it would equal x
        # So we prove x contributes to out
        self.assertFalse(torch.allclose(out, attention, atol=1e-3))

    def test_pre_layer_norm_order(self):
        # 6. PRE-LAYER-NORM ORDER
        # For a pre-LN architecture, the first operation is LayerNorm.
        # LayerNorm shifts mean to 0. If input has a massive mean shift,
        # it gets zeroed out before hitting attention.
        B, T = 1, 3
        x_base = torch.randn(B, T, self.embed_dim, device=self.device)
        x_shifted = x_base + 1000.0 # huge shift
        
        # In pre-LN, LN(x_shifted) approx equals LN(x_base)
        # So attention output should be nearly identical.
        with torch.no_grad():
            attn_base = self.block.attn(self.block.ln_1(x_base))
            attn_shifted = self.block.attn(self.block.ln_1(x_shifted))
            
        self.assertTrue(torch.allclose(attn_base, attn_shifted, atol=1e-3))
        
        # But the final output x + attn + ffn contains the shifted x, so it won't be equal
        out_base = self.block(x_base)
        out_shifted = self.block(x_shifted)
        self.assertFalse(torch.allclose(out_base, out_shifted, atol=1e-3))

    def test_causal_behavior(self):
        # 7. CAUSAL BEHAVIOR
        B, T = 2, 5
        x1 = torch.randn(B, T, self.embed_dim, device=self.device)
        x2 = x1.clone()
        
        # Change future positions
        x2[:, 3:, :] += 10.0
        
        out1 = self.block(x1)
        out2 = self.block(x2)
        
        # Positions 0, 1, 2 must remain identical
        self.assertTrue(torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5))
        
        # Future positions should change
        self.assertFalse(torch.allclose(out1[:, 3:, :], out2[:, 3:, :], atol=1e-5))

    def test_batch_independence(self):
        # 12. BATCH INDEPENDENCE
        B, T = 3, 2
        x1 = torch.randn(B, T, self.embed_dim, device=self.device)
        x2 = x1.clone()
        
        x2[1, :, :] += 10.0 # Modify batch 1
        
        out1 = self.block(x1)
        out2 = self.block(x2)
        
        self.assertTrue(torch.allclose(out1[0, :, :], out2[0, :, :]))
        self.assertTrue(torch.allclose(out1[2, :, :], out2[2, :, :]))
        self.assertFalse(torch.allclose(out1[1, :, :], out2[1, :, :]))

    def test_gradients(self):
        # 8. GRADIENT FLOW
        x = torch.randn(2, 4, self.embed_dim, requires_grad=True, device=self.device)
        
        out = self.block(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.block.ln_1.ln.weight.grad)
        self.assertIsNotNone(self.block.attn.qkv_proj.q_proj.weight.grad)
        self.assertIsNotNone(self.block.attn.out_proj.weight.grad)
        self.assertIsNotNone(self.block.ln_2.ln.bias.grad)
        self.assertIsNotNone(self.block.ffn.linear1.weight.grad)
        self.assertIsNotNone(self.block.ffn.linear2.weight.grad)
        
        self.assertTrue(torch.any(x.grad != 0))
        self.assertTrue(torch.any(self.block.ln_1.ln.weight.grad != 0))
        self.assertTrue(torch.any(self.block.ffn.linear2.weight.grad != 0))

    def test_numerical_validity(self):
        # 10. NUMERICAL VALIDITY
        x = torch.randn(2, 4, self.embed_dim, device=self.device)
        out = self.block(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_parameter_count_sanity(self):
        # 14. PARAMETER COUNT SANITY
        params = list(self.block.parameters())
        # Expected parameter tensors:
        # ln_1: weight, bias (2)
        # attn: q, k, v projections (weight, bias) * 3 = 6
        # attn output proj: weight, bias (2)
        # ln_2: weight, bias (2)
        # ffn: linear1, linear2 (weight, bias) * 2 = 4
        # Total = 16 tensors
        self.assertEqual(len(params), 16)

    def test_device_cpu(self):
        # 9. DEVICE CPU
        model = TransformerBlock(16, 4, 64).to('cpu')
        x = torch.randn(2, 4, 16, device='cpu')
        out = model(x)
        self.assertEqual(out.device.type, 'cpu')

    def test_device_cuda(self):
        # 9. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        model = TransformerBlock(16, 4, 64).to('cuda')
        x = torch.randn(2, 4, 16, device='cuda')
        out = model(x)
        self.assertEqual(out.device.type, 'cuda')

if __name__ == "__main__":
    unittest.main()
