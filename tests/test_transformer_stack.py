import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.transformer_stack import TransformerStack
from src.model.transformer_block import TransformerBlock

class TestTransformerStack(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.embed_dim = 16
        self.num_heads = 4
        self.ff_dim = 64
        self.num_layers = 3
        self.stack = TransformerStack(
            self.embed_dim, self.num_heads, self.ff_dim, self.num_layers
        ).to(self.device)

    def test_output_shape(self):
        # 1. OUTPUT SHAPE
        cases = [
            (1, 4),
            (2, 1),
            (3, 8),
        ]
        for B, T in cases:
            x = torch.randn(B, T, self.embed_dim, device=self.device)
            out = self.stack(x)
            self.assertEqual(out.shape, (B, T, self.embed_dim))

    def test_num_layers(self):
        # 2. NUM_LAYERS
        for n in [1, 2, 4]:
            stack = TransformerStack(self.embed_dim, self.num_heads, self.ff_dim, n)
            self.assertEqual(len(stack.blocks), n)
            for block in stack.blocks:
                self.assertIsInstance(block, TransformerBlock)

    def test_parameter_registration(self):
        # 3. PARAMETER REGISTRATION
        # All parameters from every block should be discoverable via named_parameters
        param_names = [name for name, _ in self.stack.named_parameters()]
        
        # Each block should have parameters prefixed with blocks.<index>
        for i in range(self.num_layers):
            prefix = f"blocks.{i}."
            block_params = [n for n in param_names if n.startswith(prefix)]
            self.assertTrue(len(block_params) > 0, f"No parameters found for block {i}")

    def test_independent_reference(self):
        # 4. INDEPENDENT REFERENCE
        torch.manual_seed(99)
        stack = TransformerStack(self.embed_dim, self.num_heads, self.ff_dim, 2).to(self.device)
        x = torch.randn(2, 4, self.embed_dim, device=self.device)
        
        out = stack(x)
        
        # Independently pass through each block
        with torch.no_grad():
            y = stack.blocks[0](x)
            y = stack.blocks[1](y)
        
        self.assertTrue(torch.allclose(out, y, atol=1e-5))

    def test_order_of_execution(self):
        # 5. ORDER OF EXECUTION
        torch.manual_seed(42)
        stack = TransformerStack(self.embed_dim, self.num_heads, self.ff_dim, 2).to(self.device)
        x = torch.randn(1, 3, self.embed_dim, device=self.device)
        
        out_normal = stack(x)
        
        # Swap block order and compute
        with torch.no_grad():
            y_swapped = stack.blocks[1](x)
            y_swapped = stack.blocks[0](y_swapped)
        
        self.assertFalse(torch.allclose(out_normal, y_swapped, atol=1e-5))

    def test_single_block_equivalence(self):
        # 6. SINGLE-BLOCK EQUIVALENCE
        torch.manual_seed(42)
        stack = TransformerStack(self.embed_dim, self.num_heads, self.ff_dim, 1).to(self.device)
        x = torch.randn(2, 4, self.embed_dim, device=self.device)
        
        stack_out = stack(x)
        block_out = stack.blocks[0](x)
        
        self.assertTrue(torch.allclose(stack_out, block_out, atol=1e-5))

    def test_gradient_flow(self):
        # 7. GRADIENT FLOW
        x = torch.randn(2, 4, self.embed_dim, requires_grad=True, device=self.device)
        
        out = self.stack(x)
        loss = out.sum()
        loss.backward()
        
        # Input gradients
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))
        
        # Every block should receive gradients
        for i, block in enumerate(self.stack.blocks):
            for name, param in block.named_parameters():
                self.assertIsNotNone(param.grad, f"No gradient for block {i} param {name}")
                self.assertTrue(torch.any(param.grad != 0), f"Zero gradient for block {i} param {name}")

    def test_causal_behavior(self):
        # 8. CAUSAL BEHAVIOR
        B, T = 2, 5
        x1 = torch.randn(B, T, self.embed_dim, device=self.device)
        x2 = x1.clone()
        x2[:, 3:, :] += 10.0
        
        out1 = self.stack(x1)
        out2 = self.stack(x2)
        
        self.assertTrue(torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5))
        self.assertFalse(torch.allclose(out1[:, 3:, :], out2[:, 3:, :], atol=1e-5))

    def test_batch_independence(self):
        # 9. BATCH INDEPENDENCE
        B, T = 3, 4
        x1 = torch.randn(B, T, self.embed_dim, device=self.device)
        x2 = x1.clone()
        x2[1, :, :] += 10.0
        
        out1 = self.stack(x1)
        out2 = self.stack(x2)
        
        self.assertTrue(torch.allclose(out1[0], out2[0]))
        self.assertTrue(torch.allclose(out1[2], out2[2]))
        self.assertFalse(torch.allclose(out1[1], out2[1]))

    def test_device_cpu(self):
        # 10. DEVICE CPU
        x = torch.randn(2, 4, self.embed_dim, device='cpu')
        out = self.stack(x)
        self.assertEqual(out.device.type, 'cpu')

    def test_device_cuda(self):
        # 10. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        stack = TransformerStack(self.embed_dim, self.num_heads, self.ff_dim, 2).to('cuda')
        x = torch.randn(2, 4, self.embed_dim, device='cuda')
        out = stack(x)
        self.assertEqual(out.device.type, 'cuda')

    def test_invalid_configurations(self):
        # 11. INVALID CONFIGURATIONS
        with self.assertRaises(ValueError):
            TransformerStack(16, 4, 64, 0)
        with self.assertRaises(ValueError):
            TransformerStack(0, 4, 64, 2)
        with self.assertRaises(ValueError):
            TransformerStack(16, 0, 64, 2)
        with self.assertRaises(ValueError):
            TransformerStack(16, 4, 0, 2)
        with self.assertRaises(ValueError):
            TransformerStack(16, 3, 64, 2)  # indivisible

    def test_numerical_validity(self):
        # 12. NUMERICAL VALIDITY
        x = torch.randn(2, 4, self.embed_dim, device=self.device)
        out = self.stack(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_no_unexpected_parameters(self):
        # 14. NO UNEXPECTED PARAMETERS
        for name, _ in self.stack.named_parameters():
            self.assertTrue(name.startswith("blocks."), f"Unexpected parameter: {name}")

if __name__ == "__main__":
    unittest.main()
