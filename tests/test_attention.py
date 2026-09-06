import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.attention import QKVProjections, ScaledDotProductAttention, MultiHeadSelfAttention

class TestQKVProjections(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.embed_dim = 16
        self.device = torch.device('cpu')
        self.qkv = QKVProjections(self.embed_dim).to(self.device)

    def test_output_shapes(self):
        B, T = 4, 8
        x = torch.randn(B, T, self.embed_dim, device=self.device)
        q, k, v = self.qkv(x)
        
        self.assertEqual(q.shape, (B, T, self.embed_dim))
        self.assertEqual(k.shape, (B, T, self.embed_dim))
        self.assertEqual(v.shape, (B, T, self.embed_dim))

    def test_different_projections_independence(self):
        B, T = 2, 4
        x = torch.randn(B, T, self.embed_dim, device=self.device)
        
        # Initial outputs
        q1, k1, v1 = self.qkv(x)
        
        # Modify Q projection
        with torch.no_grad():
            self.qkv.q_proj.weight += 1.0
            
        q2, k2, v2 = self.qkv(x)
        
        # Q should change
        self.assertFalse(torch.allclose(q1, q2))
        # K and V should remain unchanged
        self.assertTrue(torch.allclose(k1, k2))
        self.assertTrue(torch.allclose(v1, v2))
        
        # Modify K projection
        with torch.no_grad():
            self.qkv.k_proj.weight += 1.0
            
        q3, k3, v3 = self.qkv(x)
        self.assertTrue(torch.allclose(q2, q3)) # Q unmodified this step
        self.assertFalse(torch.allclose(k2, k3)) # K changed
        self.assertTrue(torch.allclose(v2, v3)) # V unmodified this step
        
        # Modify V projection
        with torch.no_grad():
            self.qkv.v_proj.weight += 1.0
            
        q4, k4, v4 = self.qkv(x)
        self.assertTrue(torch.allclose(q3, q4)) # Q unmodified this step
        self.assertTrue(torch.allclose(k3, k4)) # K unmodified this step
        self.assertFalse(torch.allclose(v3, v4)) # V changed

    def test_input_dependence(self):
        B, T = 2, 4
        x1 = torch.randn(B, T, self.embed_dim, device=self.device)
        x2 = torch.randn(B, T, self.embed_dim, device=self.device)
        
        q1, k1, v1 = self.qkv(x1)
        q2, k2, v2 = self.qkv(x2)
        
        self.assertFalse(torch.allclose(q1, q2))
        self.assertFalse(torch.allclose(k1, k2))
        self.assertFalse(torch.allclose(v1, v2))

    def test_batch_and_sequence_support(self):
        shapes_to_test = [
            (1, 5),   # B=1, sequence of 5
            (8, 1),   # B=8, sequence of 1
            (1, 1),   # B=1, T=1
            (16, 32)  # B>1, T>1
        ]
        for B, T in shapes_to_test:
            x = torch.randn(B, T, self.embed_dim, device=self.device)
            q, k, v = self.qkv(x)
            
            self.assertEqual(q.shape, (B, T, self.embed_dim))
            self.assertEqual(k.shape, (B, T, self.embed_dim))
            self.assertEqual(v.shape, (B, T, self.embed_dim))

    def test_gradients(self):
        self.qkv.zero_grad()
        
        x = torch.randn(2, 4, self.embed_dim, device=self.device)
        q, k, v = self.qkv(x)
        
        # Scalar loss
        loss = q.sum() + k.sum() + v.sum()
        loss.backward()
        
        # Verify gradients exist and are non-zero for Q, K, V
        self.assertIsNotNone(self.qkv.q_proj.weight.grad)
        self.assertIsNotNone(self.qkv.k_proj.weight.grad)
        self.assertIsNotNone(self.qkv.v_proj.weight.grad)
        
        self.assertTrue(torch.any(self.qkv.q_proj.weight.grad != 0))
        self.assertTrue(torch.any(self.qkv.k_proj.weight.grad != 0))
        self.assertTrue(torch.any(self.qkv.v_proj.weight.grad != 0))
        
        # Verify biases get gradients too
        self.assertIsNotNone(self.qkv.q_proj.bias.grad)
        self.assertTrue(torch.any(self.qkv.q_proj.bias.grad != 0))

    def test_cpu_operation(self):
        qkv_cpu = QKVProjections(self.embed_dim).to('cpu')
        x = torch.randn(2, 4, self.embed_dim, device='cpu')
        q, k, v = qkv_cpu(x)
        
        self.assertEqual(q.device.type, 'cpu')
        self.assertEqual(k.device.type, 'cpu')
        self.assertEqual(v.device.type, 'cpu')

    def test_cuda_operation(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
            
        qkv_cuda = QKVProjections(self.embed_dim).to('cuda')
        x = torch.randn(2, 4, self.embed_dim, device='cuda')
        q, k, v = qkv_cuda(x)
        
        self.assertEqual(q.device.type, 'cuda')
        self.assertEqual(k.device.type, 'cuda')
        self.assertEqual(v.device.type, 'cuda')

class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.attention = ScaledDotProductAttention().to(self.device)

    def test_shape_and_numerical_stability(self):
        B, Tq, Tk, D = 4, 8, 10, 16
        q = torch.randn(B, Tq, D, device=self.device)
        k = torch.randn(B, Tk, D, device=self.device)
        v = torch.randn(B, Tk, D, device=self.device)
        
        out, weights = self.attention(q, k, v)
        
        # 1. SHAPE TEST
        self.assertEqual(out.shape, (B, Tq, D))
        self.assertEqual(weights.shape, (B, Tq, Tk))
        
        # 11. NUMERICAL STABILITY
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        self.assertFalse(torch.isnan(weights).any())
        self.assertFalse(torch.isinf(weights).any())

    def test_softmax_property_and_non_negativity(self):
        B, Tq, Tk, D = 2, 4, 6, 8
        q = torch.randn(B, Tq, D, device=self.device)
        k = torch.randn(B, Tk, D, device=self.device)
        v = torch.randn(B, Tk, D, device=self.device)
        
        _, weights = self.attention(q, k, v)
        
        # 2. SOFTMAX PROPERTY: sum across Tk dimension must be ~1
        sums = weights.sum(dim=-1)
        expected_sums = torch.ones_like(sums)
        self.assertTrue(torch.allclose(sums, expected_sums, atol=1e-5))
        
        # 3. NON-NEGATIVITY: every weight >= 0
        self.assertTrue((weights >= -1e-6).all())

    def test_independent_reference_implementation(self):
        B, Tq, Tk, D = 1, 2, 3, 4
        # 4. INDEPENDENT REFERENCE IMPLEMENTATION
        # Deterministic inputs
        q = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]]])
        k = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]]])
        v = torch.tensor([[[10.0, 0.0, 0.0, 0.0],
                           [20.0, 0.0, 0.0, 0.0],
                           [30.0, 0.0, 0.0, 0.0]]])

        out, weights = self.attention(q, k, v)
        
        # Manual calculation
        import math
        scores = q @ k.transpose(-2, -1)
        scaled_scores = scores / math.sqrt(D)
        expected_weights = torch.softmax(scaled_scores, dim=-1)
        expected_output = expected_weights @ v
        
        self.assertTrue(torch.allclose(weights, expected_weights, atol=1e-5))
        self.assertTrue(torch.allclose(out, expected_output, atol=1e-5))

    def test_scaling(self):
        B, Tq, Tk, D = 1, 1, 2, 64
        import math
        # 5. SCALING TEST
        q = torch.zeros(B, Tq, D)
        q[0, 0, 0] = 1.0  # Q is [1, 0, 0...]
        
        k = torch.zeros(B, Tk, D)
        k[0, 0, 0] = 4.0  # K0 is [4, 0, 0...] -> score = 4.0
        k[0, 1, 0] = 0.0  # K1 is [0, 0, 0...] -> score = 0.0
        
        v = torch.randn(B, Tk, D)
        
        _, weights = self.attention(q, k, v)
        
        unscaled_scores = q @ k.transpose(-2, -1)
        unscaled_weights = torch.softmax(unscaled_scores, dim=-1)
        
        # Verify that scaled weights do NOT equal unscaled weights
        self.assertFalse(torch.allclose(weights, unscaled_weights, atol=1e-3))
        
        # Verify it DOES equal properly scaled
        scaled_scores = unscaled_scores / math.sqrt(D)
        scaled_ref_weights = torch.softmax(scaled_scores, dim=-1)
        self.assertTrue(torch.allclose(weights, scaled_ref_weights, atol=1e-5))

    def test_value_dependence(self):
        # 6. VALUE DEPENDENCE
        B, Tq, Tk, D = 2, 2, 2, 4
        q = torch.randn(B, Tq, D, device=self.device)
        k = torch.randn(B, Tk, D, device=self.device)
        v1 = torch.randn(B, Tk, D, device=self.device)
        v2 = torch.randn(B, Tk, D, device=self.device)
        
        out1, weights1 = self.attention(q, k, v1)
        out2, weights2 = self.attention(q, k, v2)
        
        # Output changes
        self.assertFalse(torch.allclose(out1, out2))
        # Weights unchanged
        self.assertTrue(torch.allclose(weights1, weights2))

    def test_query_key_dependence(self):
        # 7. QUERY/KEY DEPENDENCE
        B, Tq, Tk, D = 2, 2, 2, 4
        q1 = torch.randn(B, Tq, D, device=self.device)
        q2 = torch.randn(B, Tq, D, device=self.device)
        k1 = torch.randn(B, Tk, D, device=self.device)
        k2 = torch.randn(B, Tk, D, device=self.device)
        v = torch.randn(B, Tk, D, device=self.device)
        
        out_base, w_base = self.attention(q1, k1, v)
        
        # Change Q
        out_q2, w_q2 = self.attention(q2, k1, v)
        self.assertFalse(torch.allclose(w_base, w_q2))
        self.assertFalse(torch.allclose(out_base, out_q2))
        
        # Change K
        out_k2, w_k2 = self.attention(q1, k2, v)
        self.assertFalse(torch.allclose(w_base, w_k2))
        self.assertFalse(torch.allclose(out_base, out_k2))

    def test_gradient_propagation(self):
        # 8. GRADIENT TEST
        B, Tq, Tk, D = 2, 4, 4, 8
        q = torch.randn(B, Tq, D, requires_grad=True, device=self.device)
        k = torch.randn(B, Tk, D, requires_grad=True, device=self.device)
        v = torch.randn(B, Tk, D, requires_grad=True, device=self.device)
        
        out, _ = self.attention(q, k, v)
        
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        
        self.assertTrue(torch.any(q.grad != 0))
        self.assertTrue(torch.any(k.grad != 0))
        self.assertTrue(torch.any(v.grad != 0))

    def test_edge_cases(self):
        # 9. EDGE CASES
        cases = [
            (1, 4, 4, 8),  # B=1
            (2, 1, 4, 8),  # Tq=1
            (2, 4, 1, 8),  # Tk=1
            (2, 4, 6, 8),  # Tq != Tk
            (2, 4, 4, 1)   # small D
        ]
        for B, Tq, Tk, D in cases:
            q = torch.randn(B, Tq, D, device=self.device)
            k = torch.randn(B, Tk, D, device=self.device)
            v = torch.randn(B, Tk, D, device=self.device)
            
            out, weights = self.attention(q, k, v)
            self.assertEqual(out.shape, (B, Tq, D))
            self.assertEqual(weights.shape, (B, Tq, Tk))

    def test_device_cpu(self):
        # 10. DEVICE CPU
        attention = ScaledDotProductAttention().to('cpu')
        q = torch.randn(2, 4, 8, device='cpu')
        k = torch.randn(2, 4, 8, device='cpu')
        v = torch.randn(2, 4, 8, device='cpu')
        
        out, weights = attention(q, k, v)
        self.assertEqual(out.device.type, 'cpu')
        self.assertEqual(weights.device.type, 'cpu')

    def test_device_cuda(self):
        # 10. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
            
        attention = ScaledDotProductAttention().to('cuda')
        q = torch.randn(2, 4, 8, device='cuda')
        k = torch.randn(2, 4, 8, device='cuda')
        v = torch.randn(2, 4, 8, device='cuda')
        
        out, weights = attention(q, k, v)
        self.assertEqual(out.device.type, 'cuda')
        self.assertEqual(weights.device.type, 'cuda')

class TestCausalMasking(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.attention = ScaledDotProductAttention().to(self.device)

    def test_causal_mask_structure(self):
        B, T, D = 1, 4, 8
        q = torch.randn(B, T, D, device=self.device)
        k = torch.randn(B, T, D, device=self.device)
        v = torch.randn(B, T, D, device=self.device)
        
        _, weights = self.attention(q, k, v, causal=True)
        
        # Weights for j > i should be exactly 0
        self.assertEqual(weights[0, 0, 1].item(), 0.0)
        self.assertEqual(weights[0, 0, 2].item(), 0.0)
        self.assertEqual(weights[0, 0, 3].item(), 0.0)
        
        self.assertEqual(weights[0, 1, 2].item(), 0.0)
        self.assertEqual(weights[0, 1, 3].item(), 0.0)
        
        self.assertEqual(weights[0, 2, 3].item(), 0.0)

    def test_causal_past_positions_possible(self):
        B, T, D = 1, 4, 8
        q = torch.randn(B, T, D, device=self.device)
        k = torch.randn(B, T, D, device=self.device)
        v = torch.randn(B, T, D, device=self.device)
        
        _, weights = self.attention(q, k, v, causal=True)
        
        # Verify valid non-zero attention values exist in allowed positions
        self.assertTrue(weights[0, 1, 0].item() > 0)
        self.assertTrue(weights[0, 1, 1].item() > 0)
        self.assertTrue(weights[0, 2, 0].item() > 0)
        self.assertTrue(weights[0, 3, 3].item() > 0)

    def test_causal_independent_reference(self):
        B, T, D = 1, 3, 4
        q = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]]])
        k = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]]])
        v = torch.tensor([[[10.0, 0.0],
                           [20.0, 0.0],
                           [30.0, 0.0]]])
                           
        out, weights = self.attention(q, k, v, causal=True)
        
        import math
        scores = q @ k.transpose(-2, -1)
        scaled = scores / math.sqrt(D)
        
        # Independent mask
        mask = torch.tensor([[True, False, False],
                             [True, True, False],
                             [True, True, True]])
        scaled[0, ~mask] = float('-inf')
        
        expected_weights = torch.softmax(scaled, dim=-1)
        expected_out = expected_weights @ v
        
        self.assertTrue(torch.allclose(weights, expected_weights, atol=1e-5))
        self.assertTrue(torch.allclose(out, expected_out, atol=1e-5))

    def test_causal_future_information_leak(self):
        B, T, D = 2, 4, 8
        q = torch.randn(B, T, D, device=self.device)
        k = torch.randn(B, T, D, device=self.device)
        v1 = torch.randn(B, T, D, device=self.device)
        
        v2 = v1.clone()
        # Change future values of V completely
        v2[:, 2:, :] += 10.0
        
        out1, _ = self.attention(q, k, v1, causal=True)
        out2, _ = self.attention(q, k, v2, causal=True)
        
        # Changing V at pos 2,3 should NOT change out[..., 0:2, :]
        self.assertTrue(torch.allclose(out1[:, :2, :], out2[:, :2, :]))
        # It SHOULD change out[..., 2:, :]
        self.assertFalse(torch.allclose(out1[:, 2:, :], out2[:, 2:, :]))

    def test_causal_future_k_effect(self):
        B, T, D = 1, 4, 8
        q = torch.randn(B, T, D, device=self.device)
        k1 = torch.randn(B, T, D, device=self.device)
        v = torch.randn(B, T, D, device=self.device)
        
        k2 = k1.clone()
        # Change future K at pos 3
        k2[:, 3, :] += 10.0
        
        out1, w1 = self.attention(q, k1, v, causal=True)
        out2, w2 = self.attention(q, k2, v, causal=True)
        
        # Output at pos 0, 1, 2 should not change
        self.assertTrue(torch.allclose(out1[:, :3, :], out2[:, :3, :]))
        self.assertTrue(torch.allclose(w1[:, :3, :], w2[:, :3, :]))
        
        # Output at pos 3 SHOULD change since pos 3 can attend to pos 3
        self.assertFalse(torch.allclose(out1[:, 3, :], out2[:, 3, :]))

    def test_causal_property_across_random_cases(self):
        import random
        for _ in range(10):
            B = random.randint(1, 4)
            T = random.randint(2, 16)
            D = random.randint(4, 32)
            
            q = torch.randn(B, T, D, device=self.device)
            k = torch.randn(B, T, D, device=self.device)
            v = torch.randn(B, T, D, device=self.device)
            
            _, weights = self.attention(q, k, v, causal=True)
            
            # Check upper triangle
            for i in range(T):
                for j in range(i + 1, T):
                    self.assertEqual(weights[0, i, j].item(), 0.0)

    def test_causal_softmax_property(self):
        B, T, D = 2, 6, 8
        q = torch.randn(B, T, D, device=self.device)
        k = torch.randn(B, T, D, device=self.device)
        v = torch.randn(B, T, D, device=self.device)
        
        _, weights = self.attention(q, k, v, causal=True)
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_causal_gradients(self):
        B, T, D = 2, 4, 8
        q = torch.randn(B, T, D, requires_grad=True, device=self.device)
        k = torch.randn(B, T, D, requires_grad=True, device=self.device)
        v = torch.randn(B, T, D, requires_grad=True, device=self.device)
        
        out, _ = self.attention(q, k, v, causal=True)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        
        self.assertTrue(torch.any(q.grad != 0))
        self.assertTrue(torch.any(k.grad != 0))
        self.assertTrue(torch.any(v.grad != 0))

    def test_causal_edge_cases(self):
        cases = [
            (1, 4, 8),  # B=1
            (2, 1, 8),  # T=1
            (2, 4, 1)   # small D
        ]
        for B, T, D in cases:
            q = torch.randn(B, T, D, device=self.device)
            k = torch.randn(B, T, D, device=self.device)
            v = torch.randn(B, T, D, device=self.device)
            
            out, weights = self.attention(q, k, v, causal=True)
            self.assertEqual(out.shape, (B, T, D))
            self.assertEqual(weights.shape, (B, T, T))

    def test_causal_numerical_validity(self):
        B, T, D = 2, 10, 16
        q = torch.randn(B, T, D, device=self.device)
        k = torch.randn(B, T, D, device=self.device)
        v = torch.randn(B, T, D, device=self.device)
        
        out, weights = self.attention(q, k, v, causal=True)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        self.assertFalse(torch.isnan(weights).any())
        self.assertFalse(torch.isinf(weights).any())

    def test_causal_device_cpu(self):
        attention = ScaledDotProductAttention().to('cpu')
        q = torch.randn(2, 4, 8, device='cpu')
        k = torch.randn(2, 4, 8, device='cpu')
        v = torch.randn(2, 4, 8, device='cpu')
        out, weights = attention(q, k, v, causal=True)
        self.assertEqual(out.device.type, 'cpu')
        self.assertEqual(weights.device.type, 'cpu')

    def test_causal_device_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        attention = ScaledDotProductAttention().to('cuda')
        q = torch.randn(2, 4, 8, device='cuda')
        k = torch.randn(2, 4, 8, device='cuda')
        v = torch.randn(2, 4, 8, device='cuda')
        out, weights = attention(q, k, v, causal=True)
        self.assertEqual(out.device.type, 'cuda')
        self.assertEqual(weights.device.type, 'cuda')

class TestMultiHeadSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.embed_dim = 16
        self.num_heads = 4
        self.mhsa = MultiHeadSelfAttention(self.embed_dim, self.num_heads).to(self.device)

    def test_mhsa_shape_and_numerical_validity(self):
        # 1. SHAPE TEST and 11. NUMERICAL VALIDITY
        cases = [
            (2, 4, 16, 4),  # B=2, T=4, D=16, H=4
            (1, 8, 32, 8),
            (4, 2, 8, 2)
        ]
        for B, T, D, H in cases:
            model = MultiHeadSelfAttention(D, H).to(self.device)
            x = torch.randn(B, T, D, device=self.device)
            out = model(x)
            self.assertEqual(out.shape, (B, T, D))
            self.assertFalse(torch.isnan(out).any())
            self.assertFalse(torch.isinf(out).any())

    def test_invalid_head_configuration(self):
        # 2. INVALID HEAD CONFIGURATION
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(16, 3) # not divisible
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(16, 0) # <= 0
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(0, 2)  # <= 0

    def test_head_splitting_merging(self):
        # 3. HEAD SPLITTING / MERGING
        B, T, D, H = 2, 4, 16, 4
        x = torch.randn(B, T, D, device=self.device)
        # Check overall module output shape
        out = self.mhsa(x)
        self.assertEqual(out.shape, (B, T, D))

    def test_causal_future_information_leak(self):
        # 4 & 5. FUTURE-INFORMATION LEAK TEST (MANDATORY)
        B, T, D = 2, 5, 16
        x1 = torch.randn(B, T, D, device=self.device)
        x2 = x1.clone()
        # Change future values
        x2[:, 3:, :] += 10.0
        
        out1 = self.mhsa(x1)
        out2 = self.mhsa(x2)
        
        # Output at pos 0, 1, 2 must NOT change
        self.assertTrue(torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5))
        # Output at pos 3, 4 SHOULD change
        self.assertFalse(torch.allclose(out1[:, 3:, :], out2[:, 3:, :], atol=1e-5))

    def test_different_heads(self):
        # 6. DIFFERENT HEADS
        B, T, D, H = 1, 2, 16, 4
        x = torch.randn(B, T, D, device=self.device)
        
        out1 = self.mhsa(x)
        
        # Modify the Q projection parameters for ONLY the second head
        # weight is [16, 16]. Let's say head_dim is 4.
        # Head 1 (second head) uses rows 4 to 7 in the output of q_proj.
        with torch.no_grad():
            self.mhsa.qkv_proj.q_proj.weight[4:8, :] += 1.0
            
        out2 = self.mhsa(x)
        
        self.assertFalse(torch.allclose(out1, out2))

    def test_output_projection(self):
        # 7. OUTPUT PROJECTION
        B, T, D = 1, 2, 16
        x = torch.randn(B, T, D, device=self.device)
        
        out1 = self.mhsa(x)
        
        with torch.no_grad():
            self.mhsa.out_proj.weight += 1.0
            
        out2 = self.mhsa(x)
        self.assertFalse(torch.allclose(out1, out2))

    def test_gradients(self):
        # 8. GRADIENTS
        B, T, D = 2, 4, 16
        x = torch.randn(B, T, D, requires_grad=True, device=self.device)
        
        out = self.mhsa(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(self.mhsa.qkv_proj.q_proj.weight.grad)
        self.assertIsNotNone(self.mhsa.qkv_proj.k_proj.weight.grad)
        self.assertIsNotNone(self.mhsa.qkv_proj.v_proj.weight.grad)
        self.assertIsNotNone(self.mhsa.out_proj.weight.grad)
        
        self.assertTrue(torch.any(self.mhsa.qkv_proj.q_proj.weight.grad != 0))
        self.assertTrue(torch.any(self.mhsa.qkv_proj.k_proj.weight.grad != 0))
        self.assertTrue(torch.any(self.mhsa.qkv_proj.v_proj.weight.grad != 0))
        self.assertTrue(torch.any(self.mhsa.out_proj.weight.grad != 0))

    def test_edge_cases(self):
        # 10. EDGE CASES
        cases = [
            (1, 4, 16, 4),  # B=1
            (2, 1, 16, 4),  # T=1
            (2, 4, 16, 1),  # num_heads=1
            (2, 4, 4, 2)    # small D
        ]
        for B, T, D, H in cases:
            model = MultiHeadSelfAttention(D, H).to(self.device)
            x = torch.randn(B, T, D, device=self.device)
            out = model(x)
            self.assertEqual(out.shape, (B, T, D))

    def test_independent_reference(self):
        # 13. INDEPENDENT REFERENCE TEST
        B, T, D, H = 1, 3, 4, 2
        head_dim = D // H
        
        torch.manual_seed(42)
        model = MultiHeadSelfAttention(D, H, causal=True).to(self.device)
        
        x = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]]])
                           
        out = model(x)
        
        # --- Independent Reconstruction ---
        with torch.no_grad():
            Wq = model.qkv_proj.q_proj.weight
            bq = model.qkv_proj.q_proj.bias
            Wk = model.qkv_proj.k_proj.weight
            bk = model.qkv_proj.k_proj.bias
            Wv = model.qkv_proj.v_proj.weight
            bv = model.qkv_proj.v_proj.bias
            Wo = model.out_proj.weight
            bo = model.out_proj.bias
            
            # 1. Q, K, V
            q = x @ Wq.T + bq
            k = x @ Wk.T + bk
            v = x @ Wv.T + bv
            
            # 2. Reshape into heads [B, H, T, head_dim]
            q_heads = q.view(B, T, H, head_dim).transpose(1, 2)
            k_heads = k.view(B, T, H, head_dim).transpose(1, 2)
            v_heads = v.view(B, T, H, head_dim).transpose(1, 2)
            
            # 3. Scaled dot-product + causal mask
            import math
            scores = q_heads @ k_heads.transpose(-2, -1)
            scaled_scores = scores / math.sqrt(head_dim)
            
            mask = torch.tensor([[True, False, False],
                                 [True, True, False],
                                 [True, True, True]])
                                 
            scaled_scores = scaled_scores.masked_fill(~mask, float('-inf'))
            weights = torch.softmax(scaled_scores, dim=-1)
            head_outs = weights @ v_heads
            
            # 4. Concatenate
            concat_outs = head_outs.transpose(1, 2).contiguous().view(B, T, D)
            
            # 5. Final output projection
            expected_out = concat_outs @ Wo.T + bo
            
        self.assertTrue(torch.allclose(out, expected_out, atol=1e-5))

    def test_device_cpu(self):
        # 9. DEVICE CPU
        model = MultiHeadSelfAttention(16, 4).to('cpu')
        x = torch.randn(2, 4, 16, device='cpu')
        out = model(x)
        self.assertEqual(out.device.type, 'cpu')

    def test_device_cuda(self):
        # 9. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        model = MultiHeadSelfAttention(16, 4).to('cuda')
        x = torch.randn(2, 4, 16, device='cuda')
        out = model(x)
        self.assertEqual(out.device.type, 'cuda')

if __name__ == "__main__":
    unittest.main()
