import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.gpt import GPT
from src.model.gpt_embeddings import GPTInputEmbedding
from src.model.transformer_stack import TransformerStack
from src.model.final_normalization import FinalLayerNorm
from src.model.lm_head import LanguageModelHead

class TestGPT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.vocab_size = 50
        self.embed_dim = 16
        self.max_len = 32
        self.num_heads = 4
        self.ff_dim = 64
        self.num_layers = 2
        self.model = GPT(
            self.vocab_size, self.embed_dim, self.max_len,
            self.num_heads, self.ff_dim, self.num_layers
        ).to(self.device)

    def test_end_to_end_shape(self):
        # 1. END-TO-END SHAPE
        cases = [
            (1, 4, 50),
            (2, 1, 50),
            (3, 8, 50),
        ]
        for B, T, V in cases:
            ids = torch.randint(0, V, (B, T), device=self.device)
            out = self.model(ids)
            self.assertEqual(out.shape, (B, T, self.vocab_size))

    def test_different_vocab_sizes(self):
        # 1 continued — multiple vocabulary sizes
        for V in [10, 100]:
            model = GPT(V, 16, 32, 4, 64, 2).to(self.device)
            ids = torch.randint(0, V, (2, 4), device=self.device)
            out = model(ids)
            self.assertEqual(out.shape, (2, 4, V))

    def test_independent_assembly_reference(self):
        # 2. INDEPENDENT ASSEMBLY REFERENCE
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=self.device)
        out = self.model(ids)

        with torch.no_grad():
            x = self.model.embedding(ids)
            x = self.model.transformer_stack(x)
            x = self.model.final_norm(x)
            expected = self.model.lm_head(x)

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_component_connection(self):
        # 3. COMPONENT CONNECTION
        self.assertIsInstance(self.model.embedding, GPTInputEmbedding)
        self.assertIsInstance(self.model.transformer_stack, TransformerStack)
        self.assertIsInstance(self.model.final_norm, FinalLayerNorm)
        self.assertIsInstance(self.model.lm_head, LanguageModelHead)

        # Verify configurations match
        self.assertEqual(self.model.embedding.vocab_size, self.vocab_size)
        self.assertEqual(self.model.embedding.embedding_dim, self.embed_dim)
        self.assertEqual(self.model.embedding.max_context_length, self.max_len)
        self.assertEqual(len(self.model.transformer_stack.blocks), self.num_layers)

    def test_context_length(self):
        # 4. CONTEXT LENGTH
        # T < max
        ids = torch.randint(0, self.vocab_size, (1, 4), device=self.device)
        _ = self.model(ids)

        # T == max
        ids = torch.randint(0, self.vocab_size, (1, self.max_len), device=self.device)
        _ = self.model(ids)

        # T > max
        ids = torch.randint(0, self.vocab_size, (1, self.max_len + 1), device=self.device)
        with self.assertRaises(ValueError):
            self.model(ids)

    def test_raw_logits(self):
        # 5. RAW LOGITS
        ids = torch.randint(0, self.vocab_size, (2, 4), device=self.device)
        out = self.model(ids)

        sums = out.sum(dim=-1)
        ones = torch.ones_like(sums)
        self.assertFalse(torch.allclose(sums, ones, atol=1e-2))
        self.assertTrue((out < 0).any())

    def test_input_dependence(self):
        # 6. INPUT DEPENDENCE
        ids1 = torch.tensor([[1, 2, 3]], device=self.device)
        ids2 = torch.tensor([[1, 99 % self.vocab_size, 3]], device=self.device)

        out1 = self.model(ids1)
        out2 = self.model(ids2)

        self.assertFalse(torch.allclose(out1, out2))

    def test_batch_independence(self):
        # 7. BATCH INDEPENDENCE
        ids1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=self.device)
        ids2 = ids1.clone()
        ids2[1, :] = torch.tensor([10, 11, 12])

        out1 = self.model(ids1)
        out2 = self.model(ids2)

        self.assertTrue(torch.allclose(out1[0], out2[0]))
        self.assertTrue(torch.allclose(out1[2], out2[2]))
        self.assertFalse(torch.allclose(out1[1], out2[1]))

    def test_position_sequence_behavior(self):
        # 8. POSITION/SEQUENCE BEHAVIOR
        B, T = 2, 5
        ids = torch.randint(0, self.vocab_size, (B, T), device=self.device)
        out = self.model(ids)

        self.assertEqual(out.shape[0], B)
        self.assertEqual(out.shape[1], T)
        self.assertEqual(out.shape[2], self.vocab_size)

    def test_gradient_flow(self):
        # 9. GRADIENT FLOW
        ids = torch.randint(0, self.vocab_size, (2, 4), device=self.device)
        out = self.model(ids)
        loss = out.sum()
        loss.backward()

        # Token embedding
        self.assertIsNotNone(self.model.embedding.token_embedding.embedding.weight.grad)
        self.assertTrue(torch.any(self.model.embedding.token_embedding.embedding.weight.grad != 0))

        # Positional embedding
        self.assertIsNotNone(self.model.embedding.positional_embedding.embedding.weight.grad)
        self.assertTrue(torch.any(self.model.embedding.positional_embedding.embedding.weight.grad != 0))

        # Attention Q/K/V from first block
        block0 = self.model.transformer_stack.blocks[0]
        self.assertIsNotNone(block0.attn.qkv_proj.q_proj.weight.grad)
        self.assertIsNotNone(block0.attn.qkv_proj.k_proj.weight.grad)
        self.assertIsNotNone(block0.attn.qkv_proj.v_proj.weight.grad)
        self.assertIsNotNone(block0.attn.out_proj.weight.grad)

        # FFN from first block
        self.assertIsNotNone(block0.ffn.linear1.weight.grad)
        self.assertIsNotNone(block0.ffn.linear2.weight.grad)

        # Block LayerNorms
        self.assertIsNotNone(block0.ln_1.ln.weight.grad)
        self.assertIsNotNone(block0.ln_2.ln.weight.grad)

        # Final LayerNorm
        self.assertIsNotNone(self.model.final_norm.ln.ln.weight.grad)

        # LM head
        self.assertIsNotNone(self.model.lm_head.linear.weight.grad)

    def test_parameter_registration(self):
        # 10. PARAMETER REGISTRATION
        all_names = [n for n, _ in self.model.named_parameters()]

        # All parameters should belong to known components
        valid_prefixes = ["embedding.", "transformer_stack.", "final_norm.", "lm_head."]
        for name in all_names:
            self.assertTrue(
                any(name.startswith(p) for p in valid_prefixes),
                f"Unexpected parameter: {name}"
            )

    def test_deterministic_forward(self):
        # 11. DETERMINISTIC FORWARD
        ids = torch.randint(0, self.vocab_size, (2, 4), device=self.device)
        out1 = self.model(ids)
        out2 = self.model(ids)
        self.assertTrue(torch.allclose(out1, out2))

    def test_device_cpu(self):
        # 12. DEVICE CPU
        ids = torch.randint(0, self.vocab_size, (2, 4), device='cpu')
        out = self.model(ids)
        self.assertEqual(out.device.type, 'cpu')

    def test_device_cuda(self):
        # 12. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        model = GPT(self.vocab_size, self.embed_dim, self.max_len,
                     self.num_heads, self.ff_dim, self.num_layers).to('cuda')
        ids = torch.randint(0, self.vocab_size, (2, 4), device='cuda')
        out = model(ids)
        self.assertEqual(out.device.type, 'cuda')

    def test_edge_cases(self):
        # 13. EDGE CASES
        cases = [
            (1, 1, 10, 2, 2, 8, 1),   # B=1, T=1, V=10, D=2, H=2, FF=8, L=1
            (2, 4, 20, 4, 4, 16, 1),   # small config
        ]
        for B, T, V, D, H, FF, L in cases:
            model = GPT(V, D, 32, H, FF, L).to(self.device)
            ids = torch.randint(0, V, (B, T), device=self.device)
            out = model(ids)
            self.assertEqual(out.shape, (B, T, V))

    def test_numerical_validity(self):
        # 14. NUMERICAL VALIDITY
        ids = torch.randint(0, self.vocab_size, (2, 4), device=self.device)
        out = self.model(ids)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_causal_behavior(self):
        # 15. CAUSAL BEHAVIOR
        ids1 = torch.tensor([[1, 2, 3, 4]], device=self.device)
        ids2 = torch.tensor([[1, 2, 40 % self.vocab_size, 4]], device=self.device)

        out1 = self.model(ids1)
        out2 = self.model(ids2)

        # Positions 0 and 1 must be unchanged
        self.assertTrue(torch.allclose(out1[:, :2, :], out2[:, :2, :], atol=1e-5))
        # Position 2 should change
        self.assertFalse(torch.allclose(out1[:, 2, :], out2[:, 2, :]))

    def test_no_softmax(self):
        # 16. NO SOFTMAX
        ids = torch.randint(0, self.vocab_size, (1, 4), device=self.device)
        out = self.model(ids)

        # Softmax outputs are in [0, 1]; raw logits can be outside
        self.assertTrue((out > 1.0).any() or (out < 0.0).any())

if __name__ == "__main__":
    unittest.main()
