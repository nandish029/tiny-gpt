import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.gpt_embeddings import GPTInputEmbedding

class TestGPTInputEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.vocab_size = 100
        self.embed_dim = 16
        self.max_len = 32
        self.model = GPTInputEmbedding(self.vocab_size, self.embed_dim, self.max_len).to(self.device)

    def test_output_shape(self):
        # 1. OUTPUT SHAPE
        cases = [
            (1, 4),   # B=1
            (2, 1),   # T=1
            (4, 16)   # Several T
        ]
        for B, T in cases:
            ids = torch.randint(0, self.vocab_size, (B, T), device=self.device)
            out = self.model(ids)
            self.assertEqual(out.shape, (B, T, self.embed_dim))

    def test_independent_reference(self):
        # 2. INDEPENDENT REFERENCE
        B, T = 2, 4
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=self.device)
        
        out = self.model(ids)
        
        # Independently calculate
        with torch.no_grad():
            W_tok = self.model.token_embedding.embedding.weight
            W_pos = self.model.positional_embedding.embedding.weight
            
            # PyTorch indexing for token IDs
            expected_tok = W_tok[ids]
            
            # PyTorch indexing for positions (0 to T-1)
            positions = torch.arange(T, device=self.device).unsqueeze(0).expand(B, T)
            expected_pos = W_pos[positions]
            
            expected = expected_tok + expected_pos
            
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_token_contribution(self):
        # 3. TOKEN CONTRIBUTION
        ids1 = torch.tensor([[1, 2, 3]], device=self.device)
        ids2 = torch.tensor([[1, 99, 3]], device=self.device)
        
        out1 = self.model(ids1)
        out2 = self.model(ids2)
        
        # position 1 changed
        self.assertTrue(torch.allclose(out1[:, 0, :], out2[:, 0, :]))
        self.assertFalse(torch.allclose(out1[:, 1, :], out2[:, 1, :]))
        self.assertTrue(torch.allclose(out1[:, 2, :], out2[:, 2, :]))

    def test_position_contribution(self):
        # 4. POSITION CONTRIBUTION
        # Identical tokens across positions
        ids = torch.tensor([[5, 5, 5]], device=self.device)
        out = self.model(ids)
        
        # Position 0 should not equal position 1 or 2 despite same token
        self.assertFalse(torch.allclose(out[0, 0, :], out[0, 1, :]))
        self.assertFalse(torch.allclose(out[0, 1, :], out[0, 2, :]))

    def test_position_independence_from_tokens(self):
        # 5. POSITION INDEPENDENCE FROM TOKEN IDs
        ids1 = torch.tensor([[1, 2, 3]], device=self.device)
        ids2 = torch.tensor([[9, 8, 7]], device=self.device)
        
        pos_out1 = self.model.positional_embedding(ids1)
        pos_out2 = self.model.positional_embedding(ids2)
        
        self.assertTrue(torch.allclose(pos_out1, pos_out2))

    def test_component_gradients(self):
        # 6. COMPONENT GRADIENTS
        ids = torch.tensor([[1, 2, 3]], device=self.device)
        out = self.model(ids)
        
        loss = out.sum()
        loss.backward()
        
        # Check gradients reach both embeddings
        tok_grad = self.model.token_embedding.embedding.weight.grad
        pos_grad = self.model.positional_embedding.embedding.weight.grad
        
        self.assertIsNotNone(tok_grad)
        self.assertIsNotNone(pos_grad)
        
        # Expect gradients specifically at the indices accessed
        self.assertTrue(torch.any(tok_grad[1] != 0))
        self.assertTrue(torch.any(pos_grad[0] != 0))

    def test_no_extra_parameters(self):
        # 7. NO EXTRA PARAMETERS
        params = list(self.model.parameters())
        # Token embedding weight, Positional embedding weight
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].shape, (self.vocab_size, self.embed_dim))
        self.assertEqual(params[1].shape, (self.max_len, self.embed_dim))

    def test_context_length(self):
        # 8. CONTEXT LENGTH
        valid_ids = torch.randint(0, self.vocab_size, (1, self.max_len), device=self.device)
        # Should not raise
        _ = self.model(valid_ids)
        
        invalid_ids = torch.randint(0, self.vocab_size, (1, self.max_len + 1), device=self.device)
        with self.assertRaises(ValueError):
            _ = self.model(invalid_ids)

    def test_device_cpu(self):
        # 9. DEVICE CPU
        model = GPTInputEmbedding(self.vocab_size, self.embed_dim, self.max_len).to('cpu')
        ids = torch.tensor([[1, 2]], device='cpu')
        out = model(ids)
        self.assertEqual(out.device.type, 'cpu')

    def test_device_cuda(self):
        # 9. DEVICE CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        model = GPTInputEmbedding(self.vocab_size, self.embed_dim, self.max_len).to('cuda')
        ids = torch.tensor([[1, 2]], device='cuda')
        out = model(ids)
        self.assertEqual(out.device.type, 'cuda')

    def test_numerical_validity(self):
        # 10. NUMERICAL VALIDITY
        ids = torch.randint(0, self.vocab_size, (2, 4), device=self.device)
        out = self.model(ids)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_batch_independence(self):
        # 12. BATCH INDEPENDENCE
        ids1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=self.device)
        ids2 = ids1.clone()
        
        ids2[1, :] = torch.tensor([9, 9, 9]) # modify batch 1
        
        out1 = self.model(ids1)
        out2 = self.model(ids2)
        
        self.assertTrue(torch.allclose(out1[0], out2[0]))
        self.assertTrue(torch.allclose(out1[2], out2[2]))
        self.assertFalse(torch.allclose(out1[1], out2[1]))

if __name__ == "__main__":
    unittest.main()
