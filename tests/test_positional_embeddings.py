import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.positional_embeddings import PositionalEmbedding

class TestPositionalEmbeddings(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.max_ctx = 32
        self.embed_dim = 16
        self.device = torch.device('cpu')
        self.pos_emb = PositionalEmbedding(self.max_ctx, self.embed_dim).to(self.device)

    def test_output_shape(self):
        B, T = 4, 8
        x = torch.zeros((B, T), dtype=torch.long, device=self.device)
        out = self.pos_emb(x)
        self.assertEqual(out.shape, (B, T, self.embed_dim))

    def test_batch_size_one(self):
        B, T = 1, 5
        x = torch.zeros((B, T), dtype=torch.long, device=self.device)
        out = self.pos_emb(x)
        self.assertEqual(out.shape, (1, 5, self.embed_dim))

    def test_larger_batch(self):
        B, T = 16, 5
        x = torch.zeros((B, T), dtype=torch.long, device=self.device)
        out = self.pos_emb(x)
        self.assertEqual(out.shape, (16, 5, self.embed_dim))

    def test_different_sequence_lengths(self):
        x1 = torch.zeros((2, 3), dtype=torch.long, device=self.device)
        x2 = torch.zeros((2, 10), dtype=torch.long, device=self.device)
        out1 = self.pos_emb(x1)
        out2 = self.pos_emb(x2)
        self.assertEqual(out1.shape, (2, 3, self.embed_dim))
        self.assertEqual(out2.shape, (2, 10, self.embed_dim))
        # Verify that the first 3 positions of x2 match x1 perfectly
        self.assertTrue(torch.allclose(out1, out2[:, :3, :]))

    def test_position_identity_across_batches(self):
        # Repeated positions across batches produce the same vectors
        B, T = 2, 4
        x = torch.zeros((B, T), dtype=torch.long, device=self.device)
        out = self.pos_emb(x)
        # Vector at b=0, t=2 should be identical to b=1, t=2
        self.assertTrue(torch.allclose(out[0, 2], out[1, 2]))

    def test_independent_of_token_ids(self):
        B, T = 2, 4
        x1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long, device=self.device)
        x2 = torch.tensor([[7, 7, 7, 7], [0, 0, 0, 0]], dtype=torch.long, device=self.device)
        
        out1 = self.pos_emb(x1)
        out2 = self.pos_emb(x2)
        
        # Positional embedding should completely ignore the token ID values
        self.assertTrue(torch.allclose(out1, out2))

    def test_different_positions_different_vectors(self):
        x = torch.zeros((1, 4), dtype=torch.long, device=self.device)
        out = self.pos_emb(x)
        # Verify that pos 0, 1, 2, 3 are mutually different (highly probable with random init)
        self.assertFalse(torch.allclose(out[0, 0], out[0, 1]))
        self.assertFalse(torch.allclose(out[0, 1], out[0, 2]))
        self.assertFalse(torch.allclose(out[0, 2], out[0, 3]))

    def test_reject_length_exceeding_max(self):
        x = torch.zeros((1, self.max_ctx + 1), dtype=torch.long, device=self.device)
        with self.assertRaises(ValueError):
            self.pos_emb(x)

    def test_cpu_operation(self):
        pos_emb_cpu = PositionalEmbedding(self.max_ctx, self.embed_dim).to('cpu')
        x_cpu = torch.zeros((1, 4), dtype=torch.long, device='cpu')
        out = pos_emb_cpu(x_cpu)
        self.assertEqual(out.device.type, 'cpu')

    def test_cuda_operation(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
            
        pos_emb_cuda = PositionalEmbedding(self.max_ctx, self.embed_dim).to('cuda')
        x_cuda = torch.zeros((1, 4), dtype=torch.long, device='cuda')
        out = pos_emb_cuda(x_cuda)
        self.assertEqual(out.device.type, 'cuda')

    def test_gradients_flow(self):
        self.pos_emb.zero_grad()
        
        # Test with a sequence of length 5
        x = torch.zeros((1, 5), dtype=torch.long, device=self.device)
        out = self.pos_emb(x)
        
        loss = out.sum()
        loss.backward()
        
        grad = self.pos_emb.embedding.weight.grad
        self.assertIsNotNone(grad)
        
        # Since we fed sequence length 5, positions 0,1,2,3,4 were accessed.
        # They should have non-zero gradients.
        for i in range(5):
            self.assertTrue(torch.any(grad[i] != 0))
            
        # Positions >= 5 were never accessed and should have zero gradients.
        for i in range(5, self.max_ctx):
            self.assertTrue(torch.all(grad[i] == 0))

if __name__ == "__main__":
    unittest.main()
