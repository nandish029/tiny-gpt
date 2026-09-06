import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.embeddings import TokenEmbedding
from src.utils.device import get_device

class TestTokenEmbeddings(unittest.TestCase):
    def setUp(self):
        # Deterministic seed for reproducible tests
        torch.manual_seed(42)
        self.vocab_size = 100
        self.embed_dim = 16
        self.device = torch.device('cpu')
        self.embedding = TokenEmbedding(self.vocab_size, self.embed_dim).to(self.device)

    def test_output_shape(self):
        batch_size = 4
        seq_len = 8
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        out = self.embedding(x)
        self.assertEqual(out.shape, (batch_size, seq_len, self.embed_dim))

    def test_output_dtype(self):
        x = torch.tensor([[1, 2, 3]], device=self.device)
        out = self.embedding(x)
        self.assertTrue(out.is_floating_point())

    def test_different_ids_different_embeddings(self):
        x = torch.tensor([[1, 2]], device=self.device)
        out = self.embedding(x)
        emb_1 = out[0, 0]
        emb_2 = out[0, 1]
        self.assertFalse(torch.allclose(emb_1, emb_2))

    def test_repeated_ids_identical_embeddings(self):
        x = torch.tensor([[5, 5]], device=self.device)
        out = self.embedding(x)
        emb_1 = out[0, 0]
        emb_2 = out[0, 1]
        self.assertTrue(torch.allclose(emb_1, emb_2))

    def test_invalid_ids_raise_error(self):
        # Negative ID
        x_neg = torch.tensor([[-1]], device=self.device)
        with self.assertRaises(IndexError):
            self.embedding(x_neg)
            
        # Out of bounds ID
        x_out = torch.tensor([[self.vocab_size]], device=self.device)
        with self.assertRaises(IndexError):
            self.embedding(x_out)

    def test_batch_size_one(self):
        x = torch.tensor([[10, 11, 12]], device=self.device)
        out = self.embedding(x)
        self.assertEqual(out.shape, (1, 3, self.embed_dim))

    def test_different_sequence_lengths(self):
        x1 = torch.tensor([[1, 2]], device=self.device)
        x2 = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
        
        out1 = self.embedding(x1)
        out2 = self.embedding(x2)
        
        self.assertEqual(out1.shape, (1, 2, self.embed_dim))
        self.assertEqual(out2.shape, (1, 5, self.embed_dim))

    def test_cpu_operation_works(self):
        embedding_cpu = TokenEmbedding(self.vocab_size, self.embed_dim).to('cpu')
        x_cpu = torch.randint(0, self.vocab_size, (2, 5), device='cpu')
        out_cpu = embedding_cpu(x_cpu)
        self.assertEqual(out_cpu.device.type, 'cpu')

    def test_cuda_operation_works(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
            
        embedding_cuda = TokenEmbedding(self.vocab_size, self.embed_dim).to('cuda')
        x_cuda = torch.randint(0, self.vocab_size, (2, 5), device='cuda')
        out_cuda = embedding_cuda(x_cuda)
        self.assertEqual(out_cuda.device.type, 'cuda')

    def test_gradients(self):
        # Ensure gradients are zeroed before test
        self.embedding.zero_grad()
        
        x = torch.tensor([[4, 5, 6]], device=self.device)
        out = self.embedding(x)
        
        loss = out.sum()
        loss.backward()
        
        # Verify gradients exist for the embedding weights
        grad = self.embedding.embedding.weight.grad
        self.assertIsNotNone(grad)
        
        # We only updated indices 4, 5, 6. Let's verify they received gradients
        # and other indices (like 0, 1, 2) did not (i.e. they are zero).
        self.assertTrue(torch.any(grad[4] != 0))
        self.assertTrue(torch.any(grad[5] != 0))
        self.assertTrue(torch.any(grad[6] != 0))
        self.assertTrue(torch.all(grad[0] == 0))

if __name__ == "__main__":
    unittest.main()
