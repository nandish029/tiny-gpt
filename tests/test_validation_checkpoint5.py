import unittest
import torch
import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.language_dataset import LanguageDataset

class DummyTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

# 2. ADD INDEPENDENT REFERENCE TESTS
class TestIndependentReference(unittest.TestCase):
    def test_independent_reference(self):
        dataset = LanguageDataset.__new__(LanguageDataset)
        dataset.tokenizer = DummyTokenizer()
        dataset.context_length = 4
        dataset.device = torch.device('cpu')
        
        # We inject a small sequence directly.
        dataset.train_data = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
        
        original_randint = torch.randint
        
        # Based on implementation max_idx = 10 - 4 - 1 = 5
        # torch.randint(0, 5) yields [0, 1, 2, 3, 4]
        for start_idx in range(0, 5):
            # Mock randint to deterministically pick our start_idx
            torch.randint = lambda low, high, size: torch.full(size, start_idx, dtype=torch.long)
            
            x, y = dataset.get_batch("train", batch_size=1)
            
            expected_x = [start_idx, start_idx+1, start_idx+2, start_idx+3]
            expected_y = [start_idx+1, start_idx+2, start_idx+3, start_idx+4]
            
            self.assertEqual(x[0].tolist(), expected_x)
            self.assertEqual(y[0].tolist(), expected_y)
            
        torch.randint = original_randint

# 3. ADD RANDOMIZED PROPERTY TESTS
class TestRandomizedProperties(unittest.TestCase):
    def test_randomized_properties(self):
        random.seed(42)
        torch.manual_seed(42)
        
        dataset = LanguageDataset.__new__(LanguageDataset)
        dataset.tokenizer = DummyTokenizer(vocab_size=100)
        dataset.device = torch.device('cpu')
        
        for _ in range(100):
            context_length = random.randint(2, 32)
            batch_size = random.randint(1, 16)
            # Must be at least context_length + 2 based on implementation max_idx > 0 requirement
            seq_len = context_length + random.randint(2, 100)
            
            dataset.context_length = context_length
            dataset.train_data = torch.randint(0, 100, (seq_len,), dtype=torch.long)
            
            x, y = dataset.get_batch("train", batch_size=batch_size)
            
            # Dimensions
            self.assertEqual(x.shape, (batch_size, context_length))
            self.assertEqual(y.shape, (batch_size, context_length))
            
            # Integer dtype
            self.assertEqual(x.dtype, torch.long)
            self.assertEqual(y.dtype, torch.long)
            
            # Ranges
            self.assertTrue((x >= 0).all() and (x < 100).all())
            self.assertTrue((y >= 0).all() and (y < 100).all())
            
            # Y is X shifted by 1
            self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))

# 4. TEST BOUNDARY CONDITIONS
class TestBoundaryConditions(unittest.TestCase):
    def setUp(self):
        self.dataset = LanguageDataset.__new__(LanguageDataset)
        self.dataset.tokenizer = DummyTokenizer(vocab_size=10)
        self.dataset.device = torch.device('cpu')
        
    def test_smallest_valid_context_length(self):
        self.dataset.context_length = 1
        # Implementation requires len(data) - ctx - 1 > 0 => len(data) >= 3 for context 1
        self.dataset.train_data = torch.tensor([0, 1, 2], dtype=torch.long)
        
        original_randint = torch.randint
        torch.randint = lambda low, high, size: torch.zeros(size, dtype=torch.long)
        
        x, y = self.dataset.get_batch("train", batch_size=1)
        torch.randint = original_randint
        
        self.assertEqual(x.shape, (1, 1))
        self.assertEqual(y.shape, (1, 1))
        self.assertEqual(x[0, 0].item(), 0)
        self.assertEqual(y[0, 0].item(), 1)
        
    def test_batch_size_one(self):
        self.dataset.context_length = 4
        # Need len >= 6
        self.dataset.train_data = torch.tensor([0,1,2,3,4,5], dtype=torch.long)
        x, y = self.dataset.get_batch("train", batch_size=1)
        self.assertEqual(x.shape, (1, 4))
        self.assertEqual(y.shape, (1, 4))
        
    def test_context_length_near_available_sequence_length(self):
        self.dataset.context_length = 4
        # Min length is 6 for context 4
        self.dataset.train_data = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        x, y = self.dataset.get_batch("train", batch_size=2)
        # max_idx = 6 - 4 - 1 = 1, torch.randint(0, 1) always yields 0
        self.assertTrue(torch.equal(x[0], torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.equal(y[0], torch.tensor([1, 2, 3, 4])))
        
    def test_repeated_token_ids(self):
        self.dataset.context_length = 2
        # Need len >= 4
        self.dataset.train_data = torch.tensor([5, 5, 5, 5], dtype=torch.long)
        x, y = self.dataset.get_batch("train", batch_size=1)
        self.assertTrue(torch.equal(x[0], torch.tensor([5, 5])))
        self.assertTrue(torch.equal(y[0], torch.tensor([5, 5])))
        
    def test_zero_and_vocab_size_minus_1(self):
        self.dataset.context_length = 2
        self.dataset.train_data = torch.tensor([0, 9, 0, 9], dtype=torch.long)
        x, y = self.dataset.get_batch("train", batch_size=1)
        self.assertTrue((x >= 0).all() and (x < 10).all())
        self.assertTrue((y >= 0).all() and (y < 10).all())
        
    def test_invalid_configuration_not_enough_tokens(self):
        self.dataset.context_length = 4
        self.dataset.train_data = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        # Length is 5. context_length is 4. According to implementation, max_idx = 5 - 4 - 1 = 0.
        # This raises ValueError.
        with self.assertRaises(ValueError):
            self.dataset.get_batch("train", batch_size=1)

# 5. TEST TRAIN/VALIDATION SEPARATION
class TestTrainValidationSeparation(unittest.TestCase):
    def test_separation(self):
        dataset = LanguageDataset.__new__(LanguageDataset)
        dataset.tokenizer = DummyTokenizer(vocab_size=20)
        dataset.context_length = 4
        dataset.device = torch.device('cpu')
        
        # Train tokens are all < 10
        dataset.train_data = torch.randint(0, 10, (100,), dtype=torch.long)
        # Valid tokens are all >= 10
        dataset.valid_data = torch.randint(10, 20, (100,), dtype=torch.long)
        
        x_train, y_train = dataset.get_batch("train", batch_size=10)
        self.assertTrue((x_train < 10).all())
        self.assertTrue((y_train < 10).all())
        
        x_valid, y_valid = dataset.get_batch("valid", batch_size=10)
        self.assertTrue((x_valid >= 10).all())
        self.assertTrue((y_valid >= 10).all())

# 6. TEST DEVICE BEHAVIOR
class TestDeviceBehavior(unittest.TestCase):
    def test_cpu_behavior(self):
        dataset = LanguageDataset.__new__(LanguageDataset)
        dataset.tokenizer = DummyTokenizer()
        dataset.context_length = 4
        dataset.device = torch.device('cpu')
        dataset.train_data = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        
        x, y = dataset.get_batch("train", batch_size=1)
        self.assertEqual(x.device.type, 'cpu')
        self.assertEqual(y.device.type, 'cpu')
        
    def test_cuda_behavior(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
            
        dataset = LanguageDataset.__new__(LanguageDataset)
        dataset.tokenizer = DummyTokenizer()
        dataset.context_length = 4
        dataset.device = torch.device('cuda')
        dataset.train_data = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        
        x, y = dataset.get_batch("train", batch_size=1)
        self.assertEqual(x.device.type, 'cuda')
        self.assertEqual(y.device.type, 'cuda')

if __name__ == "__main__":
    unittest.main()
