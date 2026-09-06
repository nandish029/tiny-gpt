import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tokenizer.character import CharacterTokenizer
from src.data.language_dataset import LanguageDataset
from src.utils.device import get_device

class TestLanguageDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CharacterTokenizer()
        cls.tokenizer.load("data/processed/tokenizer.json")
        cls.dataset = LanguageDataset(cls.tokenizer, data_dir="data/processed", context_length=16)

    def test_dimensions(self):
        """1. X and Y have the expected dimensions. 5. Batch size correct. 6. Context length correct."""
        x, y = self.dataset.get_batch("train", batch_size=4)
        self.assertEqual(x.shape, (4, 16))
        self.assertEqual(y.shape, (4, 16))
        
    def test_data_types_and_values(self):
        """2. X and Y contain integer token IDs. 3. Both contain valid vocabulary IDs."""
        x, y = self.dataset.get_batch("valid", batch_size=2)
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)
        
        self.assertTrue((x >= 0).all() and (x < self.tokenizer.vocab_size).all())
        self.assertTrue((y >= 0).all() and (y < self.tokenizer.vocab_size).all())
        
    def test_shifted_target_controlled(self):
        """4. Y is shifted exactly one position relative to X. (Using controlled sequence)"""
        # Create a mock dataset to strictly verify the shift mapping
        dataset = LanguageDataset.__new__(LanguageDataset)
        dataset.context_length = 4
        dataset.device = torch.device('cpu') 
        # A B C D E F -> 0 1 2 3 4 5
        dataset.train_data = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        
        # Override random choice to start at index 0
        original_randint = torch.randint
        torch.randint = lambda low, high, size: torch.zeros(size, dtype=torch.long)
        
        x, y = dataset.get_batch("train", batch_size=1)
        
        # Restore randint
        torch.randint = original_randint
        
        # Input: A B C D (0, 1, 2, 3)
        self.assertEqual(x[0].tolist(), [0, 1, 2, 3])
        # Target: B C D E (1, 2, 3, 4)
        self.assertEqual(y[0].tolist(), [1, 2, 3, 4])
        
    def test_data_separation(self):
        """7. Training and validation data remain separate."""
        self.assertNotEqual(id(self.dataset.train_data), id(self.dataset.valid_data))
        
    def test_device_placement(self):
        """8 & 9. Batches can be created on CPU (and CUDA if available)."""
        device = get_device()
        x, y = self.dataset.get_batch("train", batch_size=2)
        self.assertEqual(x.device.type, device.type)
        self.assertEqual(y.device.type, device.type)

if __name__ == "__main__":
    unittest.main()
