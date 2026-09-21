import unittest
import torch
import os
import shutil
import json
from src.tokenizer.factory import get_tokenizer
from src.data.language_dataset import LanguageDataset
from src.model.gpt import GPT
from scripts.train_subword_tokenizer import main as train_tokenizer_main

class TestGPT3Integration(unittest.TestCase):
    def setUp(self):
        # Create a small fake dataset for testing
        self.data_dir = "data/test_gpt3_processed"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.train_path = os.path.join(self.data_dir, "train.jsonl")
        self.val_path = os.path.join(self.data_dir, "val.jsonl")
        
        with open(self.train_path, "w") as f:
            for i in range(20):
                f.write(json.dumps({"text": f"This is training text {i}. It has some repeated words to form subwords."}) + "\n")
                
        with open(self.val_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"text": f"This is validation text {i}. Validate validate!"}) + "\n")
                
        # Train tokenizer on train.jsonl
        self.tokenizer = get_tokenizer("subword")
        # Generator for train texts
        def text_gen():
            with open(self.train_path, "r") as f:
                for line in f:
                    yield json.loads(line)["text"]
                    
        self.tokenizer.build_vocab(text_gen(), vocab_size=64)
        
    def tearDown(self):
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
            
    def test_vocab_size_consistency(self):
        self.assertEqual(self.tokenizer.vocab_size, 64)
        
    def test_dataset_integration(self):
        dataset = LanguageDataset(self.tokenizer, data_dir=self.data_dir, context_length=16)
        x, y = dataset.get_batch("train", batch_size=4)
        
        self.assertEqual(x.shape, (4, 16))
        self.assertEqual(y.shape, (4, 16))
        
        # Verify ids are within vocab bounds
        self.assertTrue(torch.all(x >= 0))
        self.assertTrue(torch.all(x < self.tokenizer.vocab_size))
        
        self.assertTrue(torch.all(y >= 0))
        self.assertTrue(torch.all(y < self.tokenizer.vocab_size))
        
    def test_model_integration(self):
        model = GPT(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=32,
            max_context_length=16,
            num_heads=2,
            feed_forward_dim=64,
            num_layers=2
        )
        
        # Verify internal Transformer architecture is unchanged (except embeddings/lm_head due to vocab)
        # We can check specific layers exist
        self.assertTrue(hasattr(model, 'transformer_stack'))
        self.assertEqual(len(model.transformer_stack.blocks), 2)
        
        dataset = LanguageDataset(self.tokenizer, data_dir=self.data_dir, context_length=16)
        x, y = dataset.get_batch("train", batch_size=4)
        
        logits = model(x)
        self.assertEqual(logits.shape, (4, 16, self.tokenizer.vocab_size))
        self.assertTrue(torch.all(torch.isfinite(logits)))

if __name__ == '__main__':
    unittest.main()
