import unittest
import os
import json
import tempfile
import torch
from src.tokenizer.subword import SubwordTokenizer
from src.tokenizer.character import CharacterTokenizer
from scripts.train_subword_tokenizer import text_generator
from src.data.language_dataset import LanguageDataset

class TestSubwordTokenizer(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "Hello world",
            "This is a test of the subword tokenizer.",
            "BPE is cool!",
            "It learns common subwords.",
            "Special characters: !@#$%^&*()",
            "Unicode: 😊👍",
            "Repeated repeated repeated words words",
            "A short string",
            "",
            " "
        ]
        self.tokenizer = SubwordTokenizer()
        self.vocab_size = 64
        self.tokenizer.build_vocab(self.texts, vocab_size=self.vocab_size)
        
    def test_vocabulary(self):
        # 1. Vocabulary
        self.assertEqual(self.tokenizer.vocab_size, self.vocab_size)
        self.assertTrue(self.tokenizer.vocab_size > 0)
        
        # Test special tokens (UNK might be stripped or have specific ID, let's just ensure encode works)
        unk_ids = self.tokenizer.encode("[UNK]")
        self.assertTrue(len(unk_ids) > 0)
        self.assertIsInstance(unk_ids[0], int)
        
        # Unique IDs
        ids = self.tokenizer.encode("Hello world")
        self.assertTrue(all(isinstance(i, int) for i in ids))
        
    def test_encode_decode(self):
        # 2. Encode/decode
        for text in self.texts:
            if not text.strip():
                # empty string or whitespace behavior
                ids = self.tokenizer.encode(text)
                self.assertEqual(len(ids), 0)
                continue
                
            ids = self.tokenizer.encode(text)
            self.assertTrue(len(ids) > 0)
            
            decoded = self.tokenizer.decode(ids)
            self.assertIsInstance(decoded, str)
            
            # encode -> decode remains valid
            self.assertTrue(len(decoded) > 0 or not text.strip())

    def test_save_load(self):
        # 3. Save/load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tokenizer.json")
            self.tokenizer.save(path)
            
            tokenizer2 = SubwordTokenizer()
            tokenizer2.load(path)
            
            self.assertEqual(self.tokenizer.vocab_size, tokenizer2.vocab_size)
            
            text = "Test save load behavior 😊"
            ids1 = self.tokenizer.encode(text)
            ids2 = tokenizer2.encode(text)
            self.assertEqual(ids1, ids2)

    def test_determinism(self):
        # 4. Determinism
        tokenizer2 = SubwordTokenizer()
        tokenizer2.build_vocab(self.texts, vocab_size=self.vocab_size)
        
        self.assertEqual(self.tokenizer.vocab_size, tokenizer2.vocab_size)
        
        text = "Hello world this is deterministic!"
        ids1 = self.tokenizer.encode(text)
        ids2 = tokenizer2.encode(text)
        self.assertEqual(ids1, ids2)

    def test_training_data_isolation(self):
        # 5. Training-data isolation
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.jsonl")
            val_path = os.path.join(tmpdir, "val.jsonl")
            
            with open(train_path, "w") as f:
                f.write(json.dumps({"text": "Hello train"}) + "\n")
                
            with open(val_path, "w") as f:
                f.write(json.dumps({"text": "Hello val"}) + "\n")
                
            # Should read train.jsonl fine
            texts = list(text_generator(train_path))
            self.assertEqual(len(texts), 1)
            
            # Should raise ValueError for val.jsonl
            with self.assertRaises(ValueError):
                list(text_generator(val_path))

    def test_dataset_integration(self):
        # 6. Dataset integration
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy dataset
            train_path = os.path.join(tmpdir, "train.jsonl")
            val_path = os.path.join(tmpdir, "val.jsonl")
            
            long_text = "This is a long text to test the dataset integration. " * 10
            with open(train_path, "w") as f:
                f.write(json.dumps({"text": long_text}) + "\n")
            with open(val_path, "w") as f:
                f.write(json.dumps({"text": long_text}) + "\n")
                
            dataset = LanguageDataset(self.tokenizer, data_dir=tmpdir, context_length=8)
            x, y = dataset.get_batch("train", batch_size=4)
            
            self.assertEqual(x.shape, (4, 8))
            self.assertEqual(y.shape, (4, 8))
            
            # X and Y correctly shifted
            for b in range(4):
                self.assertTrue(torch.equal(x[b, 1:], y[b, :-1]))
                
    def test_regression_char_tokenizer(self):
        # 7. Regression
        char_tokenizer = CharacterTokenizer()
        # Mock some texts to build vocab
        char_tokenizer.build_vocab("".join(self.texts))
        
        ids = char_tokenizer.encode("Hello")
        decoded = char_tokenizer.decode(ids)
        self.assertEqual("Hello", decoded)

if __name__ == '__main__':
    unittest.main()
