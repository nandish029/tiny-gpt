import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tokenizer.character import CharacterTokenizer

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.tokenizer.build_vocab(["hello world", "abc", "ABC"])
        self.test_save_path = "data/test_processed_tokenizer.json"

    def tearDown(self):
        if os.path.exists(self.test_save_path):
            os.remove(self.test_save_path)
            
    def test_encoding_returns_ids(self):
        ids = self.tokenizer.encode("hello")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))
        
    def test_decoding_returns_strings(self):
        ids = self.tokenizer.encode("hello")
        text = self.tokenizer.decode(ids)
        self.assertIsInstance(text, str)
        
    def test_round_trip(self):
        original_text = "hello world ABC"
        ids = self.tokenizer.encode(original_text)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(decoded, original_text)
        
    def test_empty_text(self):
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")
        
    def test_repeated_encoding_identical(self):
        ids1 = self.tokenizer.encode("hello")
        ids2 = self.tokenizer.encode("hello")
        self.assertEqual(ids1, ids2)
        
    def test_vocab_size(self):
        expected_chars = set("hello worldabcABC")
        self.assertEqual(self.tokenizer.vocab_size, len(expected_chars))
        self.assertEqual(len(self.tokenizer.char_to_id), len(expected_chars))
        
    def test_save_and_load(self):
        self.tokenizer.save(self.test_save_path)
        
        new_tokenizer = CharacterTokenizer()
        new_tokenizer.load(self.test_save_path)
        
        self.assertEqual(self.tokenizer.vocab_size, new_tokenizer.vocab_size)
        self.assertEqual(self.tokenizer.char_to_id, new_tokenizer.char_to_id)
        
        # Verify functionality is preserved
        test_text = "hello"
        self.assertEqual(self.tokenizer.encode(test_text), new_tokenizer.encode(test_text))
        
    def test_unsupported_characters(self):
        with self.assertRaises(ValueError):
            self.tokenizer.encode("x") # 'x' is not in our small vocab
            
        with self.assertRaises(ValueError):
            self.tokenizer.decode([999])

if __name__ == "__main__":
    unittest.main()
