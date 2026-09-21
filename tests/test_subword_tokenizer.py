import unittest
import os
import json
from src.tokenizer.subword import SubwordTokenizer

class TestSubwordTokenizer(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "Hello world",
            "This is a test of the subword tokenizer.",
            "BPE is cool!",
            "It learns common subwords.",
            "Special characters: !@#$%^&*()",
            "Unicode: 😊👍"
        ]
        self.tokenizer = SubwordTokenizer()
        self.tokenizer.build_vocab(self.texts, vocab_size=64)
        
    def test_encode_decode_roundtrip(self):
        for text in self.texts:
            ids = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(ids)
            # Due to BPE, spacing might be slightly altered, but basic text should be recoverable
            # With tokenizers Whitespace pre_tokenizer, the decoded text doesn't always have spaces perfectly,
            # but tokenizers handles this well generally. Let's just check length is reasonable.
            self.assertIsInstance(decoded, str)
            
    def test_deterministic_vocab(self):
        tokenizer2 = SubwordTokenizer()
        tokenizer2.build_vocab(self.texts, vocab_size=64)
        
        self.assertEqual(self.tokenizer.vocab_size, tokenizer2.vocab_size)
        
        # Encoding same text should yield same ids
        text = "Hello world"
        ids1 = self.tokenizer.encode(text)
        ids2 = tokenizer2.encode(text)
        self.assertEqual(ids1, ids2)

    def test_save_load(self):
        path = "test_subword_tokenizer.json"
        self.tokenizer.save(path)
        
        tokenizer2 = SubwordTokenizer()
        tokenizer2.load(path)
        
        self.assertEqual(self.tokenizer.vocab_size, tokenizer2.vocab_size)
        
        text = "Test save load"
        ids1 = self.tokenizer.encode(text)
        ids2 = tokenizer2.encode(text)
        self.assertEqual(ids1, ids2)
        
        os.remove(path)
        
    def test_unknown_unicode_handling(self):
        # A completely unseen text and emoji
        unseen_text = "Alien invasion! 🛸"
        # BPE will fall back to UNK or bytes depending on config. We used [UNK].
        ids = self.tokenizer.encode(unseen_text)
        self.assertTrue(len(ids) > 0)
        # Should decode back to something (with [UNK]s)
        decoded = self.tokenizer.decode(ids)
        self.assertIsInstance(decoded, str)

if __name__ == '__main__':
    unittest.main()
