import json
import os

class CharacterTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        """Builds a deterministic vocabulary from an iterable of strings."""
        chars = set()
        for text in texts:
            chars.update(list(text))
            
        # Deterministic sorting
        sorted_chars = sorted(list(chars))
        
        self.char_to_id = {ch: i for i, ch in enumerate(sorted_chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(sorted_chars)}
        self.vocab_size = len(sorted_chars)
        
    def encode(self, text):
        """Converts text to integer IDs."""
        ids = []
        for ch in text:
            if ch not in self.char_to_id:
                raise ValueError(f"Character '{ch}' not in vocabulary.")
            ids.append(self.char_to_id[ch])
        return ids
        
    def decode(self, ids):
        """Converts integer IDs back to text."""
        chars = []
        for idx in ids:
            if idx not in self.id_to_char:
                raise ValueError(f"Token ID '{idx}' not in vocabulary.")
            chars.append(self.id_to_char[idx])
        return "".join(chars)
        
    def save(self, filepath):
        """Saves the vocabulary configuration to a file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "char_to_id": self.char_to_id,
                "vocab_size": self.vocab_size
            }, f, ensure_ascii=False, indent=2)
            
    def load(self, filepath):
        """Loads the vocabulary configuration from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.char_to_id = data["char_to_id"]
        # JSON keys are always strings, so we need to convert back to int for id_to_char
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self.vocab_size = data["vocab_size"]
