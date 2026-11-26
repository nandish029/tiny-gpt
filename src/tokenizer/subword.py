import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class SubwordTokenizer:
    def __init__(self, byte_level: bool = False):
        self.byte_level = byte_level
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        if byte_level:
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = ByteLevelDecoder()
        else:
            self.tokenizer.pre_tokenizer = Whitespace()
        self.vocab_size = 0
        
    def build_vocab(self, texts, vocab_size=512):
        """Builds a deterministic vocabulary from an iterable of strings."""
        initial_alphabet = ByteLevel.alphabet() if self.byte_level else []
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "<|endoftext|>"] if self.byte_level else ["[UNK]", "[PAD]"],
            initial_alphabet=initial_alphabet,
        )
        self.tokenizer.train_from_iterator(texts, trainer)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
    def encode(self, text):
        """Converts text to integer IDs."""
        return self.tokenizer.encode(text).ids
        
    def decode(self, ids):
        """Converts integer IDs back to text."""
        return self.tokenizer.decode(ids)
        
    def save(self, filepath):
        """Saves the vocabulary configuration to a file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.tokenizer.save(filepath)
        
    def load(self, filepath):
        """Loads the vocabulary configuration from a file."""
        self.tokenizer = Tokenizer.from_file(filepath)
        self.vocab_size = self.tokenizer.get_vocab_size()
