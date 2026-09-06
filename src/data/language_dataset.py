import torch
import os
from src.data.dataset import TinyStoriesDataset
from src.utils.device import get_device

class LanguageDataset:
    def __init__(self, tokenizer, data_dir="data/processed", context_length=64):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.device = get_device()
        
        self.train_data = self._load_and_encode("train", data_dir)
        self.valid_data = self._load_and_encode("valid", data_dir)
        
    def _load_and_encode(self, split, data_dir):
        """Loads dataset split and encodes it into a 1D tensor of IDs."""
        ds = TinyStoriesDataset(split=split, data_dir=data_dir)
        # Join all stories. Using space is simple and effective.
        all_text = " ".join(ds.stories)
        
        # Filter unsupported characters to prevent tokenizer errors.
        # This keeps the pipeline robust even if validation data has unseen characters.
        valid_chars = set(self.tokenizer.char_to_id.keys())
        filtered_text = "".join(c for c in all_text if c in valid_chars)
        
        ids = self.tokenizer.encode(filtered_text)
        return torch.tensor(ids, dtype=torch.long)
        
    def get_batch(self, split="train", batch_size=4):
        """
        Samples a batch of context_length from the specified split.
        X is the input sequence, Y is the target sequence shifted by 1.
        """
        if split == "train":
            data = self.train_data
        elif split == "valid":
            data = self.valid_data
        else:
            raise ValueError(f"Invalid split: {split}")
            
        max_idx = len(data) - self.context_length - 1
        if max_idx <= 0:
            raise ValueError("Dataset is too small for the given context length.")
            
        # Sample starting indices randomly
        ix = torch.randint(0, max_idx, (batch_size,))
        
        # Extract sequences
        x = torch.stack([data[i : i + self.context_length] for i in ix])
        y = torch.stack([data[i + 1 : i + self.context_length + 1] for i in ix])
        
        return x.to(self.device), y.to(self.device)
