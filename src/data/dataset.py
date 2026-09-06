import json
import os

class TinyStoriesDataset:
    def __init__(self, split="train", data_dir="data/processed"):
        self.data_path = os.path.join(data_dir, f"{split}.jsonl")
        self.stories = []
        self._load_data()
        
    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}. Run src/data/prepare.py first.")
            
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.stories.append(item["text"])
                
    def __len__(self):
        return len(self.stories)
        
    def __getitem__(self, idx):
        return self.stories[idx]
