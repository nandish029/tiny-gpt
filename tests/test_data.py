import unittest
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset import TinyStoriesDataset

try:
    import datasets
    from src.data.prepare import prepare_data
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

@unittest.skipIf(not DATASETS_AVAILABLE, "datasets library not installed")
class TestDataPipeline(unittest.TestCase):
    def test_pipeline(self):
        test_dir1 = "data/test_processed_1"
        test_dir2 = "data/test_processed_2"
        
        # 1 & 5. Deterministic preparation
        prepare_data(num_samples=100, seed=42, data_dir=test_dir1)
        prepare_data(num_samples=100, seed=42, data_dir=test_dir2)
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(test_dir1, "train.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(test_dir1, "valid.jsonl")))
        
        # 6. Dataset loader successfully loads the processed data
        train_ds = TinyStoriesDataset(split="train", data_dir=test_dir1)
        valid_ds = TinyStoriesDataset(split="valid", data_dir=test_dir1)
        
        # 2. Non-empty sets
        self.assertEqual(len(train_ds), 90)
        self.assertEqual(len(valid_ds), 10)
        
        # 3. No processed document is empty
        for story in train_ds:
            self.assertTrue(len(story.strip()) > 0)
        for story in valid_ds:
            self.assertTrue(len(story.strip()) > 0)
            
        # 4. Train and validation sets do not overlap
        train_set = set(train_ds.stories)
        valid_set = set(valid_ds.stories)
        overlap = train_set.intersection(valid_set)
        self.assertEqual(len(overlap), 0, "Train and validation sets must not overlap")
        
        # Determinism check
        train_ds_2 = TinyStoriesDataset(split="train", data_dir=test_dir2)
        self.assertEqual(train_ds.stories, train_ds_2.stories, "Pipeline is not deterministic with the same seed")
        
if __name__ == "__main__":
    unittest.main()
