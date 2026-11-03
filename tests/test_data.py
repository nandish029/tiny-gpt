import unittest
import os
import sys
import shutil
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset import TinyStoriesDataset

try:
    import datasets
    from src.data.prepare import prepare_data
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class MockTinyStoriesDataset:
    """Deterministic offline mock for the streaming TinyStories dataset."""
    def __init__(self, total=500):
        self.total = total

    def __iter__(self):
        for i in range(self.total):
            yield {
                "text": (
                    f"Once upon a time, story {i} unfolded in a quiet little village. "
                    f"A cheerful little puppy named Toby liked to explore the green garden. "
                    f"Every afternoon he shared sweet apples with all of his kind friends. The end."
                )
            }


@unittest.skipIf(not DATASETS_AVAILABLE, "datasets library not installed")
class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir1 = "data/test_processed_1"
        self.test_dir2 = "data/test_processed_2"
        shutil.rmtree(self.test_dir1, ignore_errors=True)
        shutil.rmtree(self.test_dir2, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir1, ignore_errors=True)
        shutil.rmtree(self.test_dir2, ignore_errors=True)

    @patch("src.data.prepare.load_dataset", side_effect=lambda *args, **kwargs: MockTinyStoriesDataset())
    def test_pipeline(self, mock_load):
        test_dir1 = self.test_dir1
        test_dir2 = self.test_dir2
        
        # 1 & 5. Deterministic preparation
        prepare_data(num_samples=100, seed=42, data_dir=test_dir1)
        prepare_data(num_samples=100, seed=42, data_dir=test_dir2)
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(test_dir1, "train.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(test_dir1, "val.jsonl")))
        
        # 6. Dataset loader successfully loads the processed data
        train_ds = TinyStoriesDataset(split="train", data_dir=test_dir1)
        valid_ds = TinyStoriesDataset(split="val", data_dir=test_dir1)
        
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
