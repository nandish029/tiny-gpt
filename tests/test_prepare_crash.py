import unittest
import os
import json
import tempfile
import shutil
from src.data.prepare import prepare_data

class TestPrepareData(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_prepare_data_files_exist(self):
        # Run preparation for a small number of samples
        prepare_data(num_samples=100, seed=42, data_dir=self.temp_dir)
        
        # Verify both output files exist
        train_path = os.path.join(self.temp_dir, "train.jsonl")
        val_path = os.path.join(self.temp_dir, "val.jsonl")
        
        self.assertTrue(os.path.exists(train_path), "train.jsonl should exist")
        self.assertTrue(os.path.exists(val_path), "val.jsonl should exist")
        
        # Verify they contain the expected split
        with open(train_path, "r", encoding="utf-8") as f:
            train_lines = f.readlines()
            
        with open(val_path, "r", encoding="utf-8") as f:
            val_lines = f.readlines()
            
        self.assertEqual(len(train_lines), 90, "Expected 90 train samples")
        self.assertEqual(len(val_lines), 10, "Expected 10 val samples")

if __name__ == "__main__":
    unittest.main()
