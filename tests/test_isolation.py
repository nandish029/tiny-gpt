import unittest
import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_gpt1 import main

class TestTestIsolation(unittest.TestCase):
    @patch('scripts.train_gpt1.get_tokenizer')
    def test_smoke_test_does_not_overwrite_canonical_artifacts(self, mock_get_tokenizer):
        """Verifies that invoking a smoke test does not modify canonical results or checkpoints."""
        
        mock_tokenizer = mock_get_tokenizer.return_value
        mock_tokenizer.vocab_size = 89
        
        canonical_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'gpt1', 'gpt1_baseline.pt')
        canonical_results = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'gpt1', 'results.md')
        
        # Get modification times before running smoke test
        ckpt_mtime_before = os.path.getmtime(canonical_checkpoint) if os.path.exists(canonical_checkpoint) else None
        res_mtime_before = os.path.getmtime(canonical_results) if os.path.exists(canonical_results) else None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = os.path.join(temp_dir, "checkpoints")
            results_path = os.path.join(temp_dir, "results.md")
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'gpt1.yaml')
            
            with patch('sys.argv', ['train_gpt1.py', '--out_dir', ckpt_dir, '--config', config_path, '--steps', '1', '--results_path', results_path]):
                with patch('scripts.train_gpt1.LanguageDataset') as mock_dataset:
                    import torch
                    mock_dataset.return_value.get_batch.return_value = (torch.zeros(16, 64, dtype=torch.long), torch.zeros(16, 64, dtype=torch.long))
                    main()
            
            # Verify the temporary output exists
            self.assertTrue(os.path.exists(results_path))
            
        # Get modification times after running smoke test
        ckpt_mtime_after = os.path.getmtime(canonical_checkpoint) if os.path.exists(canonical_checkpoint) else None
        res_mtime_after = os.path.getmtime(canonical_results) if os.path.exists(canonical_results) else None
        
        # Assert they are unchanged
        self.assertEqual(ckpt_mtime_before, ckpt_mtime_after, "Canonical checkpoint was modified!")
        self.assertEqual(res_mtime_before, res_mtime_after, "Canonical results.md was modified!")

if __name__ == '__main__':
    unittest.main()
