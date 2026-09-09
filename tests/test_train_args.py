import unittest
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestTrainCLIArgs(unittest.TestCase):
    @patch('scripts.train_gpt1.load_config')
    @patch('scripts.train_gpt1.CharacterTokenizer')
    @patch('scripts.train_gpt1.LanguageDataset')
    @patch('scripts.train_gpt1.GPT')
    @patch('scripts.train_gpt1.create_optimizer')
    @patch('scripts.train_gpt1.resolve_device')
    @patch('scripts.train_gpt1.torch')
    @patch('scripts.train_gpt1.os.makedirs')
    @patch('scripts.train_gpt1.train_step')
    @patch('builtins.open')
    def test_device_argument_parsing(self, mock_open, mock_train_step, mock_makedirs, mock_torch, mock_resolve_device, mock_gpt, mock_dataset, mock_tokenizer, mock_load_config):
        """Verifies that --device is correctly parsed and passed to resolve_device."""
        
        # We need to mock sys.exit to prevent the script from exiting on argparse errors
        # if the test fails or argument is missing.
        from scripts.train_gpt1 import main
        
        # Mock implementations
        mock_load_config.return_value = {
            'model': {'vocab_size': 89, 'embedding_dim': 32, 'max_context_length': 64, 'num_heads': 4, 'feed_forward_dim': 128, 'num_layers': 2},
            'training': {'batch_size': 16, 'learning_rate': 3e-4, 'train_steps': 1, 'val_interval': 1, 'seed': 42}
        }
        mock_tokenizer_instance = mock_tokenizer.return_value
        mock_tokenizer_instance.vocab_size = 89
        
        mock_dataset_instance = mock_dataset.return_value
        mock_dataset_instance.get_batch.return_value = (mock_torch.Tensor(), mock_torch.Tensor())
        
        mock_gpt_instance = mock_gpt.return_value
        mock_gpt_instance.parameters.return_value = [mock_torch.Tensor()]
        
        mock_train_step.return_value = mock_torch.Tensor([1.0])
        mock_torch.isfinite.return_value = True
        
        # Test default (auto)
        with patch('sys.argv', ['train_gpt1.py']):
            main()
            mock_resolve_device.assert_called_with('auto')
            
        # Test explicit cpu
        with patch('sys.argv', ['train_gpt1.py', '--device', 'cpu']):
            main()
            mock_resolve_device.assert_called_with('cpu')
            
        # Test explicit cuda
        with patch('sys.argv', ['train_gpt1.py', '--device', 'cuda']):
            main()
            mock_resolve_device.assert_called_with('cuda')
            
    def test_invalid_device_rejected(self):
        """Verify argparse rejects invalid devices."""
        from scripts.train_gpt1 import main
        with patch('sys.argv', ['train_gpt1.py', '--device', 'invalid']):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 2)

if __name__ == '__main__':
    unittest.main()
