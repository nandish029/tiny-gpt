import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import io
import torch

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.chat import parse_args, validate_args, validate_architecture, main

class TestCLI(unittest.TestCase):
    
    def test_help_works(self):
        """1. --help works."""
        with patch('sys.argv', ['chat.py', '--help']):
            with self.assertRaises(SystemExit) as cm:
                parse_args()
            self.assertEqual(cm.exception.code, 0)
            
    def test_argument_parser_valid(self):
        """2. argument parser accepts valid arguments."""
        with patch('sys.argv', ['chat.py', '--checkpoint', 'ckpt.pt', '--tokenizer', 'tok.json', '--config', 'conf.yaml', '--device', 'cpu', '--prompt', 'hello']):
            args = parse_args()
            self.assertEqual(args.checkpoint, 'ckpt.pt')
            self.assertEqual(args.device, 'cpu')
            
    def test_invalid_device_rejected(self):
        """3. invalid device rejected."""
        with patch('sys.argv', ['chat.py', '--checkpoint', 'ckpt.pt', '--tokenizer', 'tok.json', '--config', 'conf.yaml', '--device', 'invalid']):
            with self.assertRaises(SystemExit) as cm: # argparse exits on invalid choices
                parse_args()
            self.assertEqual(cm.exception.code, 2)
            
    def test_missing_files(self):
        """4, 5, 6. missing checkpoint/tokenizer/config gives useful failure."""
        with patch('sys.argv', ['chat.py', '--checkpoint', 'ckpt.pt', '--tokenizer', 'tok.json', '--config', 'conf.yaml']):
            args = parse_args()
            with self.assertRaisesRegex(FileNotFoundError, "Checkpoint file not found"):
                validate_args(args)
                
        # To test tokenizer failure, create dummy ckpt file
        with open("dummy_ckpt.pt", "w") as f: f.write("")
        with patch('sys.argv', ['chat.py', '--checkpoint', 'dummy_ckpt.pt', '--tokenizer', 'tok.json', '--config', 'conf.yaml']):
            args = parse_args()
            with self.assertRaisesRegex(FileNotFoundError, "Tokenizer file not found"):
                validate_args(args)
                
        # To test config failure, create dummy tok file
        with open("dummy_tok.json", "w") as f: f.write("")
        with patch('sys.argv', ['chat.py', '--checkpoint', 'dummy_ckpt.pt', '--tokenizer', 'dummy_tok.json', '--config', 'conf.yaml']):
            args = parse_args()
            with self.assertRaisesRegex(FileNotFoundError, "Config file not found"):
                validate_args(args)
                
        os.remove("dummy_ckpt.pt")
        os.remove("dummy_tok.json")

    def test_invalid_generation_params(self):
        """7, 8. invalid temperature/max-new-tokens rejected."""
        # Need dummy files so file check passes, or just construct Namespace manually
        class DummyArgs:
            def __init__(self, t, m):
                self.temperature = t
                self.max_new_tokens = m
                self.checkpoint = "nonexistent.pt"
                self.tokenizer = "nonexistent.json"
                self.config = "nonexistent.yaml"
        
        with self.assertRaisesRegex(ValueError, "Temperature must be a finite positive"):
            validate_args(DummyArgs(-1.0, 10))
            
        with self.assertRaisesRegex(ValueError, "Max new tokens must be a positive"):
            validate_args(DummyArgs(1.0, 0))

    @patch('scripts.chat.resolve_device')
    @patch('scripts.chat.load_config')
    @patch('scripts.chat.torch.load')
    @patch('scripts.chat.GPT')
    @patch('scripts.chat.get_tokenizer')
    @patch('scripts.chat.generate')
    def test_mocked_one_shot(self, mock_generate, mock_tokenizer_class, mock_gpt_class, mock_torch_load, mock_load_config, mock_resolve_device):
        """9, 11, 12, 13. Test the main logic via mocks."""
        # Setup mocks
        mock_device = torch.device('cpu')
        mock_resolve_device.return_value = mock_device
        
        mock_load_config.return_value = {'model': {'vocab_size': 100}}
        mock_torch_load.return_value = {'config': {'vocab_size': 100}, 'model_state_dict': {}}
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_gpt_class.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_generate.return_value = "generated text"
        
        # We patch os.path.exists so validate_args doesn't fail on missing files
        with patch('scripts.chat.os.path.exists', return_value=True):
            with patch('sys.argv', ['chat.py', '--checkpoint', 'dummy.pt', '--tokenizer', 'dummy.json', '--config', 'dummy.yaml', '--prompt', 'hello']):
                main()
                
        # 11. model is put into eval mode
        mock_model.eval.assert_called_once()
        
        # 13. selected device is actually used
        mock_gpt_class.assert_called_with(vocab_size=100)
        mock_model.to.assert_called_with(mock_device)
        
        # 12. generation runs
        mock_generate.assert_called_once()
        args, kwargs = mock_generate.call_args
        self.assertEqual(args[2], 'hello') # the prompt

    @patch('scripts.chat.resolve_device')
    @patch('scripts.chat.load_config')
    @patch('scripts.chat.torch.load')
    @patch('scripts.chat.GPT')
    @patch('scripts.chat.get_tokenizer')
    @patch('scripts.chat.generate')
    def test_mocked_interactive(self, mock_generate, mock_tokenizer_class, mock_gpt_class, mock_torch_load, mock_load_config, mock_resolve_device):
        """10. interactive mode can be tested without requiring manual input."""
        mock_resolve_device.return_value = torch.device('cpu')
        mock_load_config.return_value = {'model': {}}
        mock_torch_load.return_value = {'config': {}, 'model_state_dict': {}}
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_gpt_class.return_value = mock_model
        
        # Inject "hello" then "quit"
        with patch('builtins.input', side_effect=['hello', 'quit']):
            with patch('scripts.chat.os.path.exists', return_value=True):
                with patch('sys.argv', ['chat.py', '--checkpoint', 'dummy.pt', '--tokenizer', 'dummy.json', '--config', 'dummy.yaml']):
                    main()
        
        # Generate should be called exactly once for "hello"
        mock_generate.assert_called_once()
        
    def test_real_gpt1_execution(self):
        """9, 14, 15. Tests real execution if GPT-1 checkpoint exists."""
        ckpt_path = "checkpoints/gpt1/gpt1_baseline.pt"
        tok_path = "data/processed/tokenizer.json"
        cfg_path = "configs/gpt1.yaml"
        
        if not (os.path.exists(ckpt_path) and os.path.exists(tok_path) and os.path.exists(cfg_path)):
            self.skipTest("GPT-1 checkpoint or data not found. Skipping real integration test.")
            
        import hashlib
        # compute hash of checkpoint to verify it doesn't change
        with open(ckpt_path, "rb") as f:
            original_hash = hashlib.md5(f.read()).hexdigest()
            
        with patch('sys.argv', ['chat.py', '--checkpoint', ckpt_path, '--tokenizer', tok_path, '--config', cfg_path, '--prompt', 'The', '--device', 'cpu', '--seed', '123']):
            # redirect stdout
            captured_out1 = io.StringIO()
            with patch('sys.stdout', new=captured_out1):
                main()
                
        # 14. same seed produces repeatable output
        with patch('sys.argv', ['chat.py', '--checkpoint', ckpt_path, '--tokenizer', tok_path, '--config', cfg_path, '--prompt', 'The', '--device', 'cpu', '--seed', '123']):
            captured_out2 = io.StringIO()
            with patch('sys.stdout', new=captured_out2):
                main()
                
        self.assertEqual(captured_out1.getvalue(), captured_out2.getvalue())
        
        # 15. GPT-1 checkpoint remains unchanged
        with open(ckpt_path, "rb") as f:
            new_hash = hashlib.md5(f.read()).hexdigest()
        self.assertEqual(original_hash, new_hash)
    @patch('scripts.chat.resolve_device')
    @patch('scripts.chat.load_config')
    @patch('scripts.chat.torch.load')
    @patch('scripts.chat.GPT')
    @patch('scripts.chat.get_tokenizer')
    @patch('scripts.chat.generate')
    def test_interactive_immediate_eof_exits_with_error(self, mock_generate, mock_tokenizer_class, mock_gpt_class, mock_torch_load, mock_load_config, mock_resolve_device):
        """Simulates docker run without -i: input() raises EOFError immediately."""
        mock_resolve_device.return_value = torch.device('cpu')
        mock_load_config.return_value = {'model': {}}
        mock_torch_load.return_value = {'config': {}, 'model_state_dict': {}}

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_gpt_class.return_value = mock_model

        # input() raises EOFError on first call (no stdin attached)
        with patch('builtins.input', side_effect=EOFError):
            with patch('scripts.chat.os.path.exists', return_value=True):
                with patch('sys.argv', ['chat.py', '--checkpoint', 'dummy.pt', '--tokenizer', 'dummy.json', '--config', 'dummy.yaml']):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 1)

        # Generate should NOT have been called
        mock_generate.assert_not_called()

    @patch('scripts.chat.resolve_device')
    @patch('scripts.chat.load_config')
    @patch('scripts.chat.torch.load')
    @patch('scripts.chat.GPT')
    @patch('scripts.chat.get_tokenizer')
    @patch('scripts.chat.generate')
    def test_interactive_mid_session_eof_exits_gracefully(self, mock_generate, mock_tokenizer_class, mock_gpt_class, mock_torch_load, mock_load_config, mock_resolve_device):
        """Mid-session Ctrl+D (EOF after some successful input) exits gracefully."""
        mock_resolve_device.return_value = torch.device('cpu')
        mock_load_config.return_value = {'model': {}}
        mock_torch_load.return_value = {'config': {}, 'model_state_dict': {}}

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_gpt_class.return_value = mock_model

        # First call succeeds, second raises EOF
        with patch('builtins.input', side_effect=['hello', EOFError]):
            with patch('scripts.chat.os.path.exists', return_value=True):
                with patch('sys.argv', ['chat.py', '--checkpoint', 'dummy.pt', '--tokenizer', 'dummy.json', '--config', 'dummy.yaml']):
                    main()  # Should NOT raise SystemExit

        mock_generate.assert_called_once()

    @patch('scripts.chat.resolve_device')
    @patch('scripts.chat.load_config')
    @patch('scripts.chat.torch.load')
    @patch('scripts.chat.GPT')
    @patch('scripts.chat.get_tokenizer')
    @patch('scripts.chat.generate')
    def test_interactive_multiple_prompts_then_exit(self, mock_generate, mock_tokenizer_class, mock_gpt_class, mock_torch_load, mock_load_config, mock_resolve_device):
        """Interactive mode accepts multiple prompts then exits on 'exit'."""
        mock_resolve_device.return_value = torch.device('cpu')
        mock_load_config.return_value = {'model': {}}
        mock_torch_load.return_value = {'config': {}, 'model_state_dict': {}}

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_gpt_class.return_value = mock_model

        with patch('builtins.input', side_effect=['first', 'second', '', 'exit']):
            with patch('scripts.chat.os.path.exists', return_value=True):
                with patch('sys.argv', ['chat.py', '--checkpoint', 'dummy.pt', '--tokenizer', 'dummy.json', '--config', 'dummy.yaml']):
                    main()

        # '' (empty) should be skipped, so generate called for 'first' and 'second'
        self.assertEqual(mock_generate.call_count, 2)

if __name__ == "__main__":
    unittest.main()
