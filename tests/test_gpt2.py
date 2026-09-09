import unittest
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.gpt import GPT
from src.utils.config import load_config
from src.tokenizer.character import CharacterTokenizer

class TestGPT2Architecture(unittest.TestCase):
    def setUp(self):
        self.gpt1_config = load_config("configs/gpt1.yaml")
        self.gpt2_config = load_config("configs/gpt2.yaml")

    def test_architecture_equality(self):
        """Verify GPT-2 architecture is exactly identical to GPT-1."""
        m1_cfg = self.gpt1_config['model']
        m2_cfg = self.gpt2_config['model']

        self.assertEqual(m1_cfg['vocab_size'], m2_cfg['vocab_size'])
        self.assertEqual(m1_cfg['max_context_length'], m2_cfg['max_context_length'])
        self.assertEqual(m1_cfg['embedding_dim'], m2_cfg['embedding_dim'])
        self.assertEqual(m1_cfg['num_heads'], m2_cfg['num_heads'])
        self.assertEqual(m1_cfg['feed_forward_dim'], m2_cfg['feed_forward_dim'])
        self.assertEqual(m1_cfg['num_layers'], m2_cfg['num_layers'])

        gpt1 = GPT(**m1_cfg)
        gpt2 = GPT(**m2_cfg)

        gpt1_params = sum(p.numel() for p in gpt1.parameters() if p.requires_grad)
        gpt2_params = sum(p.numel() for p in gpt2.parameters() if p.requires_grad)

        self.assertEqual(gpt1_params, gpt2_params)
        self.assertEqual(gpt1_params, 33305)

    def test_training_configuration(self):
        """Verify GPT-2 differs from GPT-1 only in training steps."""
        t1_cfg = self.gpt1_config['training']
        t2_cfg = self.gpt2_config['training']

        self.assertEqual(t1_cfg['batch_size'], t2_cfg['batch_size'])
        self.assertEqual(t1_cfg['learning_rate'], t2_cfg['learning_rate'])
        self.assertEqual(t1_cfg['seed'], t2_cfg['seed'])
        
        # This is the deliberate change!
        self.assertEqual(t1_cfg['train_steps'], 1000)
        self.assertEqual(t2_cfg['train_steps'], 5000)

    def test_gpt1_regression(self):
        """Verify GPT-1 frozen checkpoint is unmodified."""
        ckpt_path = "checkpoints/gpt1/gpt1_baseline.pt"
        if not os.path.exists(ckpt_path):
            self.skipTest("Frozen GPT-1 checkpoint not found. Run validation to recreate.")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint['config']['vocab_size'], 89)
        
        gpt1 = GPT(**checkpoint['config'])
        gpt1.load_state_dict(checkpoint['model_state_dict'])
        param_count = sum(p.numel() for p in gpt1.parameters() if p.requires_grad)
        self.assertEqual(param_count, 33305)

if __name__ == "__main__":
    unittest.main()
