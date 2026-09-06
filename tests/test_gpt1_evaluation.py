import unittest
import torch
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.gpt import GPT
from src.tokenizer.character import CharacterTokenizer
from src.data.language_dataset import LanguageDataset
from src.training.step import validation_step
from src.generation.generate import generate

class TestGPT1Evaluation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'gpt1', 'gpt1_baseline.pt')
        self.tok_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'tokenizer.json')
        
    def test_gpt1_checkpoint_integrity(self):
        # 1. checkpoint exists
        self.assertTrue(os.path.exists(self.ckpt_path))
        
        # 2. checkpoint loads
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
        self.assertIn('config', checkpoint)
        
        # 10. checkpoint step remains 1000
        self.assertEqual(checkpoint['step'], 1000)
        
        # 3. configuration is correct
        config = checkpoint['config']
        self.assertEqual(config['vocab_size'], 89)
        self.assertEqual(config['embedding_dim'], 32)
        self.assertEqual(config['max_context_length'], 64)
        self.assertEqual(config['num_heads'], 4)
        self.assertEqual(config['feed_forward_dim'], 128)
        self.assertEqual(config['num_layers'], 2)
        
        model = GPT(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 4. parameter count is 33,305
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(param_count, 33305)
        
        # 5. all parameters are finite
        for p in model.parameters():
            self.assertTrue(torch.all(torch.isfinite(p)))
            
    def test_gpt1_evaluation_read_only(self):
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        model = GPT(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        tokenizer = CharacterTokenizer()
        tokenizer.load(self.tok_path)
        
        # Pre-evaluation parameter clone
        cloned_params = {n: p.clone() for n, p in model.named_parameters()}
        
        model.eval()
        dataset = LanguageDataset(tokenizer, context_length=checkpoint['config']['max_context_length'])
        
        # Validation
        x, y = dataset.get_batch("valid", batch_size=4)
        val_loss = validation_step(model, x, y)
        
        # 6. validation produces finite loss
        self.assertTrue(torch.isfinite(val_loss))
        
        # 7. perplexity is finite and positive
        perplexity = math.exp(val_loss.item())
        self.assertTrue(math.isfinite(perplexity))
        self.assertGreater(perplexity, 0)
        
        # 8. generation succeeds
        out = generate(model, tokenizer, "Test", max_new_tokens=5, temperature=1.0)
        self.assertTrue(len(out) > len("Test"))
        
        # 9. parameters remain unchanged after evaluation
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(p, cloned_params[n]))

if __name__ == "__main__":
    unittest.main()
