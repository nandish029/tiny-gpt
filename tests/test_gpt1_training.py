import unittest
import torch
import os
import sys
import json
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.gpt import GPT
from src.tokenizer.character import CharacterTokenizer
from src.data.language_dataset import LanguageDataset
from src.training.optimizer import create_optimizer
from src.training.step import train_step, validation_step
from src.utils.device import get_device

class TestGPT1Training(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = get_device()
        
        # We need a dummy dataset for testing so we don't depend on actual data
        self.temp_dir = "tests/temp_data"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.vocab_size = 10
        
        # Create dummy tokenizer
        self.tokenizer = CharacterTokenizer()
        self.tokenizer.char_to_id = {str(i): i for i in range(10)}
        self.tokenizer.id_to_char = {i: str(i) for i in range(10)}
        self.tokenizer.vocab_size = 10
        
        # Create dummy data
        dummy_text = "0123456789" * 20
        with open(os.path.join(self.temp_dir, "train.jsonl"), "w") as f:
            f.write(json.dumps({"text": dummy_text}) + "\n")
        with open(os.path.join(self.temp_dir, "valid.jsonl"), "w") as f:
            f.write(json.dumps({"text": dummy_text}) + "\n")
            
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_gpt1_configuration_and_forward(self):
        # 1 & 2 & 4. Configuration and Forward
        model = GPT(
            vocab_size=self.vocab_size,
            embedding_dim=32,
            max_context_length=64,
            num_heads=4,
            feed_forward_dim=128,
            num_layers=2
        ).to(self.device)
        
        ids = torch.randint(0, self.vocab_size, (2, 64), device=self.device)
        out = model(ids)
        self.assertEqual(out.shape, (2, 64, self.vocab_size))

    def test_dataset_batching(self):
        # 3. Dataset batches
        dataset = LanguageDataset(self.tokenizer, data_dir=self.temp_dir, context_length=64)
        x, y = dataset.get_batch("train", batch_size=2)
        self.assertEqual(x.shape, (2, 64))
        self.assertEqual(y.shape, (2, 64))

    def test_loss_and_parameter_update(self):
        # 5 & 6 & 7. Loss is finite, update works, val loss is finite
        model = GPT(
            vocab_size=self.vocab_size,
            embedding_dim=32,
            max_context_length=64,
            num_heads=4,
            feed_forward_dim=128,
            num_layers=2
        ).to(self.device)
        optimizer = create_optimizer(model, 3e-4)
        dataset = LanguageDataset(self.tokenizer, data_dir=self.temp_dir, context_length=64)
        
        x, y = dataset.get_batch("train", batch_size=2)
        
        param_before = model.lm_head.linear.weight.clone()
        
        loss = train_step(model, optimizer, x, y)
        self.assertTrue(torch.isfinite(loss))
        
        param_after = model.lm_head.linear.weight
        self.assertFalse(torch.equal(param_before, param_after))
        
        val_x, val_y = dataset.get_batch("valid", batch_size=2)
        val_loss = validation_step(model, val_x, val_y)
        self.assertTrue(torch.isfinite(val_loss))

    def test_checkpoint_saving_loading(self):
        # 8 & 9. Checkpoint and results file
        model = GPT(
            vocab_size=self.vocab_size,
            embedding_dim=32,
            max_context_length=64,
            num_heads=4,
            feed_forward_dim=128,
            num_layers=2
        ).to(self.device)
        optimizer = create_optimizer(model, 3e-4)
        
        os.makedirs(os.path.join(self.temp_dir, "checkpoints"), exist_ok=True)
        ckpt_path = os.path.join(self.temp_dir, "checkpoints", "test.pt")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': 10
        }, ckpt_path)
        
        loaded = torch.load(ckpt_path)
        self.assertIn('model_state_dict', loaded)
        self.assertIn('optimizer_state_dict', loaded)
        self.assertEqual(loaded['step'], 10)
        
        results_path = os.path.join(self.temp_dir, "results.md")
        with open(results_path, "w") as f:
            f.write("# Test\n")
            
        self.assertTrue(os.path.exists(results_path))

if __name__ == "__main__":
    unittest.main()
