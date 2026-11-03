import unittest
import os
import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.gpt import GPT
from src.utils.config import load_config
from src.tokenizer.factory import get_tokenizer
from src.data.language_dataset import LanguageDataset
from src.generation.generate import generate

class TestGPT4E0Reproduction(unittest.TestCase):
    """
    Acceptance test suite for E0: Reproduce the frozen GPT-3 baseline.
    Verifies architecture reconstruction, parameter count, checkpoint loading,
    tokenizer functionality, read-only parameter immutability, and evaluation.
    """

    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.config_path = os.path.join(self.project_root, 'configs', 'gpt3.yaml')
        self.ckpt_path = os.path.join(self.project_root, 'checkpoints', 'gpt3', 'gpt3_baseline.pt')
        self.tok_path = os.path.join(self.project_root, 'data', 'processed', 'gpt3_tokenizer.json')
        self.val_data_path = os.path.join(self.project_root, 'data', 'processed', 'val.jsonl')
        self.device = torch.device('cpu')

    def test_artifact_existence(self):
        """Verify all canonical GPT-3 artifacts required for reproduction exist."""
        self.assertTrue(os.path.isfile(self.config_path), f"Missing config: {self.config_path}")
        self.assertTrue(os.path.isfile(self.ckpt_path), f"Missing checkpoint: {self.ckpt_path}")
        self.assertTrue(os.path.isfile(self.tok_path), f"Missing tokenizer: {self.tok_path}")
        self.assertTrue(os.path.isfile(self.val_data_path), f"Missing validation data: {self.val_data_path}")

    def test_checkpoint_structure_and_metadata(self):
        """Verify checkpoint contains all required keys and valid metadata."""
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.assertIn('model_state_dict', ckpt)
        self.assertIn('optimizer_state_dict', ckpt)
        self.assertIn('config', ckpt)
        self.assertEqual(ckpt.get('step'), 5000)

        ckpt_config = ckpt['config']
        self.assertEqual(ckpt_config.get('tokenizer_type'), 'subword')
        self.assertEqual(ckpt_config.get('vocab_size'), 512)
        self.assertEqual(ckpt_config.get('embedding_dim'), 32)
        self.assertEqual(ckpt_config.get('max_context_length'), 64)
        self.assertEqual(ckpt_config.get('num_heads'), 4)
        self.assertEqual(ckpt_config.get('feed_forward_dim'), 128)
        self.assertEqual(ckpt_config.get('num_layers'), 2)

    def test_architecture_reconstruction_and_parameter_count(self):
        """Verify GPT-3 reconstructs from config and has exactly 60,800 parameters."""
        raw_config = load_config(self.config_path)
        model_cfg = raw_config['model']

        model = GPT(**model_cfg)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(param_count, 60800, f"Expected exactly 60,800 parameters, got {param_count}")

        # Verify state dict loads with strict=True
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.assertEqual(len(missing_keys), 0)
        self.assertEqual(len(unexpected_keys), 0)

        # All parameters must be finite
        for name, p in model.named_parameters():
            self.assertTrue(torch.all(torch.isfinite(p)), f"Non-finite weights in parameter {name}")

    def test_tokenizer_functionality(self):
        """Verify SubwordTokenizer loads and encodes/decodes correctly."""
        tokenizer = get_tokenizer('subword')
        tokenizer.load(self.tok_path)
        self.assertEqual(tokenizer.vocab_size, 512)

        prompt = "Once upon a time"
        tokens = tokenizer.encode(prompt)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(0 <= t < 512 for t in tokens))

        decoded = tokenizer.decode(tokens)
        self.assertIsInstance(decoded, str)
        self.assertIn("Once upon", decoded)

    def test_evaluation_reproduction_and_immutability(self):
        """Verify evaluation pipeline runs and preserves model weights identically."""
        raw_config = load_config(self.config_path)
        model_cfg = raw_config['model']
        training_cfg = raw_config['training']

        model = GPT(**model_cfg).to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        model.eval()

        # Clone weights to verify parameter immutability
        cloned_params = {n: p.clone() for n, p in model.named_parameters()}

        tokenizer = get_tokenizer('subword')
        tokenizer.load(self.tok_path)
        dataset = LanguageDataset(tokenizer, data_dir=os.path.join(self.project_root, 'data', 'processed'),
                                  context_length=model_cfg['max_context_length'])

        # Deterministic evaluation using seed 42
        torch.manual_seed(42)
        val_losses = []
        val_batches = 50
        with torch.no_grad():
            for _ in range(val_batches):
                x, y = dataset.get_batch('val', batch_size=training_cfg['batch_size'])
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        perplexity = math.exp(avg_val_loss)

        # Baseline evaluation criteria
        self.assertTrue(math.isfinite(avg_val_loss))
        self.assertTrue(math.isfinite(perplexity))
        self.assertGreater(perplexity, 0)

        # Check loss is in expected range (~3.90 +/- 0.05)
        self.assertAlmostEqual(avg_val_loss, 3.90, delta=0.05)
        self.assertAlmostEqual(perplexity, 49.5, delta=2.5)

        # Verify weights were not altered during evaluation
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(p, cloned_params[n]), f"Parameter {n} was modified during evaluation!")

    def test_generation_sample(self):
        """Verify model produces generated text from prompt."""
        raw_config = load_config(self.config_path)
        model = GPT(**raw_config['model']).to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        model.eval()

        tokenizer = get_tokenizer('subword')
        tokenizer.load(self.tok_path)

        prompt = "Once upon a time"
        torch.manual_seed(42)
        out = generate(model, tokenizer, prompt, max_new_tokens=15, temperature=0.8)
        self.assertTrue(len(out) > len(prompt))
        self.assertTrue(out.startswith("Once upon a time"))

if __name__ == '__main__':
    unittest.main()
