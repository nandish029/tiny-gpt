import unittest
import os
import yaml

class TestConfigConsistency(unittest.TestCase):
    """
    Ensures that GPT-1 and GPT-2 architectural parameters are absolutely identical.
    GPT-2 is a controlled training experiment over GPT-1.
    """
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        self.gpt1_path = os.path.join(self.project_root, 'configs', 'gpt1.yaml')
        self.gpt2_path = os.path.join(self.project_root, 'configs', 'gpt2.yaml')
        
    def test_architectures_match_exactly(self):
        # Allow missing GPT-2 config if it hasn't been created yet, but in this checkpoint
        # it MUST exist because the user explicitly approved it.
        self.assertTrue(os.path.exists(self.gpt1_path), "GPT-1 config missing")
        self.assertTrue(os.path.exists(self.gpt2_path), "GPT-2 config missing")
        
        with open(self.gpt1_path, 'r') as f:
            gpt1 = yaml.safe_load(f)
            
        with open(self.gpt2_path, 'r') as f:
            gpt2 = yaml.safe_load(f)
            
        architecture_keys = [
            'vocab_size',
            'max_context_length',
            'embedding_dim',
            'num_heads',
            'feed_forward_dim',
            'num_layers'
        ]
        
        for key in architecture_keys:
            self.assertIn(key, gpt1['model'])
            self.assertIn(key, gpt2['model'])
            self.assertEqual(
                gpt1['model'][key], gpt2['model'][key],
                f"Architecture mismatch for {key}: GPT-1={gpt1['model'][key]}, GPT-2={gpt2['model'][key]}"
            )
            
    def test_gpt2_is_longer_training(self):
        with open(self.gpt1_path, 'r') as f:
            gpt1 = yaml.safe_load(f)
            
        with open(self.gpt2_path, 'r') as f:
            gpt2 = yaml.safe_load(f)
            
        # GPT-1 baseline was 1000 steps.
        # GPT-2 training is 5000 steps.
        self.assertEqual(gpt1['training'].get('train_steps', 1000), 1000)
        self.assertEqual(gpt2['training'].get('train_steps', 5000), 5000)

if __name__ == "__main__":
    unittest.main()
