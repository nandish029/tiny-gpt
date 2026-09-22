import unittest
import torch
import torch.nn.functional as F
import os
import sys
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.gpt import GPT
from src.tokenizer.character import CharacterTokenizer
from src.generation.generate import generate

class TestGeneration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        
        # Tiny configuration for fast testing
        self.vocab_size = 10
        self.embed_dim = 16
        self.max_len = 8
        self.num_heads = 2
        self.ff_dim = 32
        self.num_layers = 1
        
        self.model = GPT(
            self.vocab_size, self.embed_dim, self.max_len,
            self.num_heads, self.ff_dim, self.num_layers
        ).to(self.device)
        self.model.eval()
        
        # Simple tokenizer (0-9)
        self.tokenizer = CharacterTokenizer()
        self.tokenizer.char_to_id = {str(i): i for i in range(10)}
        self.tokenizer.id_to_char = {i: str(i) for i in range(10)}
        self.tokenizer.vocab_size = 10

    def test_basic_generation(self):
        # 1. BASIC GENERATION
        prompt = "123"
        output = generate(self.model, self.tokenizer, prompt, max_new_tokens=2)
        self.assertIsInstance(output, str)

    def test_prompt_preservation(self):
        # 2. PROMPT PRESERVATION
        prompt = "123"
        output = generate(self.model, self.tokenizer, prompt, max_new_tokens=5)
        self.assertTrue(output.startswith(prompt))

    def test_length(self):
        # 3. LENGTH
        prompt = "123"
        max_new_tokens = 4
        output = generate(self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens)
        self.assertEqual(len(output), len(prompt) + max_new_tokens)

    def test_temperature_validation(self):
        # 4. TEMPERATURE VALIDATION
        prompt = "123"
        
        # Valid temperature
        _ = generate(self.model, self.tokenizer, prompt, max_new_tokens=1, temperature=0.5)
        
        # Zero temperature
        with self.assertRaises(ValueError):
            generate(self.model, self.tokenizer, prompt, max_new_tokens=1, temperature=0.0)
            
        # Negative temperature
        with self.assertRaises(ValueError):
            generate(self.model, self.tokenizer, prompt, max_new_tokens=1, temperature=-1.0)

    def test_context_window(self):
        # 5. CONTEXT WINDOW
        # Max context is 8. Prompt is 5, generate 6 -> total 11.
        prompt = "12345"
        # Since it successfully generates 6 tokens, it means the context truncation worked.
        # Otherwise, the model would throw a ValueError because sequence length exceeds max_context_length.
        output = generate(self.model, self.tokenizer, prompt, max_new_tokens=6)
        self.assertEqual(len(output), len(prompt) + 6)
        
    def test_no_gradients(self):
        # 6. NO GRADIENTS
        prompt = "123"
        _ = generate(self.model, self.tokenizer, prompt, max_new_tokens=2)
        
        # Ensure no gradients are accumulated
        for param in self.model.parameters():
            self.assertIsNone(param.grad)

    def test_parameters_unchanged(self):
        # 7. PARAMETERS UNCHANGED
        prompt = "123"
        params_before = {name: param.clone() for name, param in self.model.named_parameters()}
        
        _ = generate(self.model, self.tokenizer, prompt, max_new_tokens=5)
        
        for name, param in self.model.named_parameters():
            self.assertTrue(torch.equal(param, params_before[name]))

    def test_eval_mode(self):
        # 8. EVAL MODE
        self.model.train() # Set to train
        prompt = "123"
        _ = generate(self.model, self.tokenizer, prompt, max_new_tokens=2)
        
        # Generate should set to eval
        self.assertFalse(self.model.training)

    def test_deterministic_sampling(self):
        # 9. DETERMINISTIC SAMPLING
        prompt = "123"
        
        gen1 = torch.Generator(device=self.device).manual_seed(123)
        out1 = generate(self.model, self.tokenizer, prompt, max_new_tokens=10, generator=gen1)
        
        gen2 = torch.Generator(device=self.device).manual_seed(123)
        out2 = generate(self.model, self.tokenizer, prompt, max_new_tokens=10, generator=gen2)
        
        self.assertEqual(out1, out2)

    def test_token_validity(self):
        # 10. TOKEN VALIDITY
        prompt = "123"
        output = generate(self.model, self.tokenizer, prompt, max_new_tokens=50)
        
        # If decode() didn't fail, they are valid vocabulary IDs.
        # But we can also check the characters.
        for char in output:
            self.assertIn(char, self.tokenizer.char_to_id)

    def test_numerical_validity(self):
        # 11. NUMERICAL VALIDITY
        prompt = "123"
        # Just generate, if it doesn't crash from NaN in multinomial, it's valid.
        try:
            _ = generate(self.model, self.tokenizer, prompt, max_new_tokens=5)
        except RuntimeError as e:
            self.fail(f"Generation failed with: {e}")

    def test_device_cuda(self):
        # 12. DEVICE
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        
        model = GPT(
            self.vocab_size, self.embed_dim, self.max_len,
            self.num_heads, self.ff_dim, self.num_layers
        ).to('cuda')
        model.eval()
        
        prompt = "123"
        gen = torch.Generator(device='cuda').manual_seed(42)
        out = generate(model, self.tokenizer, prompt, max_new_tokens=2, generator=gen)
        self.assertIsInstance(out, str)

    def test_checkpoint_load(self):
        # 13. CHECKPOINT LOAD TEST
        ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'gpt1', 'gpt1_baseline.pt')
        if not os.path.exists(ckpt_path):
            self.skipTest("Checkpoint not found.")
            
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        config = checkpoint['config']
        model_kwargs = {k: v for k, v in config.items() if k != 'tokenizer_type'}
        
        model = GPT(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        tokenizer = CharacterTokenizer()
        tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'tokenizer.json'))
        
        prompt = "Once"
        out = generate(model, tokenizer, prompt, max_new_tokens=5)
        self.assertIsInstance(out, str)
        self.assertEqual(len(out), len(prompt) + 5)

    def test_independent_reference(self):
        # INDEPENDENT REFERENCE
        prompt = "12"
        max_new_tokens = 2
        temperature = 0.5
        
        gen = torch.Generator(device=self.device).manual_seed(999)
        output = generate(self.model, self.tokenizer, prompt, max_new_tokens, temperature, generator=gen)
        
        # Manual reference computation
        gen_ref = torch.Generator(device=self.device).manual_seed(999)
        input_ids = torch.tensor([[1, 2]], dtype=torch.long, device=self.device)
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(input_ids)
                last_logits = logits[0, -1, :]
                scaled = last_logits / temperature
                probs = F.softmax(scaled, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1, generator=gen_ref)
                input_ids = torch.cat([input_ids, next_tok.unsqueeze(0)], dim=1)
                
        ref_output = "".join(self.tokenizer.id_to_char[i.item()] for i in input_ids[0])
        self.assertEqual(output, ref_output)

if __name__ == "__main__":
    unittest.main()
