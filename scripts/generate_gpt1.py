import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.gpt import GPT
from src.tokenizer.character import CharacterTokenizer
from src.generation.generate import generate
from src.utils.device import get_device

def main():
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = CharacterTokenizer()
    tokenizer.load("data/processed/tokenizer.json")
    
    checkpoint_path = "checkpoints/gpt1/gpt1_baseline.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print("Reconstructing GPT-1...")
    model = GPT(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    prompts = [
        "Once upon a time",
        "The little girl",
        "One day"
    ]
    
    temperature = 0.8
    max_new_tokens = 200
    
    print("\n" + "="*50)
    print(f"GENERATION SETTINGS:")
    print(f"Temperature: {temperature}")
    print(f"Max new tokens: {max_new_tokens}")
    print("="*50 + "\n")
    
    # Use a fixed generator for reproducible samples across runs if needed, 
    # but not strictly required for the script output.
    generator = torch.Generator(device=device).manual_seed(42)
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        print("-" * 20)
        output = generate(
            model, 
            tokenizer, 
            prompt, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            generator=generator
        )
        print(output)
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
