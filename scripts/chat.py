import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.gpt import GPT
from src.tokenizer.character import CharacterTokenizer
from src.generation.generate import generate
from src.utils.device import resolve_device
from src.utils.config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Tiny GPT Command Line Interface")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer definition (.json)")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration (.yaml)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device (auto, cpu, cuda)")
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt for one-shot generation. If omitted, enters interactive mode.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature (must be > 0.0)")
    parser.add_argument("--max-new-tokens", type=int, default=200, dest="max_new_tokens", help="Maximum number of tokens to generate (must be > 0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    return parser.parse_args()

def validate_args(args):
    if args.temperature <= 0.0 or not torch.isfinite(torch.tensor(args.temperature)):
        raise ValueError(f"Temperature must be a finite positive value, got {args.temperature}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"Max new tokens must be a positive integer, got {args.max_new_tokens}")
        
    for path, name in [(args.checkpoint, "Checkpoint"), (args.tokenizer, "Tokenizer"), (args.config, "Config")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found at {path}")

def validate_architecture(yaml_config, checkpoint_config):
    """Compares the YAML configuration to the checkpoint's internal configuration to prevent obscure tensor shape errors."""
    keys_to_check = [
        'vocab_size', 'embedding_dim', 'max_context_length', 
        'num_heads', 'feed_forward_dim', 'num_layers'
    ]
    
    for key in keys_to_check:
        yaml_val = yaml_config.get(key)
        ckpt_val = checkpoint_config.get(key)
        if yaml_val != ckpt_val:
            raise ValueError(
                f"Architecture mismatch for '{key}': Config specifies {yaml_val}, "
                f"but checkpoint requires {ckpt_val}."
            )

def main():
    args = parse_args()
    validate_args(args)
    
    # 1. Resolve Device
    device = resolve_device(args.device)
    print(f"Device: {device.type}")
    
    # 2. Load Config
    full_yaml_config = load_config(args.config)
    model_config = full_yaml_config.get('model', {})
    
    # 3. Load Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_config = checkpoint.get('config', {})
    
    # 4. Validate Architecture
    validate_architecture(model_config, checkpoint_config)
    
    # 5. Construct Model
    model = GPT(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
    
    # 6. Load Tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.load(args.tokenizer)
    
    # Setup Generation
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # 7. Execute Mode
    if args.prompt is not None:
        # One-shot mode
        with torch.no_grad():
            output = generate(
                model, 
                tokenizer, 
                args.prompt, 
                max_new_tokens=args.max_new_tokens, 
                temperature=args.temperature,
                generator=generator
            )
        print(output)
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_input = input("You: ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue
                
                with torch.no_grad():
                    output = generate(
                        model, 
                        tokenizer, 
                        user_input, 
                        max_new_tokens=args.max_new_tokens, 
                        temperature=args.temperature,
                        generator=generator
                    )
                # We print only the GPT portion of the response or the whole output
                print(f"GPT: {output}")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
