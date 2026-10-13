import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.device import resolve_device
from src.utils.config import load_config
from src.tokenizer.factory import get_tokenizer
from src.model.gpt import GPT

def verify_model(model_name: str, device: str = "cpu"):
    """Verifies a model checkpoint and its corresponding tokenizer."""
    print(f"=== Verifying Model: {model_name} ===")

    # 1. Paths
    config_path = f"configs/{model_name}.yaml"
    ckpt_path = f"checkpoints/{model_name}/{model_name}_baseline.pt"

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    model_cfg = cfg['model']
    expected_tokenizer = "subword" if model_name == "gpt3" else "char"
    expected_tokenizer_path = f"data/processed/{model_name}_tokenizer.json" if model_name == "gpt3" else "data/processed/tokenizer.json"

    # 2. Checkpoint exists
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}. Artifact not published yet or not downloaded.")
        sys.exit(1)

    print(f"✓ Checkpoint found: {ckpt_path}")

    # 3. Checkpoint loads
    try:
        resolved_device = resolve_device(device)
        checkpoint = torch.load(ckpt_path, map_location=resolved_device, weights_only=False)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # 4. Checkpoint metadata
    if 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
        print("Error: Missing config or model_state_dict in checkpoint")
        sys.exit(1)

    ckpt_config = checkpoint['config']
    print("✓ Checkpoint metadata structure valid")

    # 5. Tokenizer exists and loads
    if not os.path.exists(expected_tokenizer_path):
        print(f"Error: Tokenizer not found at {expected_tokenizer_path}")
        sys.exit(1)

    try:
        tokenizer = get_tokenizer(expected_tokenizer)
        tokenizer.load(expected_tokenizer_path)
        print(f"✓ Tokenizer loaded successfully ({expected_tokenizer})")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    # 6. Tokenizer consistency
    if tokenizer.vocab_size != ckpt_config.get('vocab_size') or tokenizer.vocab_size != model_cfg.get('vocab_size', ckpt_config.get('vocab_size')):
         print(f"Error: Vocab size mismatch. Tokenizer={tokenizer.vocab_size}, Checkpoint={ckpt_config.get('vocab_size')}, Config={model_cfg.get('vocab_size')}")
         sys.exit(1)
    print(f"✓ Vocabulary size consistent: {tokenizer.vocab_size}")

    # 7. Model Architecture Matches
    for key in ['embedding_dim', 'max_context_length', 'num_heads', 'feed_forward_dim', 'num_layers']:
        if ckpt_config.get(key) != model_cfg.get(key, ckpt_config.get(key)):
            print(f"Error: Architecture mismatch on {key}. Config={model_cfg.get(key)}, Checkpoint={ckpt_config.get(key)}")
            sys.exit(1)

    print("✓ Architecture configuration matches")

    # 8. Instantiating Expected Model from Repository Configuration
    try:
        # Some configs might miss vocab_size if they expect it dynamically,
        # but in our repo configs it's either present or we inject tokenizer vocab size
        if 'vocab_size' not in model_cfg:
            model_cfg['vocab_size'] = tokenizer.vocab_size
        expected_model = GPT(**model_cfg)
        expected_param_count = sum(p.numel() for p in expected_model.parameters() if p.requires_grad)
        print(f"✓ Expected architecture dynamically derived ({expected_param_count} parameters)")
    except Exception as e:
        print(f"Error instantiating expected model from repository config: {e}")
        sys.exit(1)

    # 9. Model instantiates and loads
    try:
        model_kwargs = {k: v for k, v in ckpt_config.items() if k != 'tokenizer_type'}
        model = GPT(**model_kwargs).to(resolved_device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ Model instantiated and state dict loaded")
    except Exception as e:
        print(f"Error instantiating model: {e}")
        sys.exit(1)

    # 10. Parameter count verification
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if param_count != expected_param_count:
        print(f"Error: Parameter count mismatch. Expected {expected_param_count} from config, but checkpoint has {param_count}.")
        sys.exit(1)
    print(f"✓ Parameter count strictly verified: {param_count}")

    # 10. Forward Pass Works
    try:
        dummy_input = torch.randint(0, tokenizer.vocab_size, (2, ckpt_config['max_context_length']), device=resolved_device)
        with torch.no_grad():
            logits = model(dummy_input)

        if logits.shape != (2, ckpt_config['max_context_length'], tokenizer.vocab_size):
             print(f"Error: Output shape mismatch. Got {logits.shape}")
             sys.exit(1)

        if not torch.all(torch.isfinite(logits)):
            print("Error: Output contains non-finite values (NaN/Inf)")
            sys.exit(1)

        print("✓ Forward pass successful, outputs are finite and correct shape")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        sys.exit(1)

    # 11. Generation Works
    try:
        from src.generation.generate import generate
        out = generate(model, tokenizer, "Test", max_new_tokens=2, temperature=1.0)
        if len(out) == 0:
            print("Error: Generation produced empty output")
            sys.exit(1)
        print("✓ Generation successful")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

    # 12. Training step metadata
    step = checkpoint.get('step', -1)
    if step <= 0:
        print(f"Error: Invalid training step metadata: {step}")
        sys.exit(1)
    print(f"✓ Training metadata valid (Step: {step})")

    print(f"\n✅ {model_name.upper()} Model successfully verified on {resolved_device}.")

def main():
    parser = argparse.ArgumentParser(description="Verify Canonical GPT Model")
    parser.add_argument("--model", type=str, required=True, choices=["gpt1", "gpt2", "gpt3"], help="Model to verify")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="cpu", help="Device to use for verification")
    args = parser.parse_args()
    verify_model(args.model, args.device)

if __name__ == "__main__":
    main()
