import subprocess
import sys
import os
import torch
import shutil

def run_cmd(cmd, env=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, check=True)
    return result

def verify_frozen_baseline():
    print("--- Verifying frozen GPT-1 baseline ---")
    ckpt_path = "checkpoints/gpt1/gpt1_baseline.pt"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Frozen baseline checkpoint missing at {ckpt_path}")
        sys.exit(1)
        
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint['config']
    
    # Expected config
    expected_config = {
        'vocab_size': 89,
        'embedding_dim': 32,
        'max_context_length': 64,
        'num_heads': 4,
        'feed_forward_dim': 128,
        'num_layers': 2
    }
    
    for k, v in expected_config.items():
        if config.get(k) != v:
            print(f"ERROR: Baseline config mismatch for {k}. Expected {v}, got {config.get(k)}")
            sys.exit(1)
            
    # Verify parameter count
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.model.gpt import GPT
    model = GPT(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if param_count != 33305:
        print(f"ERROR: Baseline parameter count mismatch. Expected 33305, got {param_count}")
        sys.exit(1)
        
    print(f"OK: Frozen baseline intact (Params: {param_count}).\n")

def run_smoke_reproduction():
    print("--- Running GPT-1 Smoke Reproduction Pipeline ---")
    scratch_dir = "scratch/reproduce"
    data_dir = os.path.join(scratch_dir, "data")
    ckpt_dir = os.path.join(scratch_dir, "checkpoints")
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    results_path = os.path.join(scratch_dir, "results.md")
    
    # Clean scratch dir
    if os.path.exists(scratch_dir):
        shutil.rmtree(scratch_dir)
    os.makedirs(scratch_dir)
    
    # 1. Prepare data
    run_cmd([
        sys.executable, "src/data/prepare.py",
        "--num_samples", "5000",
        "--seed", "42",
        "--data_dir", data_dir
    ])
    
    # 2. Build tokenizer
    run_cmd([
        sys.executable, "scripts/build_tokenizer.py",
        "--data_dir", data_dir,
        "--out_path", tokenizer_path
    ])
    
    # 3. Train GPT-1 (Smoke test with 5 steps)
    run_cmd([
        sys.executable, "scripts/train_gpt1.py",
        "--config", "configs/gpt1.yaml",
        "--tokenizer", tokenizer_path,
        "--data_dir", data_dir,
        "--out_dir", ckpt_dir,
        "--results_path", results_path,
        "--steps", "5"
    ])
    
    smoke_ckpt_path = os.path.join(ckpt_dir, "gpt1_baseline.pt")
    if not os.path.exists(smoke_ckpt_path):
        print(f"ERROR: Smoke test failed to produce a checkpoint at {smoke_ckpt_path}")
        sys.exit(1)
        
    # Verify smoke parameter count
    checkpoint = torch.load(smoke_ckpt_path, map_location="cpu", weights_only=False)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.model.gpt import GPT
    model = GPT(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if param_count != 33305:
        print(f"ERROR: Smoke baseline parameter count mismatch. Expected 33305, got {param_count}")
        sys.exit(1)
        
    # 4. Evaluate GPT-1
    run_cmd([
        sys.executable, "scripts/evaluate_gpt1.py",
        "--checkpoint", smoke_ckpt_path,
        "--tokenizer", tokenizer_path,
        "--data_dir", data_dir,
        "--results_path", os.path.join(scratch_dir, "eval_results.md")
    ])
    
    # 5. Generate with CLI (chat.py)
    run_cmd([
        sys.executable, "scripts/chat.py",
        "--checkpoint", smoke_ckpt_path,
        "--tokenizer", tokenizer_path,
        "--config", "configs/gpt1.yaml",
        "--device", "cpu",
        "--prompt", "Once upon a time"
    ])
    
    print("\nSUCCESS: End-to-end GPT-1 reproduction smoke test completed.")

if __name__ == "__main__":
    verify_frozen_baseline()
    run_smoke_reproduction()
