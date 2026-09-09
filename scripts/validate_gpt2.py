import subprocess
import sys
import os
import shutil

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    print("--- Running GPT-2 Smoke Test & Resume Validation ---")
    scratch_dir = "scratch/gpt2_smoke"
    data_dir = os.path.join(scratch_dir, "data")
    ckpt_dir = os.path.join(scratch_dir, "checkpoints")
    results_path = os.path.join(scratch_dir, "results.md")
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    
    if os.path.exists(scratch_dir):
        shutil.rmtree(scratch_dir)
    os.makedirs(scratch_dir)
    
    # 1. Prepare data (use enough samples to get full vocab)
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
    
    # 3. Train GPT-2 (Smoke test with 3 steps)
    run_cmd([
        sys.executable, "scripts/train_gpt1.py",
        "--config", "configs/gpt2.yaml",
        "--tokenizer", tokenizer_path,
        "--data_dir", data_dir,
        "--out_dir", ckpt_dir,
        "--results_path", results_path,
        "--steps", "3"
    ])
    
    ckpt_path = os.path.join(ckpt_dir, "gpt2_baseline.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Failed to save checkpoint at {ckpt_path}")
        sys.exit(1)
        
    print("\n--- Testing Resume Functionality ---")
    # 4. Resume training (steps 4 and 5)
    run_cmd([
        sys.executable, "scripts/train_gpt1.py",
        "--config", "configs/gpt2.yaml",
        "--tokenizer", tokenizer_path,
        "--data_dir", data_dir,
        "--out_dir", ckpt_dir,
        "--results_path", results_path,
        "--steps", "5",
        "--resume", ckpt_path
    ])
    
    # 5. Evaluate GPT-2
    print("\n--- Evaluating GPT-2 Smoke Checkpoint ---")
    run_cmd([
        sys.executable, "scripts/evaluate_gpt1.py",
        "--checkpoint", ckpt_path,
        "--tokenizer", tokenizer_path,
        "--data_dir", data_dir,
        "--results_path", os.path.join(scratch_dir, "eval_results.md")
    ])
    
    print("\nSUCCESS: GPT-2 Smoke Test & Resume Validation completed.")

if __name__ == "__main__":
    main()
