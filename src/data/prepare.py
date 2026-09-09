import os
import json
import random
from datasets import load_dataset

def prepare_data(num_samples=5000, seed=42, data_dir="data/processed"):
    """
    Downloads a deterministic subset of TinyStories via streaming,
    cleans it, and creates a train/validation split.
    """
    random.seed(seed)
    
    print(f"Loading dataset in streaming mode...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    iterator = iter(ds)
    
    processed_data = []
    seen = set()
    
    print(f"Collecting {num_samples} unique samples...")
    while len(processed_data) < num_samples:
        try:
            item = next(iterator)
            text = item.get('text', '')
            
            # Remove empty samples
            if not text or not text.strip():
                continue
                
            # Normalize whitespace
            text = " ".join(text.split())
            
            # Avoid duplicates to ensure strict train/val separation
            if text in seen:
                continue
                
            seen.add(text)
            processed_data.append({"text": text})
        except StopIteration:
            break
            
    # Deterministic shuffle
    random.shuffle(processed_data)
    
    # Train / Validation split (90% / 10%)
    split_idx = int(len(processed_data) * 0.9)
    train_data = processed_data[:split_idx]
    valid_data = processed_data[split_idx:]
    
    # Save processed data
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(valid_path, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(train_data)} train samples and {len(valid_data)} valid samples to {data_dir}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    args = parser.parse_args()
    
    prepare_data(num_samples=args.num_samples, seed=args.seed, data_dir=args.data_dir)
