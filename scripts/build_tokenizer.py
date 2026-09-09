import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tokenizer.character import CharacterTokenizer
from src.data.dataset import TinyStoriesDataset

def main():
    parser = argparse.ArgumentParser(description="Build Character Tokenizer")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing train.jsonl")
    parser.add_argument("--out_path", type=str, default="data/processed/tokenizer.json", help="Path to save tokenizer.json")
    args = parser.parse_args()

    print(f"Loading training dataset from {args.data_dir}...")
    train_dataset = TinyStoriesDataset(split="train", data_dir=args.data_dir)
    
    print("Building tokenizer vocabulary from training data only...")
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(train_dataset.stories)
    
    save_path = args.out_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    
    print(f"Tokenizer saved to {save_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    print("\n--- Manual Demonstration ---")
    sample_text = "Once upon a time,"
    print(f"Original text: '{sample_text}'")
    
    try:
        encoded = tokenizer.encode(sample_text)
        print(f"Encoded IDs: {encoded}")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded text: '{decoded}'")
        
        if sample_text == decoded:
            print("Round-trip validation: SUCCEEDED")
        else:
            print("Round-trip validation: FAILED")
    except ValueError as e:
        print(f"Error during encoding/decoding: {e}")

if __name__ == "__main__":
    main()
