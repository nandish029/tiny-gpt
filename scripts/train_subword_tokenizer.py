import argparse
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tokenizer.factory import get_tokenizer
from src.tokenizer.subword import SubwordTokenizer

def text_generator(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Verify it's explicitly NOT val.jsonl
    if "val.jsonl" in filepath:
        raise ValueError("Validation data MUST NOT be used for tokenizer training.")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                yield item['text']

def main():
    parser = argparse.ArgumentParser(description="Train Subword Tokenizer")
    parser.add_argument("--data", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--out", type=str, default="data/processed/gpt3_tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--byte_level", action="store_true", help="Use modern Byte-Level BPE")
    args = parser.parse_args()

    print(f"Initializing subword tokenizer (vocab={args.vocab_size}, byte_level={args.byte_level})...")
    tokenizer = SubwordTokenizer(byte_level=args.byte_level) if args.byte_level else get_tokenizer("subword")
    
    print(f"Training tokenizer on {args.data}...")
    tokenizer.build_vocab(text_generator(args.data), vocab_size=args.vocab_size)
    
    print(f"Actual vocabulary size created: {tokenizer.vocab_size}")
    
    tokenizer.save(args.out)
    print(f"Tokenizer saved to {args.out}")

if __name__ == "__main__":
    main()
