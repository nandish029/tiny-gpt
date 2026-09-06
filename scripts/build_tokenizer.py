import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tokenizer.character import CharacterTokenizer
from src.data.dataset import TinyStoriesDataset

def main():
    print("Loading training dataset...")
    train_dataset = TinyStoriesDataset(split="train")
    
    print("Building tokenizer vocabulary from training data only...")
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(train_dataset.stories)
    
    save_path = "data/processed/tokenizer.json"
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
