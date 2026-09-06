import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tokenizer.character import CharacterTokenizer
from src.data.language_dataset import LanguageDataset
from src.utils.device import get_device

def main():
    device = get_device()
    print("=== Language Dataset Batching Check ===")
    print(f"Selected Device: {device}")
    
    tokenizer = CharacterTokenizer()
    tokenizer.load("data/processed/tokenizer.json")
    
    # Use a small context length to make the manual print readable
    dataset = LanguageDataset(tokenizer, context_length=16)
    x, y = dataset.get_batch("train", batch_size=2)
    
    print(f"\nX shape: {list(x.shape)}")
    print(f"Y shape: {list(y.shape)}")
    
    print("\n--- First Sequence ---")
    x0 = x[0].tolist()
    y0 = y[0].tolist()
    
    print(f"Input IDs (X):  {x0}")
    print(f"Target IDs (Y): {y0}")
    
    decoded_x = tokenizer.decode(x0)
    decoded_y = tokenizer.decode(y0)
    
    print(f"\nDecoded Input (X):  '{decoded_x}'")
    print(f"Decoded Target (Y): '{decoded_y}'")
    
    print("\nObserve the exact 1-character shift to the left in the Target.")

if __name__ == "__main__":
    main()
