import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset import TinyStoriesDataset

def inspect():
    try:
        train_ds = TinyStoriesDataset(split="train")
        valid_ds = TinyStoriesDataset(split="val")
    except FileNotFoundError as e:
        print(e)
        return

    print("=== Dataset Inspection ===")
    print(f"Training documents: {len(train_ds)}")
    print(f"Validation documents: {len(valid_ds)}")
    
    all_stories = train_ds.stories + valid_ds.stories
    total_chars = sum(len(s) for s in all_stories)
    lengths = [len(s) for s in all_stories]
    
    print(f"Total characters: {total_chars}")
    print(f"Min document length: {min(lengths)} characters")
    print(f"Max document length: {max(lengths)} characters")
    print(f"Average document length: {total_chars / len(all_stories):.2f} characters")
    
    print("\n--- Example Story 1 ---")
    print(all_stories[0][:500] + ("..." if len(all_stories[0]) > 500 else ""))
    
    print("\n--- Example Story 2 ---")
    print(all_stories[1][:500] + ("..." if len(all_stories[1]) > 500 else ""))

if __name__ == "__main__":
    inspect()
