import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.device import get_device

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(3, 8)
        self.fc = nn.Linear(8, 3)
        
    def forward(self, x):
        return self.fc(torch.relu(self.emb(x)))

def main():
    torch.manual_seed(42)
    device = get_device()
    print("=== Sanity Learning Check ===")
    print(f"Selected Device: {device}")
    
    model = TinyNet().to(device)
    
    # Mapping: A=0, B=1, C=2
    X = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    Y = torch.tensor([1, 2, 0], dtype=torch.long, device=device)
    chars = ['A', 'B', 'C']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    
    model.eval()
    with torch.no_grad():
        initial_logits = model(X)
        initial_loss = criterion(initial_logits, Y).item()
    
    print(f"\nInitial Loss: {initial_loss:.4f}")
    
    model.train()
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        final_logits = model(X)
        final_loss = criterion(final_logits, Y).item()
        preds = final_logits.argmax(dim=-1)
        
    print(f"Final Loss:   {final_loss:.4f}")
    print("\nPredictions:")
    for i in range(3):
        input_char = chars[X[i].item()]
        pred_char = chars[preds[i].item()]
        target_char = chars[Y[i].item()]
        print(f"{input_char} -> {pred_char} (Target: {target_char})")
        
if __name__ == "__main__":
    main()
