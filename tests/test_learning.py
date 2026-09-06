import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.device import get_device

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(3, 8)
        self.fc = nn.Linear(8, 3)
        
    def forward(self, x):
        return self.fc(torch.relu(self.emb(x)))

class TestLearning(unittest.TestCase):
    def setUp(self):
        # Set seed for reproducibility
        torch.manual_seed(42)
        
    def test_deterministic_learning(self):
        """Verify that a simple network can learn A->B, B->C, C->A."""
        device = get_device()
        model = TinyNet().to(device)
        
        # Mapping: A=0, B=1, C=2
        # Inputs: 0, 1, 2
        # Targets: 1, 2, 0
        X = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        Y = torch.tensor([1, 2, 0], dtype=torch.long, device=device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.5)
        
        # Initial loss
        model.eval()
        with torch.no_grad():
            initial_logits = model(X)
            initial_loss = criterion(initial_logits, Y).item()
            
        model.train()
        # Train for a few epochs
        for _ in range(200):
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, Y)
            loss.backward()
            optimizer.step()
            
        # Final loss
        model.eval()
        with torch.no_grad():
            final_logits = model(X)
            final_loss = criterion(final_logits, Y).item()
            preds = final_logits.argmax(dim=-1)
            
        # Verify loss decreased substantially
        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 0.1)
        
        # Verify predictions exactly match A->B, B->C, C->A
        self.assertEqual(preds[0].item(), 1) # A -> B
        self.assertEqual(preds[1].item(), 2) # B -> C
        self.assertEqual(preds[2].item(), 0) # C -> A

if __name__ == "__main__":
    unittest.main()
