import os
import sys

# Make the project root importable when this file is run directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from src.model.gpt import GPT
from src.training.loss import language_model_loss
from src.training.optimizer import create_optimizer
from src.training.step import train_step
from src.utils.device import get_device


def main():
    device = get_device()

    print("=== CUDA Portability Validation ===")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Selected device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = GPT(
        vocab_size=89,
        embedding_dim=32,
        max_context_length=64,
        num_heads=4,
        feed_forward_dim=128,
        num_layers=2,
    ).to(device)

    print("Model device:", next(model.parameters()).device)

    x = torch.randint(0, 89, (2, 16), device=device)
    y = torch.randint(0, 89, (2, 16), device=device)

    logits = model(x)
    loss = language_model_loss(logits, y)

    print("Logits shape:", logits.shape)
    print("Logits device:", logits.device)
    print("Initial loss:", loss.item())

    before = [
        p.detach().clone()
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = create_optimizer(model, learning_rate=3e-4)
    train_loss = train_step(model, optimizer, x, y)

    after = [
        p.detach()
        for p in model.parameters()
        if p.requires_grad
    ]

    changed = any(
        not torch.equal(a, b)
        for a, b in zip(before, after)
    )

    finite = (
        torch.isfinite(logits).all().item()
        and torch.isfinite(loss).item()
        and torch.isfinite(train_loss).item()
        and all(torch.isfinite(p).all().item() for p in model.parameters())
    )

    same_device = (
        next(model.parameters()).device == x.device
        and x.device == y.device
        and logits.device == x.device
    )

    print("Device consistency:", "PASS" if same_device else "FAIL")
    print("Forward pass:", "PASS" if logits.shape == (2, 16, 89) else "FAIL")
    print("Loss:", "PASS" if torch.isfinite(loss).item() else "FAIL")
    print("Training step:", "PASS" if torch.isfinite(train_loss).item() else "FAIL")
    print("Parameter update:", "PASS" if changed else "FAIL")
    print("Numerical validity:", "PASS" if finite else "FAIL")

    final = same_device and changed and finite

    print("FINAL RESULT:", "PASS" if final else "FAIL")


if __name__ == "__main__":
    main()
