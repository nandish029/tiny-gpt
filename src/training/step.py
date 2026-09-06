import torch
import torch.nn as nn
from src.training.loss import language_model_loss

def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
               input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Perform a single training step.
    
    Args:
        model: The GPT model.
        optimizer: The optimizer.
        input_ids: Input token IDs of shape [B, T].
        targets: Target token IDs of shape [B, T].
    Returns:
        Scalar loss tensor.
    """
    model.train()
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = language_model_loss(logits, targets)
    loss.backward()
    optimizer.step()
    return loss

def validation_step(model: nn.Module, input_ids: torch.Tensor,
                    targets: torch.Tensor) -> torch.Tensor:
    """
    Perform a single validation step (no parameter updates).
    
    Args:
        model: The GPT model.
        input_ids: Input token IDs of shape [B, T].
        targets: Target token IDs of shape [B, T].
    Returns:
        Scalar loss tensor.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        loss = language_model_loss(logits, targets)
    return loss
