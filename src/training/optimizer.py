import torch
import torch.nn as nn

def create_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer for the model.
    
    Args:
        model: The model whose parameters to optimize.
        learning_rate: Learning rate for the optimizer.
    Returns:
        AdamW optimizer.
    """
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)
