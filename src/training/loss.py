import torch
import torch.nn as nn
import torch.nn.functional as F

def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for language modeling.
    
    Args:
        logits: Raw logits of shape [B, T, V]
        targets: Target token IDs of shape [B, T]
    Returns:
        Scalar cross-entropy loss.
    """
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    return F.cross_entropy(logits_flat, targets_flat)
