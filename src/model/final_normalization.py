import torch.nn as nn
from src.model.normalization import LayerNorm

class FinalLayerNorm(nn.Module):
    """
    Final LayerNorm applied after the Transformer stack.
    Reuses the existing LayerNorm component with its own independent parameters.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.ln = LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, D]
        Returns:
            Normalized tensor of shape [B, T, D]
        """
        return self.ln(x)
