import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization for GPT.
    Normalizes the final embedding dimension.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, D]
        Returns:
            Normalized tensor of shape [B, T, D]
        """
        return self.ln(x)
