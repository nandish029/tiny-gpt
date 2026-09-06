import torch
import torch.nn as nn
from src.model.transformer_block import TransformerBlock

class TransformerStack(nn.Module):
    """
    A configurable stack of TransformerBlock modules.
    """
    def __init__(self, embedding_dim: int, num_heads: int, feed_forward_dim: int, num_layers: int):
        super().__init__()
        
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if feed_forward_dim <= 0:
            raise ValueError(f"feed_forward_dim must be > 0, got {feed_forward_dim}")
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, feed_forward_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, D]
        Returns:
            Output tensor of shape [B, T, D]
        """
        for block in self.blocks:
            x = block(x)
        return x
