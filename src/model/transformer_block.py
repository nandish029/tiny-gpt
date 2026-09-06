import torch
import torch.nn as nn
from src.model.attention import MultiHeadSelfAttention
from src.model.feed_forward import FeedForward
from src.model.normalization import LayerNorm
from src.model.residual import ResidualAdd

class TransformerBlock(nn.Module):
    """
    A single GPT-style Pre-LayerNorm Transformer Block.
    """
    def __init__(self, embedding_dim: int, num_heads: int, feed_forward_dim: int):
        super().__init__()
        
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        if feed_forward_dim <= 0:
            raise ValueError(f"feed_forward_dim must be > 0, got {feed_forward_dim}")
            
        self.ln_1 = LayerNorm(embedding_dim)
        self.attn = MultiHeadSelfAttention(embedding_dim, num_heads, causal=True)
        self.res_1 = ResidualAdd()
        
        self.ln_2 = LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, feed_forward_dim)
        self.res_2 = ResidualAdd()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, D]
        Returns:
            Output tensor of shape [B, T, D]
        """
        # First sublayer: LayerNorm -> Attention -> Residual
        norm1 = self.ln_1(x)
        attention = self.attn(norm1)
        x_attn = self.res_1(x, attention)
        
        # Second sublayer: LayerNorm -> FeedForward -> Residual
        norm2 = self.ln_2(x_attn)
        ffn = self.ffn(norm2)
        out = self.res_2(x_attn, ffn)
        
        return out
