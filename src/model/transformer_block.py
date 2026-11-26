import torch
import torch.nn as nn
from src.model.attention import MultiHeadSelfAttention
from src.model.feed_forward import FeedForward, SwiGLU
from src.model.normalization import LayerNorm, RMSNorm
from src.model.residual import ResidualAdd

class TransformerBlock(nn.Module):
    """
    A single GPT-style Pre-Norm Transformer Block.
    Supports LayerNorm/RMSNorm, GELU/SwiGLU, and learned/RoPE positional encoding.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        norm_type: str = "layernorm",
        ffn_type: str = "gelu",
        pos_encoding: str = "learned",
        max_context_length: int = 128,
    ):
        super().__init__()
        
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        if feed_forward_dim <= 0:
            raise ValueError(f"feed_forward_dim must be > 0, got {feed_forward_dim}")
            
        norm_cls = RMSNorm if norm_type == "rmsnorm" else LayerNorm
        ffn_cls = SwiGLU if ffn_type == "swiglu" else FeedForward

        self.ln_1 = norm_cls(embedding_dim)
        self.attn = MultiHeadSelfAttention(
            embedding_dim,
            num_heads,
            causal=True,
            pos_encoding=pos_encoding,
            max_context_length=max_context_length,
        )
        self.res_1 = ResidualAdd()
        
        self.ln_2 = norm_cls(embedding_dim)
        self.ffn = ffn_cls(embedding_dim, feed_forward_dim)
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
