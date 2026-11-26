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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama-style)."""

    def __init__(self, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed.to(orig_dtype)) * self.weight
